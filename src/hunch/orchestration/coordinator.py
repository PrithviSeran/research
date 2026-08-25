"""Compact Qwen coordinator exposing only the private ``delegate`` tool."""
import json
import threading
import time
import uuid
from dataclasses import dataclass

from ..agent import AgentResult
from ..backends.ollama import OllamaBackend, OllamaSessionConfig, OllamaTool
from .contracts import DelegationRequest
from .orchestrator import SpecialistOrchestrator
from .specialists import SPECIALISTS


COORDINATOR_PROMPT = """You are a compact routing coordinator.
You cannot operate the Mac directly. Your only tool is delegate.
Choose the narrowest specialist that can complete the user task:
- applescript: scriptable Notes, Reminders, Calendar, and Mail operations; it can
  also enumerate files with a bounded `do shell script`, so prefer it for a task
  that combines file discovery with one of those scriptable applications
- native: files, clipboard, app lifecycle, and direct OS operations
- ax: native application accessibility-tree interaction
- web: browser or Electron work through CDP

Delegate a bounded objective with observable acceptance criteria. Do not put
credentials, clipboard contents, DOM/AX contents, or invented metadata in context.
Use at most the available delegation budget. A `verified_complete` receipt is the
only deterministic success. If a receipt says `verification_failed`, delegate one
bounded repair (normally to the same specialist) using the original acceptance
criteria. If it says `blocked`, use a genuinely different capable domain or report
blocked. Never claim success from `reported_complete`, `blocked`, or a failed
verification. Multiple serial delegations are allowed when the task truly spans
domains; never run specialists concurrently.
"""

DELEGATE_TOOL = OllamaTool.create("delegate", "Run one bounded specialist episode", {
    "type": "object",
    "properties": {
        "specialist": {"type": "string", "enum": [kind.value for kind in SPECIALISTS]},
        "objective": {"type": "string"},
        "context": {
            "type": "object",
            "description": (
                "Optional bounded scalar hints. Allowed keys: app, account, target, "
                "source, destination, note_title, list_name, source_directory, "
                "folder_path, window, url, service. Omit context when the objective "
                "already contains the information."),
            "propertyNames": {"enum": [
                "app", "account", "target", "source", "destination",
                "note_title", "list_name", "source_directory", "folder_path",
                "window", "url", "service"]},
            "additionalProperties": {"type": ["string", "number", "boolean", "null"]},
        },
        "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["specialist", "objective"],
})


@dataclass(frozen=True)
class CoordinatorResult:
    invocation_id: str
    trace_id: str
    result: AgentResult
    receipts: tuple
    usage: dict


def aggregate_usage(coordinator_usage, receipts):
    child_usage = [dict(receipt.usage) for receipt in receipts]
    return {
        "coordinator": dict(coordinator_usage or {}),
        "specialists": child_usage,
        "total_input_tokens": int((coordinator_usage or {}).get("input_tokens", 0))
                              + sum(int(item.get("input_tokens", 0)) for item in child_usage),
        "total_output_tokens": int((coordinator_usage or {}).get("output_tokens", 0))
                               + sum(int(item.get("output_tokens", 0)) for item in child_usage),
        "peak_context_chars": max(
            [int((coordinator_usage or {}).get("max_context_chars", 0))]
            + [int(item.get("max_context_chars", 0)) for item in child_usage]),
    }


class Coordinator:
    def __init__(self, orchestrator, *, session_factory=None, max_turns=8):
        if not isinstance(orchestrator, SpecialistOrchestrator):
            raise TypeError("orchestrator must be a SpecialistOrchestrator")
        self.orchestrator = orchestrator
        self.session_factory = session_factory or self._default_session_factory
        self.max_turns = max_turns
        self._active_lock = threading.Lock()
        self._active = None

    def _default_session_factory(self, config, dispatcher):
        return OllamaBackend(self.orchestrator.hunch, session_config=config,
                             dispatcher=dispatcher)

    def interrupt(self):
        self.orchestrator.interrupt()
        with self._active_lock:
            active = self._active
        if active is not None:
            active.interrupt()

    def run(self, task, *, deadline=None, on_event=None):
        deadline = deadline if deadline is not None else time.monotonic() + 900
        state = self.orchestrator.new_run(task, deadline=deadline)
        invocation_id = f"c-{uuid.uuid4().hex}"
        trace_id = f"ct-{uuid.uuid4().hex}"
        coordinator_events = []

        def emit(kind, data):
            coordinator_events.append({"kind": kind, "data": data,
                                       "invocation_id": invocation_id})
            if on_event is not None:
                on_event(kind, data)

        def delegate(name, arguments):
            if name != "delegate":
                return f"error: coordinator has no tool {name!r}", True
            try:
                kind = arguments.get("specialist", "")
                spec = SPECIALISTS[next(key for key in SPECIALISTS if key.value == kind)]
                request = DelegationRequest.create(
                    kind, arguments.get("objective", ""), arguments.get("context") or {},
                    arguments.get("acceptance_criteria") or (),
                    allowed_context_keys=spec.allowed_context_keys)
                receipt = self.orchestrator.run_specialist(state, request)
                return receipt.to_json(), False
            except (KeyError, StopIteration, TypeError, ValueError) as exc:
                return json.dumps({"status": "failed", "error_class": type(exc).__name__,
                                   "summary": ("invalid delegation request; use only "
                                               "documented context keys or omit context")},
                                  sort_keys=True, separators=(",", ":")), False

        config = OllamaSessionConfig.from_environment(
            role="coordinator", system_prompt=COORDINATOR_PROMPT,
            tool_names=frozenset(), allow_delegation=True,
            internal_tools=(DELEGATE_TOOL,))
        backend = self.session_factory(config, delegate)
        with self._active_lock:
            self._active = backend
        try:
            result = backend.run(task, max_turns=self.max_turns, on_event=emit,
                                 deadline=deadline)
        finally:
            coordinator_events.append({"kind": "conversation", "data": {
                "system": getattr(backend, "last_system", ""),
                "messages": list(getattr(backend, "messages", []))},
                "invocation_id": invocation_id})
            backend.close()
            with self._active_lock:
                self._active = None
        self.orchestrator.trace_sink.write(trace_id, coordinator_events)
        receipts = tuple(state.receipts)
        return CoordinatorResult(invocation_id, trace_id, result, receipts,
                                 aggregate_usage(result.usage, receipts))

    def run_oracle(self, task, specialist, *, objective=None, context=None,
                   acceptance_criteria=(), deadline=None, trusted_evidence=None):
        """Benchmark-only route that bypasses coordinator model selection."""
        deadline = deadline if deadline is not None else time.monotonic() + 900
        spec = SPECIALISTS[next(kind for kind in SPECIALISTS if kind.value == specialist)]
        request = DelegationRequest.create(
            specialist, objective or task, context or {}, acceptance_criteria,
            allowed_context_keys=spec.allowed_context_keys)
        state = self.orchestrator.new_run(task, deadline=deadline)
        return self.orchestrator.run_specialist(
            state, request, trusted_evidence=trusted_evidence)
