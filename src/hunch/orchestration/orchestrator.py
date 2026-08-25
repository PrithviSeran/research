"""Fresh-session specialist execution on one shared Hunch instance."""
import copy
import threading
import time
import uuid
from dataclasses import dataclass, field

from ..backends.ollama import OllamaBackend, OllamaSessionConfig
from .contracts import DelegationRequest, SpecialistReceipt, SpecialistStatus
from .execution import (ExecutionLease, PhysicalOperationBudget,
                        RetryableTransportError, ScopedDispatcher)
from .specialists import SPECIALISTS


@dataclass(frozen=True)
class OrchestrationLimits:
    delegations: int = 3
    specialist_turns: int = 40
    physical_operations: int = 40
    retries: int = 1
    reserve_seconds: float = 5.0

    def __post_init__(self):
        if min(self.delegations, self.specialist_turns, self.physical_operations) < 0:
            raise ValueError("orchestration limits cannot be negative")
        if self.retries not in (0, 1):
            raise ValueError("automatic retries must be zero or one")


@dataclass
class OrchestrationRunState:
    original_task: str
    deadline: float
    budget: PhysicalOperationBudget
    delegations: int = 0
    retries: int = 0
    receipts: list = field(default_factory=list)


class InMemoryTraceSink:
    """Protected full traces. Callers receive only opaque trace IDs."""

    def __init__(self):
        self._lock = threading.Lock()
        self._traces = {}

    def write(self, trace_id, events):
        with self._lock:
            self._traces[trace_id] = tuple(copy.deepcopy(events))

    def read(self, trace_id):
        with self._lock:
            return copy.deepcopy(self._traces[trace_id])


def _disturbances(hunch):
    session = getattr(getattr(hunch, "_computer", None), "session", None)
    values = getattr(session, "disturbances", {})
    return {key: int(value) for key, value in values.items()}


def _delta(before, after):
    return {key: after.get(key, 0) - before.get(key, 0)
            for key in sorted(set(before) | set(after))}


def _safe_usage(usage):
    allowed = {"input_tokens", "output_tokens", "context_compactions",
               "context_messages_pruned", "tool_results_truncated",
               "max_context_chars", "fallback_tool_calls", "continue_nudges",
               "rejected_tool_calls", "malformed_tool_calls", "no_progress_aborts"}
    return {key: value for key, value in (usage or {}).items()
            if key in allowed and isinstance(value, (str, int, float, bool, type(None)))}


class SpecialistOrchestrator:
    def __init__(self, hunch, *, limits=None, trace_sink=None, session_factory=None,
                 lease=None, can_use_tool=None, on_event=None):
        self.hunch = hunch
        self.limits = limits or OrchestrationLimits()
        self.trace_sink = trace_sink or InMemoryTraceSink()
        self.session_factory = session_factory or self._default_session_factory
        self.lease = lease or ExecutionLease()
        self.can_use_tool = (can_use_tool if can_use_tool is not None
                             else getattr(hunch, "_can_use_tool", None))
        self.on_event = on_event
        self._prompts = {kind: spec.prompt() for kind, spec in SPECIALISTS.items()}
        self._interrupted = threading.Event()
        self._active_lock = threading.Lock()
        self._active_child = None

    def new_run(self, original_task, *, deadline):
        if not isinstance(original_task, str) or not original_task.strip():
            raise ValueError("original task cannot be empty")
        self.reset_interrupt()
        return OrchestrationRunState(
            original_task=original_task.strip(), deadline=deadline,
            budget=PhysicalOperationBudget(self.limits.physical_operations))

    def _default_session_factory(self, config, dispatcher):
        return OllamaBackend(self.hunch, session_config=config, dispatcher=dispatcher,
                             can_use_tool=self.can_use_tool)

    def interrupt(self):
        self._interrupted.set()
        with self._active_lock:
            child = self._active_child
        if child is not None:
            child.interrupt()

    def reset_interrupt(self):
        self._interrupted.clear()

    def _child_task(self, state, request):
        criteria = "\n".join(f"- {item}" for item in request.acceptance_criteria) or "- report observable outcome"
        context = "\n".join(f"- {key}: {value}" for key, value in request.context.items()) or "- none"
        return (
            "Original user task:\n" + state.original_task + "\n\n"
            "Delegated objective:\n" + request.objective + "\n\n"
            "Trusted bounded context:\n" + context + "\n\n"
            "Acceptance criteria:\n" + criteria)

    def run_specialist(self, state, request, *, trusted_evidence=None):
        if not isinstance(request, DelegationRequest):
            raise TypeError("request must be a DelegationRequest")
        if state.delegations >= self.limits.delegations:
            raise ValueError("delegation budget exhausted")
        spec = SPECIALISTS[request.specialist]
        unknown_context = set(request.context) - set(spec.allowed_context_keys)
        if unknown_context:
            raise ValueError(f"context keys not allowed for {spec.kind.value}: {sorted(unknown_context)}")
        state.delegations += 1
        delegation_id = f"d-{uuid.uuid4().hex}"
        trace_id = f"t-{uuid.uuid4().hex}"
        events, dispatch_events = [], []
        before = _disturbances(self.hunch)
        used_before = state.budget.used
        result = None
        failure = None

        def emit(kind, data):
            events.append({"kind": kind, "data": data})
            if self.on_event is not None:
                self.on_event(kind, data, delegation_id)

        dispatcher = ScopedDispatcher(
            self.hunch, spec.tool_names, state.budget, deadline=state.deadline,
            can_use_tool=self.can_use_tool, on_dispatch=dispatch_events.append,
            interrupted=self._interrupted.is_set)
        config = OllamaSessionConfig.from_environment(
            role=spec.kind.value, system_prompt=self._prompts[spec.kind],
            tool_names=spec.tool_names)

        with self.lease.hold():
            attempts = 0
            while True:
                child = self.session_factory(config, dispatcher)
                with self._active_lock:
                    self._active_child = child
                try:
                    child_deadline = max(time.monotonic(),
                                         state.deadline - self.limits.reserve_seconds)
                    result = child.run(self._child_task(state, request),
                                       max_turns=self.limits.specialist_turns,
                                       on_event=emit, deadline=child_deadline)
                    break
                except Exception as exc:
                    retryable = (isinstance(exc, RetryableTransportError)
                                 or getattr(exc, "retryable", False))
                    if (not retryable or attempts >= self.limits.retries
                            or self._interrupted.is_set()):
                        failure = exc
                        break
                    attempts += 1
                    state.retries += 1
                    events.append({"kind": "retry", "data": {"reason": "transport"}})
                finally:
                    events.append({"kind": "conversation", "data": {
                        "system": getattr(child, "last_system", ""),
                        "messages": list(getattr(child, "messages", []))}})
                    child.close()
                    with self._active_lock:
                        self._active_child = None

        after = _disturbances(self.hunch)
        disturbance_delta = _delta(before, after)
        physical_operations = state.budget.used - used_before
        events.extend({"kind": "dispatch", "data": event} for event in dispatch_events)
        self.trace_sink.write(trace_id, events)
        tool_names = [event["data"].get("name") for event in events
                      if event["kind"] == "tool" and isinstance(event["data"], dict)]
        safe_tools = ",".join(name for name in tool_names if name) or "none"

        if failure is not None:
            status, stop_reason, turns, usage = (SpecialistStatus.FAILED,
                                                  type(failure).__name__, 0, {})
        else:
            stop_reason, turns, usage = result.stop_reason, result.turns, result.usage
            if stop_reason == "timeout":
                status = SpecialistStatus.TIMEOUT
            elif result.aborted:
                status = SpecialistStatus.FAILED
            elif trusted_evidence is not None:
                status = SpecialistStatus.VERIFIED_COMPLETE
            elif "blocked" in (result.text or "").lower():
                status = SpecialistStatus.BLOCKED
            else:
                status = SpecialistStatus.REPORTED_COMPLETE
        evidence = trusted_evidence.labels() if trusted_evidence is not None else ()
        if not evidence:
            evidence = (f"tools:{safe_tools}", f"dispatches:{len(dispatch_events)}")
        receipt = SpecialistReceipt.create(
            delegation_id=delegation_id, specialist=spec.kind, status=status,
            summary=(f"specialist={spec.kind.value}; status={status.value}; "
                     f"tools={safe_tools}; stop={stop_reason}"),
            evidence=evidence, stop_reason=stop_reason, turns=turns,
            tool_calls=len(tool_names), physical_operations=physical_operations,
            usage=_safe_usage(usage), trace_id=trace_id,
            disturbance_delta=disturbance_delta,
            trusted_evidence=trusted_evidence)
        state.receipts.append(receipt)
        return receipt
