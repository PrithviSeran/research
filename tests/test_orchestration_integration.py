import json
import time
from types import SimpleNamespace

import hunch.orchestration.execution as execution_mod
from hunch.backends.ollama import OllamaBackend
from hunch.orchestration.coordinator import Coordinator
from hunch.orchestration.orchestrator import InMemoryTraceSink, SpecialistOrchestrator


class _Hunch:
    def __init__(self):
        self._computer = SimpleNamespace(session=SimpleNamespace(disturbances={
            "pixel_clicks": 0, "keystrokes": 0, "key_combos": 0,
            "app_raises": 0, "drags": 0}))
        self._can_use_tool = None


def _message(content="", tool=None, arguments=None):
    message = {"role": "assistant", "content": content}
    if tool:
        message["tool_calls"] = [{"function": {"name": tool,
                                                "arguments": arguments or {}}}]
    return {"message": message, "prompt_eval_count": 10, "eval_count": 2}


def test_scripted_coordinator_specialist_path_keeps_child_trace_separate(monkeypatch):
    hunch, sink, backends, physical = _Hunch(), InMemoryTraceSink(), [], []
    monkeypatch.setattr(execution_mod, "_dispatch_core", lambda h, name, args: (
        physical.append((h, name, args)) or "created private note SECRET", False))

    responses = {
        "coordinator": [
            _message(tool="delegate", arguments={
                "specialist": "applescript", "objective": "create the requested note",
                "context": {"app": "Notes"}, "acceptance_criteria": ["note exists"]}),
            _message("Delegation reported complete."),
        ],
        "applescript": [
            _message(tool="applescript", arguments={"script": "tell app \"Notes\" to return 1"}),
            _message("Completed; private note SECRET exists."),
        ],
    }

    def factory(config, dispatcher):
        backend = OllamaBackend(hunch, session_config=config, dispatcher=dispatcher)
        sequence = iter(responses[config.role])
        backend._chat = lambda body: next(sequence)
        backends.append(backend)
        return backend

    orchestrator = SpecialistOrchestrator(hunch, trace_sink=sink, session_factory=factory)
    coordinator = Coordinator(orchestrator, session_factory=factory)
    outcome = coordinator.run("Create a note", deadline=time.monotonic() + 10)

    assert not outcome.result.aborted and len(outcome.receipts) == 1
    assert physical == [(hunch, "applescript", {"script": "tell app \"Notes\" to return 1"})]
    child = next(backend for backend in backends if backend.session_config.role == "applescript")
    outer = next(backend for backend in backends if backend.session_config.role == "coordinator")
    child_receipt = outcome.receipts[0]
    assert any("SECRET" in str(event) for event in sink.read(child_receipt.trace_id))
    assert "SECRET" not in child_receipt.to_json()
    assert "SECRET" not in json.dumps(outer.messages)
    assert any(message.get("tool_name") == "delegate" for message in outer.messages)
    assert all(tool["function"]["name"] == "delegate" for tool in outer._session_tools())
    assert outcome.usage["total_input_tokens"] == 40


def test_invalid_delegation_never_touches_mac(monkeypatch):
    hunch, physical, created = _Hunch(), [], []
    monkeypatch.setattr(execution_mod, "_dispatch_core",
                        lambda *args: physical.append(args))

    def factory(config, dispatcher):
        backend = OllamaBackend(hunch, session_config=config, dispatcher=dispatcher)
        responses = iter([
            _message(tool="delegate", arguments={
                "specialist": "ax", "objective": "x",
                "context": {"secret": "not allowed"}}),
            _message("blocked"),
        ])
        backend._chat = lambda body: next(responses)
        created.append(backend)
        return backend

    orchestrator = SpecialistOrchestrator(hunch, session_factory=lambda *a: (_ for _ in ()).throw(
        AssertionError("specialist created")))
    result = Coordinator(orchestrator, session_factory=factory).run(
        "x", deadline=time.monotonic() + 10)
    assert physical == [] and result.receipts == ()
    tool_result = next(m for m in created[0].messages if m.get("role") == "tool")
    assert "invalid delegation request" in json.loads(tool_result["content"])["summary"]


def test_documented_note_and_directory_context_reach_specialists(monkeypatch):
    hunch, requests = _Hunch(), []

    class RecordingOrchestrator(SpecialistOrchestrator):
        def run_specialist(self, state, request, **kwargs):
            requests.append(request)
            return super().run_specialist(state, request, **kwargs)

    def child_factory(config, dispatcher):
        backend = OllamaBackend(hunch, session_config=config, dispatcher=dispatcher)
        backend._chat = lambda body: _message("blocked")
        return backend

    coordinator_responses = iter([
        _message(tool="delegate", arguments={
            "specialist": "applescript", "objective": "edit note",
            "context": {"note_title": "Example"}}),
        _message(tool="delegate", arguments={
            "specialist": "applescript", "objective": "enumerate PDFs",
            "context": {"source_directory": "/tmp/docs"}}),
        _message("blocked"),
    ])

    def coordinator_factory(config, dispatcher):
        backend = OllamaBackend(hunch, session_config=config, dispatcher=dispatcher)
        backend._chat = lambda body: next(coordinator_responses)
        return backend

    orchestrator = RecordingOrchestrator(hunch, session_factory=child_factory)
    outcome = Coordinator(orchestrator, session_factory=coordinator_factory).run(
        "task", deadline=time.monotonic() + 10)
    assert len(outcome.receipts) == 2
    assert dict(requests[0].context) == {"note_title": "Example"}
    assert dict(requests[1].context) == {"source_directory": "/tmp/docs"}
