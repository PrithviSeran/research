import threading
import time
from types import SimpleNamespace

import pytest

import hunch.orchestration.execution as execution_mod
from hunch.agent import AgentResult
from hunch.orchestration.contracts import (DelegationRequest, SpecialistStatus,
                                           TrustedEvidence)
from hunch.orchestration.execution import (ExecutionLease, PhysicalOperationBudget,
                                           RetryableTransportError, ScopedDispatcher)
from hunch.orchestration.orchestrator import (InMemoryTraceSink, OrchestrationLimits,
                                              SpecialistOrchestrator)


class _Hunch:
    def __init__(self):
        self._computer = SimpleNamespace(session=SimpleNamespace(disturbances={
            "pixel_clicks": 0, "keystrokes": 0, "key_combos": 0,
            "app_raises": 0, "drags": 0}))
        self._can_use_tool = None


class _Child:
    def __init__(self, run):
        self._run = run
        self.interrupted = False
        self.closed = False

    def run(self, task, **kwargs):
        return self._run(task, **kwargs)

    def interrupt(self):
        self.interrupted = True

    def close(self):
        self.closed = True


def _request(kind="ax"):
    allowed = {"ax": {"app"}, "native": {"source"}, "web": {"url"},
               "applescript": {"app"}}[kind]
    context = ({"app": "Settings"} if "app" in allowed else
               {"source": "/tmp/a"} if "source" in allowed else
               {"url": "https://example.test"})
    return DelegationRequest.create(kind, "complete objective", context, ["state changed"],
                                    allowed_context_keys=allowed)


def test_scoped_dispatch_uses_same_hunch_and_permission_before_core(monkeypatch):
    hunch, events, order = _Hunch(), [], []

    async def permit(name, args, context):
        order.append("permission")
        return type("PermissionResultAllow", (), {})()

    def core(mac, name, args):
        order.append("core")
        assert mac is hunch
        return "ok", False

    monkeypatch.setattr(execution_mod, "_dispatch_core", core)
    dispatcher = ScopedDispatcher(hunch, {"snapshot"}, PhysicalOperationBudget(3),
                                  can_use_tool=permit, on_dispatch=events.append)
    assert dispatcher("snapshot", {"app": "Notes"}) == ("ok", False)
    assert order == ["permission", "core"] and events[0]["cost"] == 1


def test_trace_sink_does_not_expose_mutable_internal_records():
    sink = InMemoryTraceSink()
    events = [{"kind": "text", "data": {"value": "original"}}]
    sink.write("t", events)
    events[0]["data"]["value"] = "outside"
    read = sink.read("t")
    read[0]["data"]["value"] = "reader"
    assert sink.read("t")[0]["data"]["value"] == "original"


def test_permission_denial_and_scope_rejection_never_dispatch(monkeypatch):
    monkeypatch.setattr(execution_mod, "_dispatch_core", lambda *a: pytest.fail("dispatched"))
    denied = lambda *a: type("PermissionResultDeny", (), {})()
    budget = PhysicalOperationBudget(2)
    dispatcher = ScopedDispatcher(_Hunch(), {"snapshot"}, budget, can_use_tool=denied)
    assert dispatcher("snapshot", {})[0].startswith("REFUSED")
    assert dispatcher("applescript", {})[1] is True
    assert budget.used == 0


def test_operation_costs_and_atomic_rejection(monkeypatch):
    calls = []
    monkeypatch.setattr(execution_mod, "_dispatch_core",
                        lambda h, n, a: (calls.append((n, a)) or "ok", False))
    budget = PhysicalOperationBudget(3)
    dispatch = ScopedDispatcher(_Hunch(), {"act", "file_op"}, budget)
    dispatch("act", {"actions": [{"action": "click"}, {"action": "click"}]})
    refused = dispatch("file_op", {"batch": [{}, {}]})
    assert budget.used == 2 and len(calls) == 1 and refused[0].startswith("REFUSED")


def test_orchestrator_fresh_child_trace_and_bounded_receipt(monkeypatch):
    hunch, children, sink = _Hunch(), [], InMemoryTraceSink()
    monkeypatch.setattr(execution_mod, "_dispatch_core", lambda h, n, a: ("SECRET VALUE", False))

    def factory(config, dispatcher):
        def run(task, on_event, **kwargs):
            assert "Original user task" in task and "Delegated objective" in task
            on_event("tool", {"name": "snapshot", "input": {"app": "Settings"}})
            value, _ = dispatcher("snapshot", {"app": "Settings"})
            on_event("tool_result", value)
            on_event("text", "SECRET VALUE")
            return AgentResult("done", 2, "end_turn", {"input_tokens": 5}, False)
        child = _Child(run)
        children.append((child, config))
        return child

    orchestrator = SpecialistOrchestrator(hunch, trace_sink=sink, session_factory=factory)
    state = orchestrator.new_run("toggle a setting", deadline=time.monotonic() + 10)
    first = orchestrator.run_specialist(state, _request())
    second = orchestrator.run_specialist(state, _request())
    assert len(children) == 2 and children[0][0] is not children[1][0]
    assert all(not config.allow_delegation and not config.internal_tools for _, config in children)
    assert "SECRET VALUE" not in first.to_json()
    assert any(event["data"] == "SECRET VALUE" for event in sink.read(first.trace_id))
    assert first.status is SpecialistStatus.REPORTED_COMPLETE


def test_retry_once_only_for_retryable_transport_error():
    attempts = []

    def factory(config, dispatcher):
        index = len(attempts)
        attempts.append(index)
        def run(task, **kwargs):
            if index == 0:
                raise RetryableTransportError("temporary")
            return AgentResult("done", 1, "end_turn", {}, False)
        return _Child(run)

    orchestrator = SpecialistOrchestrator(
        _Hunch(), session_factory=factory,
        limits=OrchestrationLimits(retries=1, physical_operations=3))
    state = orchestrator.new_run("task", deadline=time.monotonic() + 10)
    receipt = orchestrator.run_specialist(state, _request())
    assert len(attempts) == 2 and state.retries == 1
    assert receipt.status is SpecialistStatus.BLOCKED


def test_runtime_verifier_controls_verified_status():
    evidence = iter([
        TrustedEvidence.create("runtime_verifier", {"passed": False}),
        TrustedEvidence.create("runtime_verifier", {"passed": True}),
    ])

    def factory(config, dispatcher):
        def run(task, on_event, **kwargs):
            on_event("tool", {"name": "snapshot", "input": {"app": "Settings"}})
            dispatcher("snapshot", {"app": "Settings"})
            return AgentResult("done", 1, "end_turn", {}, False)
        return _Child(run)

    orchestrator = SpecialistOrchestrator(
        _Hunch(), session_factory=factory,
        state_verifier=lambda state, request: next(evidence))
    state = orchestrator.new_run("task", deadline=time.monotonic() + 10)
    failed = orchestrator.run_specialist(state, _request())
    passed = orchestrator.run_specialist(state, _request())
    assert failed.status is SpecialistStatus.VERIFICATION_FAILED
    assert passed.status is SpecialistStatus.VERIFIED_COMPLETE


def test_ordinary_failure_is_not_retried():
    attempts = []
    def factory(config, dispatcher):
        attempts.append(1)
        return _Child(lambda task, **kwargs: (_ for _ in ()).throw(ValueError("bad task")))
    orchestrator = SpecialistOrchestrator(_Hunch(), session_factory=factory)
    state = orchestrator.new_run("task", deadline=time.monotonic() + 10)
    receipt = orchestrator.run_specialist(state, _request())
    assert len(attempts) == 1 and receipt.status is SpecialistStatus.FAILED


def test_whole_episode_lease_serializes_two_threads_and_does_not_abandon_timeout():
    entered, release = threading.Event(), threading.Event()
    counter_lock = threading.Lock()
    active = 0
    max_active = 0

    def factory(config, dispatcher):
        def run(task, **kwargs):
            nonlocal active, max_active
            with counter_lock:
                active += 1
                max_active = max(max_active, active)
            entered.set()
            release.wait(2)
            with counter_lock:
                active -= 1
            return AgentResult("done", 1, "end_turn", {}, False)
        return _Child(run)

    lease = ExecutionLease()
    orchestrator = SpecialistOrchestrator(_Hunch(), session_factory=factory, lease=lease,
        limits=OrchestrationLimits(delegations=3, physical_operations=3, reserve_seconds=0))
    state = orchestrator.new_run("task", deadline=time.monotonic() + 10)
    results = []
    first = threading.Thread(target=lambda: results.append(
        orchestrator.run_specialist(state, _request())), daemon=True)
    second = threading.Thread(target=lambda: results.append(
        orchestrator.run_specialist(state, _request())), daemon=True)
    first.start()
    assert entered.wait(1)
    second.start()
    time.sleep(0.05)
    assert active == 1 and lease.active == 1 and len(results) == 0
    release.set()
    first.join(1); second.join(1)
    assert len(results) == 2 and max_active == 1 and lease.max_active == 1


def test_interrupt_propagates_to_active_child():
    entered, release = threading.Event(), threading.Event()
    children = []
    def factory(config, dispatcher):
        child = _Child(lambda task, **kwargs: (
            entered.set(), release.wait(1), AgentResult("", 1, "interrupted", {}, True))[2])
        children.append(child)
        return child
    orchestrator = SpecialistOrchestrator(_Hunch(), session_factory=factory)
    state = orchestrator.new_run("task", deadline=time.monotonic() + 10)
    thread = threading.Thread(target=lambda: orchestrator.run_specialist(state, _request()), daemon=True)
    thread.start(); assert entered.wait(1)
    orchestrator.interrupt()
    assert children[0].interrupted
    release.set(); thread.join(1)
