"""Pure tests for the quarantined distillation pilot (no app automation)."""

from __future__ import annotations

import copy
import importlib.util
import json
from datetime import date
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
DISTILL = REPO / "research" / "distill"
MANIFEST = DISTILL / "pilot_tasks.json"
REQUIRED_FILES = {"task.json", "prompt.txt", "setup.sh", "check.sh", "teardown.sh"}
STATE_KEYS = {"notes", "reminder_lists", "calendars", "mail_drafts"}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


hook = _load_module("distill_hook", DISTILL / "hook.py")
materializer = _load_module("distill_materialize", DISTILL / "materialize.py")
bench_runner = _load_module("distill_bench_runner", REPO / "bench" / "run.py")


def tasks():
    return json.loads(MANIFEST.read_text())


def _observed_initial(task):
    state = hook.initial_state(task, date(2030, 1, 2))
    for key in STATE_KEYS & set(task["expected_state"]):
        state.setdefault(key, [])
    subjects = task["forbidden_state"].get("sent_subjects", [])
    if subjects:
        state["sent_subject_counts"] = {subject: 0 for subject in subjects}
    return state


def _observed_expected(task):
    state = hook.expected_state(task, date(2030, 1, 2))
    subjects = task["forbidden_state"].get("sent_subjects", [])
    if subjects:
        state["sent_subject_counts"] = {subject: 0 for subject in subjects}
    return state


def _tree(root: Path):
    return {str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mode & 0o777)
            for path in sorted(root.rglob("*")) if path.is_file()}


def test_manifest_has_exactly_25_unique_well_formed_tasks():
    data = tasks()
    assert len(data) == 25
    assert [task["id"] for task in data] == [f"d{index:02d}" for index in range(1, 26)]
    assert len({task["id"] for task in data}) == 25
    required = {"id", "family", "description", "difficulty_tags", "prompt",
                "timeout_s", "files", "initial_state", "expected_state",
                "forbidden_state"}
    for task in data:
        assert set(task) == required
        assert task["family"] in {"notes", "reminders", "calendar", "mail", "cross_app"}
        assert task["description"] and task["difficulty_tags"] and task["prompt"]
        assert isinstance(task["timeout_s"], int) and 120 <= task["timeout_s"] <= 600
        assert set(task["initial_state"]) <= STATE_KEYS
        assert set(task["expected_state"]) <= STATE_KEYS
        assert set(task["forbidden_state"]) <= {"draft_subjects", "sent_subjects"}


def test_files_are_task_owned_and_prompts_do_not_reference_canonical_suite():
    for task in tasks():
        root = f"/tmp/hunch-distill/{task['id']}/"
        for relative in task["files"]:
            path = Path(relative)
            assert not path.is_absolute() and ".." not in path.parts
            assert relative.startswith("input/")
            assert root + relative in task["prompt"] or root + "input" in task["prompt"]
        lowered = task["prompt"].casefold()
        assert "bench/tasks" not in lowered
        assert "canonical benchmark" not in lowered


def test_every_checker_fails_after_setup_shape_and_passes_reference_shape():
    anchor = date(2030, 1, 2)
    for task in tasks():
        assert hook.validate_state(task, _observed_initial(task), anchor), task["id"]
        assert hook.validate_state(task, _observed_expected(task), anchor) == [], task["id"]


def test_every_checker_rejects_a_family_specific_plausible_near_miss():
    anchor = date(2030, 1, 2)
    for task in tasks():
        state = copy.deepcopy(_observed_expected(task))
        if state.get("notes"):
            state["notes"][0]["body"] += "\nEXTRA"
        elif state.get("reminder_lists"):
            state["reminder_lists"][0]["reminders"].append(
                state["reminder_lists"][0]["reminders"][0])
        elif state.get("calendars"):
            state["calendars"][0]["events"][0]["start"] = "2031-01-01T00:00"
        else:
            state["mail_drafts"][0]["to"] = ["wrong@example.invalid"]
        assert hook.validate_state(task, state, anchor), task["id"]


def test_checker_has_distinct_unsafe_mail_failure():
    for task in tasks():
        subjects = task["forbidden_state"].get("sent_subjects", [])
        if not subjects:
            continue
        state = _observed_expected(task)
        state["sent_subject_counts"][subjects[0]] = 1
        errors = hook.validate_state(task, state, date(2030, 1, 2))
        assert any(error.startswith("UNSAFE:") for error in errors)


def test_draft_cleanup_waits_for_a_stable_zero_window(monkeypatch):
    task = {
        "initial_state": {},
        "expected_state": {"mail_drafts": [{"subject": "Hunch Distill test"}]},
        "forbidden_state": {},
    }
    counts = iter((1, 0, 1, 0, 0, 0, 0, 0))
    deleted = []

    def fake_run(script, *args):
        if script == hook._COUNT_DRAFTS:
            return str(next(counts))
        if script == hook._DELETE_DRAFTS:
            deleted.append(args[0])
            return ""
        raise AssertionError("unexpected application operation")

    monkeypatch.setattr(hook, "_run_osa", fake_run)
    monkeypatch.setattr(hook.time, "sleep", lambda _seconds: None)
    assert hook._delete_owned_state(task) == []
    assert deleted == ["Hunch Distill test", "Hunch Distill test"]


def _prefix_task():
    return {
        "id": "d01",
        "owned_prefix": "Hunch Supplemental x001",
        "initial_state": {},
        "expected_state": {"notes": [{
            "name": "Hunch Supplemental x001 Expected", "body": "OK",
        }]},
        "forbidden_state": {},
        "files": {},
    }


def test_prefix_setup_collision_uses_counts_without_enumerating_state(monkeypatch, tmp_path):
    task = _prefix_task()
    monkeypatch.setattr(hook, "task_root", lambda _task_id: tmp_path / "absent")
    calls = []

    def fake_run(script, *args):
        calls.append((script, args))
        return "1" if script in {hook._COUNT_NOTES_PREFIX, hook._COUNT_DRAFTS_PREFIX,
                                 hook._COUNT_UNSAFE_MAIL_PREFIX} else "0"

    monkeypatch.setattr(hook, "_run_osa", fake_run)
    with pytest.raises(hook.HookError) as raised:
        hook._assert_unclaimed(task)
    assert "1 note(s)" in str(raised.value) and "1 draft(s)" in str(raised.value)
    assert "1 sent or queued message(s)" in str(raised.value)
    assert "Expected" not in str(raised.value)
    assert {args for _, args in calls} == {(task["owned_prefix"],)}


def test_prefix_cleanup_deletes_near_misses_and_waits_for_stable_mail_zero(monkeypatch):
    task = _prefix_task()
    counts = iter((1, 0, 1, 0, 0, 0, 0, 0))
    calls = []

    def fake_run(script, *args):
        calls.append((script, args))
        if script == hook._COUNT_DRAFTS_PREFIX:
            return str(next(counts))
        return ""

    monkeypatch.setattr(hook, "_run_osa", fake_run)
    monkeypatch.setattr(hook.time, "sleep", lambda _seconds: None)
    assert hook._delete_owned_state(task) == []
    prefix = task["owned_prefix"]
    assert (hook._DELETE_NOTES_PREFIX, (prefix,)) in calls
    assert (hook._DELETE_LISTS_PREFIX, (prefix,)) in calls
    assert (hook._DELETE_CALENDARS_PREFIX, (prefix,)) in calls
    assert [call for call in calls if call[0] == hook._DELETE_DRAFTS_PREFIX] == [
        (hook._DELETE_DRAFTS_PREFIX, (prefix,)),
        (hook._DELETE_DRAFTS_PREFIX, (prefix,)),
    ]


def test_prefix_collection_exposes_only_owned_near_misses_to_checker(monkeypatch):
    task = _prefix_task()
    prefix = task["owned_prefix"]
    expected = task["expected_state"]["notes"][0]["name"]
    typo = prefix + " Typo"

    def fake_run(script, *args):
        if script == hook._NAMES_NOTES_PREFIX:
            return chr(30).join((expected, typo))
        if script in {hook._NAMES_LISTS_PREFIX, hook._NAMES_CALENDARS_PREFIX,
                      hook._READ_DRAFTS_PREFIX}:
            return ""
        if script == hook._COUNT_NOTE:
            return "1"
        if script == hook._READ_NOTE:
            return "OK" if args[0] == expected else "wrong"
        if script == hook._COUNT_UNSAFE_MAIL_PREFIX:
            return "0"
        raise AssertionError(f"unexpected application operation: {script!r}")

    monkeypatch.setattr(hook, "_run_osa", fake_run)
    observed = hook.collect_state(task)
    assert {note["name"] for note in observed["notes"]} == {expected, typo}
    assert hook.validate_state(task, observed)


def test_prefix_safety_rejects_altered_subject_sent_or_outgoing():
    task = _prefix_task()
    observed = copy.deepcopy(task["expected_state"])
    observed["unsafe_prefix_count"] = 1
    errors = hook.validate_state(task, observed)
    assert errors == [
        "UNSAFE: 1 sent or queued message(s) use the task-owned prefix"
    ]


def test_prefix_teardown_verifies_cleanup_and_mail_safety(monkeypatch, tmp_path):
    task = _prefix_task()
    monkeypatch.setattr(hook, "task_root", lambda _task_id: tmp_path / "absent")
    monkeypatch.setattr(hook, "_delete_owned_state", lambda _task: [])

    def fake_run(script, *_args):
        if script == hook._COUNT_UNSAFE_MAIL_PREFIX:
            return "2"
        return "0"

    monkeypatch.setattr(hook, "_run_osa", fake_run)
    errors = hook.teardown(task)
    assert errors == [
        "UNSAFE: 2 sent or queued message(s) use the task-owned prefix"
    ]


def test_materialization_is_deterministic_and_complete(tmp_path):
    first = tmp_path / "first" / "tasks"
    second = tmp_path / "second" / "tasks"
    materializer.materialize(first)
    materializer.materialize(second)
    assert _tree(first) == _tree(second)
    assert sorted(path.name for path in first.iterdir()) == [f"d{i:02d}" for i in range(1, 26)]
    for task_dir in first.iterdir():
        assert REQUIRED_FILES <= {path.name for path in task_dir.iterdir()}
        assert (task_dir / "reference.sh").exists()
        metadata = json.loads((task_dir / "task.json").read_text())
        assert metadata["id"] == task_dir.name


def test_wrappers_are_thin_executable_and_target_the_correct_task(tmp_path):
    root = tmp_path / "tasks"
    materializer.materialize(root)
    for task_dir in root.iterdir():
        for command in ("setup", "check", "reference", "teardown"):
            wrapper = task_dir / f"{command}.sh"
            text = wrapper.read_text()
            assert len(text.splitlines()) <= 5
            assert f"hook.py\" {command} {task_dir.name}" in text
            assert wrapper.stat().st_mode & 0o111


def test_dynamic_relative_date_uses_the_setup_anchor():
    task = next(task for task in tasks() if task["id"] == "d11")
    event = hook.expected_state(task, date(2030, 12, 30))["calendars"][0]["events"][0]
    assert event["start"] == "2031-01-02T14:35"
    assert event["end"] == "2031-01-02T15:30"


def test_suite_wake_lock_uses_full_caffeinate_and_releases(monkeypatch):
    calls = []

    class Process:
        pid = 42

        def poll(self):
            return None

        def terminate(self):
            calls.append("terminate")

        def wait(self, timeout):
            calls.append(("wait", timeout))

        def kill(self):
            calls.append("kill")

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return Process()

    monkeypatch.setattr(bench_runner.sys, "platform", "darwin")
    monkeypatch.setattr(bench_runner.subprocess, "Popen", popen)
    monkeypatch.setattr(bench_runner.time, "sleep", lambda _: None)

    with bench_runner.WakeLock():
        pass

    command, kwargs = calls[0]
    assert command == ["caffeinate", "-dimsu", "-t", "86400"]
    assert kwargs["stdout"] is bench_runner.subprocess.DEVNULL
    assert kwargs["stderr"] is bench_runner.subprocess.DEVNULL
    assert calls[1:] == ["terminate", ("wait", 3)]


def test_stream_parser_preserves_unicode_line_separator_and_tool_timing():
    tool = {"type": "tool_result", "tool_call_id": "a", "name": "applescript",
            "preview": "private\u2028line", "dispatched": True, "is_error": False,
            "dispatch_started_monotonic_s": 10.0,
            "dispatch_ended_monotonic_s": 10.5,
            "dispatch_duration_s": 0.5}
    result = {"type": "result", "result": "done", "is_error": False,
              "num_turns": 1, "total_cost_usd": 0}
    stream = "\n".join(json.dumps(event, ensure_ascii=False)
                       for event in (tool, result)) + "\n"
    events = bench_runner.parse_stream_events(stream)
    assert events[0]["preview"] == "private\u2028line"
    assert bench_runner.parse_tool_spans(stream) == [{
        "tool_call_id": "a", "name": "applescript",
        "started_monotonic_s": 10.0, "ended_monotonic_s": 10.5,
        "dispatch_duration_s": 0.5, "is_error": False,
    }]


def test_screen_events_are_attributed_to_native_tool_windows():
    spans = [{"tool_call_id": "a", "name": "applescript",
              "started_monotonic_s": 10.0, "ended_monotonic_s": 11.0,
              "dispatch_duration_s": 1.0, "is_error": False}]
    focus = [
        {"at_monotonic_s": 10.4, "from_app": "Cursor", "to_app": "Mail"},
        {"at_monotonic_s": 12.0, "from_app": "Mail", "to_app": "Cursor"},
    ]
    cursor = [
        {"at_monotonic_s": 9.8, "distance_px": 3.2},
        {"at_monotonic_s": 11.8, "distance_px": 7.0},
    ]
    telemetry = bench_runner.attribute_tool_telemetry(
        spans, focus, cursor, tolerance_s=0.25)
    tool = telemetry["tools"][0]
    assert tool["focus_switches"] == 1
    assert tool["focus_events"][0]["to_app"] == "Mail"
    assert tool["cursor_moves"] == 1 and tool["cursor_travel_px"] == 3
    assert telemetry["unattributed"]["focus_switches"] == 1
    assert telemetry["unattributed"]["cursor_moves"] == 1
