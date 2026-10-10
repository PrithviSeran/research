"""Pure tests for the isolated supplemental distillation curriculum."""

from __future__ import annotations

import importlib.util
import itertools
import json
import re
import copy
from collections import Counter
from datetime import date
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
DISTILL = REPO / "research" / "distill"
MANIFEST = DISTILL / "supplemental_tasks.json"
STATE_KEYS = {"notes", "reminder_lists", "calendars", "mail_drafts"}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


generator = _load("distill_supplemental_generator", DISTILL / "supplemental_tasks.py")
hook = _load("distill_supplemental_hook", DISTILL / "supplemental_hook.py")
materializer = _load("distill_supplemental_materializer", DISTILL / "materialize_supplemental.py")
pilot_hook = _load("distill_pilot_for_supplemental_isolation", DISTILL / "hook.py")
scaled_hook = _load("distill_scaled_for_supplemental_isolation", DISTILL / "scaled_hook.py")


def tasks():
    return json.loads(MANIFEST.read_text())


def _tree(root: Path):
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mode & 0o777)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


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


def test_manifest_is_exact_deterministic_generator_output():
    generated = generator.build_tasks()
    generator.validate_tasks(generated)
    assert tasks() == generated
    assert MANIFEST.read_text() == json.dumps(generated, indent=2, ensure_ascii=False) + "\n"
    assert generator.build_tasks() == generated


def test_exact_train_only_ids_and_family_balance():
    data = tasks()
    assert len(data) == 25
    assert [task["id"] for task in data] == [f"x{i:03d}" for i in range(1, 26)]
    assert {task["split"] for task in data} == {"train"}
    assert {task["suite"] for task in data} == {"distill-supplemental-v1"}
    assert Counter(task["family"] for task in data) == {
        "notes": 5, "reminders": 5, "calendar": 5, "mail": 5, "cross_app": 5,
    }
    assert all(task["decision_count"] >= 3 for task in data)
    assert len({task["archetype"] for task in data}) == 25


def test_schema_paths_names_and_mail_are_isolated_and_safe():
    required = {
        "id", "suite", "split", "family", "archetype", "description",
        "owned_prefix", "difficulty_tags", "decision_count", "prompt",
        "timeout_s", "files",
        "initial_state", "expected_state", "forbidden_state", "structural_axes",
        "operation_graph",
    }
    for task in tasks():
        assert set(task) == required
        task_id = task["id"]
        root = f"/tmp/hunch-distill-supplemental/{task_id}/input"
        prefix = f"Hunch Supplemental {task_id}"
        assert task["owned_prefix"] == prefix
        assert "distill-scale" not in task["prompt"]
        assert "bench/tasks" not in task["prompt"]
        assert root in task["prompt"]
        for relative in task["files"]:
            path = Path(relative)
            assert relative.startswith("input/")
            assert not path.is_absolute() and ".." not in path.parts
        for state in (task["initial_state"], task["expected_state"]):
            assert set(state) <= STATE_KEYS
            for kind, objects in state.items():
                for obj in objects:
                    owned = obj["subject"] if kind == "mail_drafts" else obj["name"]
                    assert owned.startswith(prefix)
                    if kind == "mail_drafts":
                        assert all(address.endswith("@example.invalid") for address in obj["to"])
        for subject in task["forbidden_state"].get("sent_subjects", []):
            assert subject.startswith(prefix)


def test_targeted_categories_and_structural_diversity_are_explicit():
    data = tasks()
    notes = data[:5]
    calendars = data[10:15]
    mail = data[15:20]
    cross = data[20:]
    assert all("exact-title-body" in task["difficulty_tags"] for task in notes)
    assert all("component-safe-date" in task["difficulty_tags"] for task in calendars)
    assert all("body-normalization" in task["difficulty_tags"] for task in mail)
    assert all("partial-recovery" in task["difficulty_tags"] for task in cross)
    assert all("direct-file-read" in task["difficulty_tags"] for task in data)
    for family in generator.FAMILIES:
        members = [task for task in data if task["family"] == family]
        assert len({json.dumps(task["structural_axes"], sort_keys=True) for task in members}) >= 4
        assert len({task["prompt"] for task in members}) == 5
    assert all(len(task["expected_state"]) >= 2 for task in cross)


def test_anti_monoculture_graph_cap_and_prompt_similarity():
    data = tasks()
    graphs = Counter(json.dumps(task["operation_graph"], sort_keys=True) for task in data)
    assert max(graphs.values()) <= 2
    for family in generator.FAMILIES:
        members = [task for task in data if task["family"] == family]
        for left, right in itertools.combinations(members, 2):
            left_tokens = set(re.findall(r"[a-z]+", left["prompt"].casefold()))
            right_tokens = set(re.findall(r"[a-z]+", right["prompt"].casefold()))
            jaccard = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
            assert jaccard <= 0.80, (left["id"], right["id"], jaccard)


def test_original_heldout_partitions_and_exclusive_shapes_are_not_reused():
    scaled = json.loads((DISTILL / "scaled_tasks.json").read_text())
    validation = [task for task in scaled if task["split"] == "validation"]
    training = [task for task in scaled if task["split"] == "train"]
    heldout_literals = {
        value.casefold()
        for task in validation
        for value in task["literal_partition"].values()
    }
    serialized = json.dumps(tasks(), ensure_ascii=False).casefold()
    assert all(value not in serialized for value in heldout_literals)
    train_encodings = {
        item.casefold().lstrip(".")
        for task in training for item in task["structural_axes"]["input_encoding"]
    }
    validation_encodings = {
        item.casefold().lstrip(".")
        for task in validation for item in task["structural_axes"]["input_encoding"]
    }
    exclusive_encodings = validation_encodings - train_encodings
    supplemental_encodings = {
        item.casefold().lstrip(".")
        for task in tasks() for item in task["structural_axes"]["input_encoding"]
    }
    assert supplemental_encodings.isdisjoint(exclusive_encodings)
    train_tags = {tag for task in training for tag in task["difficulty_tags"]}
    validation_tags = {tag for task in validation for tag in task["difficulty_tags"]}
    supplemental_tags = {tag for task in tasks() for tag in task["difficulty_tags"]}
    assert supplemental_tags.isdisjoint(validation_tags - train_tags)


def test_no_unobservable_constraints_are_requested():
    forbidden_phrases = ("reminder notes", "alarm", "attachment", "mark as read", "selected")
    for task in tasks():
        lowered = task["prompt"].casefold()
        assert not any(phrase in lowered for phrase in forbidden_phrases), task["id"]


def test_hook_configuration_is_private_and_rejects_other_suite_ids():
    assert hook.MANIFEST == DISTILL / "supplemental_tasks.json"
    assert hook.RUNTIME_ROOT == Path("/tmp/hunch-distill-supplemental")
    assert hook.task_root("x001") == Path("/tmp/hunch-distill-supplemental/x001")
    assert hook.task_root("x025") == Path("/tmp/hunch-distill-supplemental/x025")
    for invalid in ("d01", "s001", "x000", "x026", "x999"):
        with pytest.raises(hook.HookError):
            hook.get_task(invalid)
    assert pilot_hook.MANIFEST == DISTILL / "pilot_tasks.json"
    assert scaled_hook.MANIFEST == DISTILL / "scaled_tasks.json"
    assert pilot_hook.get_task("d01")["id"] == "d01"
    assert scaled_hook.get_task("s001")["id"] == "s001"


def test_every_expected_state_passes_and_initial_state_fails():
    for task in tasks():
        assert hook.validate_state(task, _observed_expected(task), date(2030, 1, 2)) == [], task["id"]
        assert hook.validate_state(task, _observed_initial(task), date(2030, 1, 2)), task["id"]


def test_every_checked_output_is_independently_required():
    for task in tasks():
        for key in task["expected_state"]:
            observed = copy.deepcopy(_observed_expected(task))
            observed[key].pop()
            errors = hook.validate_state(task, observed, date(2030, 1, 2))
            assert any(error.startswith(f"{key} mismatch:") for error in errors), (
                task["id"], key, errors
            )


def test_multisource_merge_oracle_retains_first_spelling_independently():
    task = next(task for task in tasks() if task["id"] == "x018")
    rows = []
    seen = set()
    for filename in ("input/a.txt", "input/b.txt"):
        for row in task["files"][filename].splitlines():
            if row and row.casefold() not in seen:
                seen.add(row.casefold())
                rows.append(row)
    rows.sort(key=lambda row: row.split(" | ", 1)[0].casefold())
    expected_body = "UPDATES\n" + "\n".join(rows) + "\nEND"
    assert task["expected_state"]["mail_drafts"][0]["body"] == expected_body
    assert "\namber | 10\n" in expected_body


def test_reminder_prompts_do_not_require_unobservable_order():
    for task in tasks():
        if task["family"] == "reminders":
            assert "sort" not in task["prompt"].casefold(), task["id"]


def test_materialization_is_deterministic_complete_and_separate(tmp_path):
    first = tmp_path / "one" / "tasks"
    second = tmp_path / "two" / "tasks"
    materializer.materialize(first)
    materializer.materialize(second)
    assert _tree(first) == _tree(second)
    assert materializer.DEFAULT_OUTPUT == REPO / "bench" / "distill-supplemental" / "tasks"
    assert materializer.DEFAULT_OUTPUT not in {
        REPO / "bench" / "distill" / "tasks",
        REPO / "bench" / "distill-scaled" / "tasks",
    }
    assert sorted(path.name for path in first.iterdir()) == [f"x{i:03d}" for i in range(1, 26)]
    required = {"task.json", "prompt.txt", "setup.sh", "check.sh", "teardown.sh", "reference.sh"}
    for task_dir in first.iterdir():
        assert {path.name for path in task_dir.iterdir()} == required
        metadata = json.loads((task_dir / "task.json").read_text())
        assert metadata["id"] == task_dir.name
        assert metadata["split"] == "train"


def test_materializer_refuses_other_suite_roots():
    for reserved in materializer.RESERVED_OUTPUTS:
        with pytest.raises(ValueError, match="another task suite"):
            materializer.materialize(reserved)


def test_wrappers_target_only_supplemental_hook(tmp_path):
    root = tmp_path / "tasks"
    materializer.materialize(root)
    for task_dir in root.iterdir():
        for command in ("setup", "check", "reference", "teardown"):
            wrapper = task_dir / f"{command}.sh"
            content = wrapper.read_text()
            assert "research/distill/supplemental_hook.py" in content
            assert "research/distill/scaled_hook.py" not in content
            assert f" {command} {task_dir.name}\n" in content
            assert wrapper.stat().st_mode & 0o111
