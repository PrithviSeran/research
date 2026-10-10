"""Pure validation for the quarantined 125-candidate expansion."""

from __future__ import annotations

import importlib.util
import json
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
DISTILL = REPO / "research" / "distill"
MANIFEST = DISTILL / "scaled_tasks.json"
STATE_KEYS = {"notes", "reminder_lists", "calendars", "mail_drafts"}
FORBIDDEN_KEYS = {"draft_subjects", "sent_subjects"}


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


generator = _load_module("distill_scaled_tasks", DISTILL / "scaled_tasks.py")
scaled_hook = _load_module("distill_scaled_hook", DISTILL / "scaled_hook.py")
scaled_materializer = _load_module(
    "distill_scaled_materializer", DISTILL / "materialize_scaled.py"
)
pilot_hook = _load_module("distill_pilot_isolation_hook", DISTILL / "hook.py")


def tasks():
    return json.loads(MANIFEST.read_text())


def _objects(state):
    for kind, values in state.items():
        for value in values:
            yield kind, value


def _duration(start, end):
    return int(
        (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
        // 60
    )


def _tree(root):
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mode & 0o777)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def _observed_initial(task):
    state = scaled_hook.initial_state(task, date(2030, 1, 2))
    for key in STATE_KEYS & set(task["expected_state"]):
        state.setdefault(key, [])
    subjects = task["forbidden_state"].get("sent_subjects", [])
    if subjects:
        state["sent_subject_counts"] = {subject: 0 for subject in subjects}
    return state


def _observed_expected(task):
    state = scaled_hook.expected_state(task, date(2030, 1, 2))
    subjects = task["forbidden_state"].get("sent_subjects", [])
    if subjects:
        state["sent_subject_counts"] = {subject: 0 for subject in subjects}
    return state


def test_manifest_is_exact_deterministic_generator_output():
    generated = generator.build_tasks()
    assert tasks() == generated
    assert MANIFEST.read_text() == json.dumps(
        generated, indent=2, ensure_ascii=False
    ) + "\n"
    assert generator.build_tasks() == generated


def test_exact_counts_ids_splits_families_and_lineages():
    data = tasks()
    assert len(data) == 125
    assert [task["id"] for task in data] == [f"s{index:03d}" for index in range(1, 126)]
    assert Counter(task["split"] for task in data) == {"train": 100, "validation": 25}
    assert Counter(task["family"] for task in data) == {
        "notes": 25,
        "reminders": 25,
        "calendar": 25,
        "mail": 25,
        "cross_app": 25,
    }
    lineages = defaultdict(list)
    for task in data:
        lineages[task["lineage"]].append(task)
    assert len(lineages) == 25
    for lineage, siblings in lineages.items():
        assert {task["archetype"] for task in siblings} == {lineage}
        assert [task["variant"] for task in siblings] == [1, 2, 3, 4, 5]
        assert [task["split"] for task in siblings] == [
            "train", "train", "train", "train", "validation"
        ]


def test_schema_and_state_shapes_are_well_formed():
    required = {
        "id", "family", "split", "lineage", "archetype", "variant",
        "operation_graph", "structural_axes", "literal_partition",
        "decision_count", "description", "difficulty_tags", "prompt",
        "timeout_s", "files", "initial_state", "expected_state",
        "forbidden_state",
    }
    for task in tasks():
        assert set(task) == required, task["id"]
        assert task["description"] and task["prompt"]
        assert len(task["difficulty_tags"]) >= 3
        assert task["decision_count"] >= 3
        assert 120 <= task["timeout_s"] <= 600
        assert set(task["initial_state"]) <= STATE_KEYS
        assert set(task["expected_state"]) <= STATE_KEYS
        assert set(task["forbidden_state"]) <= FORBIDDEN_KEYS
        assert task["expected_state"]
        for state in (task["initial_state"], task["expected_state"]):
            assert all(isinstance(values, list) for values in state.values())
        axes = task["structural_axes"]
        assert set(axes) == {
            "input_encoding", "input_sources", "nesting_depth",
            "output_channels", "output_objects", "composition_level",
            "source_records", "output_records", "syntax_markers",
            "temporal_minutes",
        }


def test_every_file_and_owned_object_is_quarantined_to_its_task():
    for task in tasks():
        task_id = task["id"]
        runtime = f"/tmp/hunch-distill-scale/{task_id}/input"
        owned_prefix = f"Hunch Scale {task_id}"
        assert "bench/tasks" not in task["prompt"].casefold()
        assert "canonical benchmark" not in task["prompt"].casefold()
        for relative in task["files"]:
            path = Path(relative)
            assert not path.is_absolute() and ".." not in path.parts
            assert relative.startswith("input/")
        if task["files"]:
            assert runtime in task["prompt"]
        for state in (task["initial_state"], task["expected_state"]):
            for kind, obj in _objects(state):
                if kind == "notes" or kind == "reminder_lists" or kind == "calendars":
                    assert obj["name"].startswith(owned_prefix), (task_id, obj)
                elif kind == "mail_drafts":
                    assert obj["subject"].startswith(owned_prefix), (task_id, obj)
                    assert all(task_id in address for address in obj["to"]), (task_id, obj)
        for subjects in task["forbidden_state"].values():
            assert all(subject.startswith(owned_prefix) for subject in subjects)


def test_mail_tasks_are_explicitly_safe_and_machine_guarded():
    for task in tasks():
        drafts = task["expected_state"].get("mail_drafts", [])
        if not drafts:
            continue
        prompt = task["prompt"].casefold()
        assert "do not send" in prompt or "no sending" in prompt
        sent = set(task["forbidden_state"].get("sent_subjects", []))
        assert {draft["subject"] for draft in drafts} <= sent
        assert all(address.endswith("@example.invalid") for draft in drafts for address in draft["to"])


def test_graph_caps_and_heldout_compositions_are_machine_checkable():
    signature = lambda task: json.dumps(task["operation_graph"], sort_keys=True)
    train_graphs = Counter(
        signature(task) for task in tasks() if task["split"] == "train"
    )
    validation_graphs = {
        signature(task) for task in tasks() if task["split"] == "validation"
    }
    assert max(train_graphs.values()) <= 2
    assert validation_graphs.isdisjoint(train_graphs)
    assert len(validation_graphs) == 25
    assert all(task["structural_axes"]["composition_level"] == 3
               for task in tasks() if task["split"] == "validation")
    assert all(task["structural_axes"]["composition_level"] < 3
               for task in tasks() if task["split"] == "train")
    by_lineage = defaultdict(list)
    for task in tasks():
        by_lineage[task["lineage"]].append(task)
    for lineage, siblings in by_lineage.items():
        def non_label_signature(task):
            axes = dict(task["structural_axes"])
            axes.pop("composition_level")
            return json.dumps(axes, sort_keys=True)
        heldout = non_label_signature(siblings[-1])
        assert heldout not in {
            non_label_signature(task) for task in siblings[:-1]
        }, lineage


def test_operation_graph_is_derived_from_semantics_not_variant_labels():
    for task in tasks():
        assert task["operation_graph"] == generator.derive_operation_graph(task)
        serialized = json.dumps(task["operation_graph"], sort_keys=True)
        assert task["id"] not in serialized
        assert task["lineage"] not in serialized
        assert "variant" not in serialized
        assert "composition_level" not in serialized
        graph = task["operation_graph"]
        assert graph["operators"] == sorted(task["difficulty_tags"])
        assert graph["sink_topology"]["channels"] == sorted(task["expected_state"])


def test_scaled_hook_is_a_private_configuration_and_rejects_pilot_ids():
    assert pilot_hook.MANIFEST == DISTILL / "pilot_tasks.json"
    assert pilot_hook.RUNTIME_ROOT == Path("/tmp/hunch-distill")
    assert scaled_hook.MANIFEST == DISTILL / "scaled_tasks.json"
    assert scaled_hook.RUNTIME_ROOT == Path("/tmp/hunch-distill-scale")
    assert scaled_hook._HOOK is not pilot_hook
    assert scaled_hook.task_root("s001") == Path("/tmp/hunch-distill-scale/s001")
    assert scaled_hook.task_root("s125") == Path("/tmp/hunch-distill-scale/s125")
    with pytest.raises(scaled_hook.HookError):
        scaled_hook.get_task("d01")
    with pytest.raises(scaled_hook.HookError):
        scaled_hook.get_task("s126")
    with pytest.raises(pilot_hook.HookError):
        pilot_hook.get_task("s001")
    # Loading/configuring scaled_hook did not alter a separately loaded pilot.
    assert pilot_hook.get_task("d01")["id"] == "d01"


def test_every_scaled_expected_state_passes_and_initial_state_fails():
    anchor = date(2030, 1, 2)
    for task in tasks():
        assert scaled_hook.validate_state(task, _observed_expected(task), anchor) == [], task["id"]
        assert scaled_hook.validate_state(task, _observed_initial(task), anchor), task["id"]


def test_malformed_calendar_state_is_a_mismatch_not_a_checker_crash():
    task = scaled_hook.get_task("s123")
    observed = _observed_expected(task)
    observed["calendars"] = [{
        "name": "Hunch Scale s123 Calendar",
        "events": [{"summary": "<duplicate-calendar>"}],
    }]
    errors = scaled_hook.validate_state(task, observed, date(2030, 1, 2))
    assert len(errors) == 1
    assert errors[0].startswith("calendars mismatch:")
    assert "<missing-start>" in errors[0]


def test_scaled_materialization_is_deterministic_complete_and_separate(tmp_path):
    first = tmp_path / "one" / "tasks"
    second = tmp_path / "two" / "tasks"
    scaled_materializer.materialize(first)
    scaled_materializer.materialize(second)
    assert _tree(first) == _tree(second)
    assert scaled_materializer.DEFAULT_OUTPUT == REPO / "bench" / "distill-scaled" / "tasks"
    assert scaled_materializer.DEFAULT_OUTPUT != REPO / "bench" / "distill" / "tasks"
    assert sorted(path.name for path in first.iterdir()) == [f"s{i:03d}" for i in range(1, 126)]
    required = {"task.json", "prompt.txt", "setup.sh", "check.sh", "teardown.sh", "reference.sh"}
    for task_dir in first.iterdir():
        assert {path.name for path in task_dir.iterdir()} == required
        metadata = json.loads((task_dir / "task.json").read_text())
        assert metadata["id"] == task_dir.name
        assert metadata["operation_graph"] == scaled_hook.get_task(task_dir.name)["operation_graph"]


def test_scaled_wrappers_target_only_scaled_hook_and_correct_task(tmp_path):
    root = tmp_path / "tasks"
    scaled_materializer.materialize(root)
    for task_dir in root.iterdir():
        for command in ("setup", "check", "reference", "teardown"):
            wrapper = task_dir / f"{command}.sh"
            content = wrapper.read_text()
            assert "research/distill/scaled_hook.py" in content
            assert "research/distill/hook.py" not in content
            assert f" {command} {task_dir.name}\n" in content
            assert wrapper.stat().st_mode & 0o111


def test_heldout_literal_partitions_are_disjoint_from_training():
    train = defaultdict(set)
    validation = defaultdict(set)
    for task in tasks():
        target = validation if task["split"] == "validation" else train
        for key, value in task["literal_partition"].items():
            target[key].add(value)
    assert train.keys() == validation.keys()
    for key in train:
        assert train[key].isdisjoint(validation[key])
    assert validation == {"keyword": {"Saffron"}, "person": {"Omar"}}


def test_lineage_siblings_have_real_structural_and_prompt_diversity():
    lineages = defaultdict(list)
    for task in tasks():
        lineages[task["lineage"]].append(task)
    for lineage, siblings in lineages.items():
        assert len({task["prompt"] for task in siblings}) == 5, lineage
        axis_values = {
            key: {json.dumps(task["structural_axes"][key], sort_keys=True)
                  for task in siblings}
            for key in siblings[0]["structural_axes"]
        }
        varying = {key for key, values in axis_values.items() if len(values) > 1}
        assert "composition_level" in varying
        assert len(varying) >= 2, (lineage, varying)
        assert len({json.dumps(task["structural_axes"], sort_keys=True)
                    for task in siblings}) >= 3


def test_family_level_diversity_covers_encodings_depth_state_and_outputs():
    by_family = defaultdict(list)
    for task in tasks():
        by_family[task["family"]].append(task)
    for family, members in by_family.items():
        encodings = {tuple(task["structural_axes"]["input_encoding"]) for task in members}
        depths = {task["structural_axes"]["nesting_depth"] for task in members}
        output_shapes = {tuple(task["structural_axes"]["output_channels"]) for task in members}
        tag_sets = {tuple(task["difficulty_tags"]) for task in members}
        assert len(encodings) >= 5, family
        if family == "calendar":
            temporal_shapes = {
                tuple(task["structural_axes"]["temporal_minutes"])
                for task in members
            }
            assert len(temporal_shapes) >= 10
        else:
            assert len(depths) >= 3, family
        assert len(output_shapes) >= (4 if family == "cross_app" else 1), family
        assert len(tag_sets) == 5, family


def test_prompts_do_not_claim_unobservable_checker_dimensions():
    # The current state reader checks reminder names, event fields, and Mail
    # recipients/subject/body/cc/bcc, but not these dimensions.
    unobservable = ("reminder notes", "alarm", "attachment")
    for task in tasks():
        lowered = task["prompt"].casefold()
        assert not any(term in lowered for term in unobservable), task["id"]


def test_recursive_inventory_oracles_are_derived_from_staged_files():
    archetypes = {
        "notes-recursive-inventory",
        "reminders-recursive-files",
        "cross-inventory-count",
    }
    for task in tasks():
        if task["archetype"] not in archetypes:
            continue
        if task["archetype"] == "notes-recursive-inventory":
            basenames = [Path(path).name for path in task["files"] if Path(path).suffix]
            groups = defaultdict(list)
            for basename in basenames:
                groups[Path(basename).suffix.lower()].append(basename)
            oracle = "\n\n".join(
                f"[{extension}]\n" + "\n".join(sorted(groups[extension], key=str.casefold))
                for extension in sorted(groups)
            )
            assert task["expected_state"]["notes"][0]["body"] == oracle
        elif task["archetype"] == "reminders-recursive-files":
            extensions = [Path(path).suffix.casefold() for path in task["files"]]
            wanted = Counter(extensions).most_common(1)[0][0]
            relative = [path.removeprefix("input/docs/") for path in task["files"]
                        if Path(path).suffix.casefold() == wanted]
            oracle = [f"Review: {path}" for path in sorted(relative, key=str.casefold)]
            assert task["expected_state"]["reminder_lists"][0]["reminders"] == oracle
        else:
            body = task["expected_state"]["notes"][0]["body"]
            paths = body.splitlines()
            reminder = task["expected_state"]["reminder_lists"][0]["reminders"][0]
            assert reminder.endswith(f": {len(paths)}")
            assert paths == sorted(paths, key=str.casefold)


def test_all_calendar_event_ends_have_positive_bounded_durations():
    for task in tasks():
        for calendar in task["expected_state"].get("calendars", []):
            for event in calendar["events"]:
                if event["start"].startswith("$ANCHOR"):
                    start = "2030-01-01T" + event["start"][-5:]
                    end = "2030-01-01T" + event["end"][-5:]
                    if end <= start:
                        end = "2030-01-02T" + event["end"][-5:]
                else:
                    start, end = event["start"], event["end"]
                assert 1 <= _duration(start, end) <= 8 * 60, (task["id"], event)


def test_cross_app_tasks_really_have_multiple_checked_output_channels():
    for task in tasks():
        if task["family"] != "cross_app":
            continue
        assert len(task["expected_state"]) >= 2
        assert len(task["structural_axes"]["output_channels"]) >= 2
