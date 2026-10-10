"""Pure checks for the frozen pre/post-training evaluation contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
DISTILL = REPO / "research" / "distill"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


baseline = _load_module("distill_baseline_eval", DISTILL / "baseline_eval.py")


def _manifest():
    return json.loads((DISTILL / "baseline_eval_manifest.json").read_text())


def _row(manifest, task_id, model, digest, version, shared_hash):
    source = manifest["contract"]["source"]
    return {
        "task": task_id,
        "tool": "hunch-ollama",
        "model": model,
        "model_digest": digest,
        "ollama_version": version,
        "shared_config_sha256": shared_hash,
        "recipe_sha256": source["recipe_sha256"],
        "source_commit": source["git_commit"] + "+" + source["python_tree_sha256"],
        "bench_code_sha256": manifest["contract"]["harness"]["sha256"],
        "status": "completed",
        "success": False,
        "scored": True,
        "teardown_ok": True,
        "focus_switches": 0,
        "cursor_moves": 0,
        "transcript": f"{task_id}.stream.jsonl",
    }


def test_frozen_manifest_matches_every_current_dependency_byte():
    frozen = _manifest()
    assert frozen == baseline.build_manifest()
    assert frozen["contract_sha256"] == baseline.canonical_hash(frozen["contract"])
    assert frozen["suites"]["primary"]["task_count"] == 25
    assert frozen["suites"]["primary"]["task_ids"] == list(baseline.PRIMARY_IDS)
    assert frozen["suites"]["regression"]["task_count"] == 25
    assert len(frozen["suites"]["regression"]["excluded"]) == 8


def test_primary_suite_is_validation_only_and_disjoint_from_training_selection():
    manifest = _manifest()
    scaled = json.loads((DISTILL / "scaled_tasks.json").read_text())
    splits = {task["id"]: task["split"] for task in scaled}
    primary = set(manifest["suites"]["primary"]["task_ids"])
    assert all(splits[task_id] == "validation" for task_id in primary)

    selection = json.loads((DISTILL / "training_trace_selection.json").read_text())
    accepted = {
        task_id for batch in selection["accepted_batches"] for task_id in batch["task_ids"]
    }
    assert primary.isdisjoint(accepted)


def test_materialized_validation_tasks_declare_transactional_lifecycle():
    manifest = _manifest()
    root = baseline.resolve_suite_root(manifest["suites"]["primary"])
    for task_id in manifest["suites"]["primary"]["task_ids"]:
        metadata = json.loads((root / task_id / "task.json").read_text())
        assert metadata["state_contract"] == "transactional-v1"


def test_nonbaseline_model_requires_an_explicit_content_digest():
    manifest = _manifest()
    with pytest.raises(ValueError, match="requires --expected-digest"):
        baseline.expected_runtime(manifest, "hunch-qwen-lora:latest", None, None)
    assert baseline.expected_runtime(
        manifest, "hunch-qwen-lora:v1", "abc123", "0.32.5"
    ) == ("abc123", "0.32.5")


def test_audit_accepts_capability_failures_but_rejects_contaminated_telemetry(tmp_path):
    manifest = _manifest()
    model_info = manifest["contract"]["model"]
    model = model_info["tag"]
    digest = model_info["digest"]
    version = model_info["ollama_version"]
    shared_hash = baseline.canonical_hash(
        baseline.runtime_contract(manifest, model, digest, version)
    )
    result_files = {}
    for name, suite in manifest["suites"].items():
        path = tmp_path / f"{name}.jsonl"
        rows = [
            _row(manifest, task_id, model, digest, version, shared_hash)
            for task_id in suite["task_ids"]
        ]
        for row in rows:
            events = [
                {"type": "conversation", "messages": []},
                {
                    "type": "result",
                    "model_digest": digest,
                    "ollama_version": version,
                    "shared_config_sha256": shared_hash,
                },
            ]
            (tmp_path / row["transcript"]).write_text(
                "".join(json.dumps(event) + "\n" for event in events)
            )
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        result_files[name] = path

    report = baseline.audit_results(
        manifest, result_files, model, digest, version, shared_hash
    )
    assert report["status"] == "pass"
    assert report["passed"] == 0
    assert report["rows"] == 50

    rows = [json.loads(line) for line in result_files["primary"].read_text().splitlines()]
    rows[0]["cursor_moves"] = 1
    result_files["primary"].write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    report = baseline.audit_results(
        manifest, result_files, model, digest, version, shared_hash
    )
    assert report["status"] == "fail"
    assert any("cursor moves=1" in error for error in report["errors"])
