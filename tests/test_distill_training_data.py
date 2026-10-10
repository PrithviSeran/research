"""Pure checks for the frozen Kimi-to-Qwen SFT dataset."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DISTILL = REPO / "research" / "distill"
sys.path.insert(0, str(DISTILL))

from extract_training_data import clean_messages, validate_tool_pairs  # noqa: E402


def test_trace_selection_has_101_unique_train_only_tasks_and_one_reserve():
    selection = json.loads((DISTILL / "training_trace_selection.json").read_text())
    selected = [
        task_id
        for batch in selection["accepted_batches"]
        for task_id in batch["task_ids"]
    ]
    assert len(selected) == len(set(selected)) == 101
    assert selection["target_records"] == 100
    assert selection["reserve_task_id"] == "x015"

    task_manifests = [json.loads((REPO / path).read_text()) for path in selection["task_manifests"]]
    tasks = {task["id"]: task for manifest in task_manifests for task in manifest}
    assert set(selected) <= set(tasks)
    assert all(tasks[task_id]["split"] == "train" for task_id in selected)
    heldout = {task["id"] for task in task_manifests[0] if task["split"] == "validation"}
    assert heldout.isdisjoint(selected)


def test_frozen_manifest_is_exactly_100_plus_one_hashed_reserve():
    manifest = json.loads((DISTILL / "training_manifest.json").read_text())
    records = manifest["records"]
    assert len(records) == 101
    assert Counter(record["partition"] for record in records) == {"train": 100, "reserve": 1}
    reserve = next(record for record in records if record["partition"] == "reserve")
    assert reserve["task_id"] == "x015"
    assert len({record["task_id"] for record in records}) == 101
    for record in records:
        for key in (
            "task_manifest_sha256",
            "task_definition_sha256",
            "result_sha256",
            "result_row_sha256",
            "transcript_sha256",
        ):
            assert len(record[key]) == 64
            int(record[key], 16)


def test_clean_messages_removes_private_reasoning_and_provider_call_indexes():
    raw = [
        {"role": "user", "content": "do it"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "private",
            "thinking": "also private",
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "applescript", "arguments": {"script": "return 1"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "1"},
        {"role": "assistant", "content": "done", "analysis": "private"},
    ]
    messages = clean_messages(raw, "system")
    serialized = json.dumps(messages)
    assert "private" not in serialized
    assert "index" not in messages[2]["tool_calls"][0]
    assert messages[2]["tool_calls"][0]["function"]["arguments"] == {"script": "return 1"}
    validate_tool_pairs(messages)


def test_exported_dataset_and_qa_report_are_training_ready():
    dataset = [
        json.loads(line)
        for line in (DISTILL / "training_sft.jsonl").read_text().splitlines()
        if line
    ]
    report = json.loads((DISTILL / "training_qa_report.json").read_text())
    assert len(dataset) == 100
    assert report["status"] == "pass"
    assert report["errors"] == []
    assert report["warnings"] == []
    assert all(report["checks"].values())
    assert Counter(item["metadata"]["family"] for item in dataset) == {
        "notes": 21,
        "reminders": 23,
        "calendar": 22,
        "mail": 15,
        "cross_app": 19,
    }
    assert all(item["messages"][0]["role"] == "system" for item in dataset)
    assert all(item["tools"] for item in dataset)
    for item in dataset:
        validate_tool_pairs(item["messages"])
        for message in item["messages"]:
            for call in message.get("tool_calls", []):
                assert isinstance(call["function"]["arguments"], dict)


def test_qwen_checkpoint_and_exact_tokenization_are_pinned():
    pin = json.loads((DISTILL / "qwen_checkpoint_pin.json").read_text())
    stats = json.loads((DISTILL / "qwen_token_stats.json").read_text())
    upstream = pin["upstream"]
    assert upstream["repository"] == "Qwen/Qwen3.5-9B"
    assert upstream["revision"] == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert upstream["weight_bytes"] == 19_306_216_416
    assert len(upstream["weight_shards"]) == 4
    assert pin["ollama"]["quantization"] == "Q4_K_M"
    assert pin["compatibility"]["tokenizer_chat_template_exact_match"] is True
    assert pin["ollama"]["embedded_chat_template_sha256"] == upstream["chat_template_sha256"]

    assert stats["status"] == "pass"
    assert stats["checkpoint"]["revision"] == upstream["revision"]
    assert stats["checkpoint"]["chat_template_sha256"] == upstream["chat_template_sha256"]
    assert stats["records"] == 100
    assert stats["tokens"] == {
        "total_per_epoch": 791_929,
        "min": 7_264,
        "median": 7_877,
        "p95": 8_962,
        "max": 9_616,
        "model_context_length": 262_144,
        "all_records_fit": True,
    }
    assert stats["serialization"]["enable_thinking"] is False
