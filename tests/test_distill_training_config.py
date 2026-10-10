"""Checks for the pinned Qwen LoRA training contract."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DISTILL = REPO / "research" / "distill"
sys.path.insert(0, str(DISTILL))

from training_preprocess import IGNORE_INDEX, build_training_example  # noqa: E402


def _load(name: str):
    return json.loads((DISTILL / name).read_text())


def test_training_config_pins_data_model_and_optimizer_math():
    config = _load("training_config.json")
    pin = _load("qwen_checkpoint_pin.json")
    stats = _load("qwen_token_stats.json")
    dataset = DISTILL / "training_sft.jsonl"

    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == config["data"]["dataset_sha256"]
    assert config["model"]["repository"] == pin["upstream"]["repository"]
    assert config["model"]["revision"] == pin["upstream"]["revision"]
    assert config["data"]["chat_template_sha256"] == pin["upstream"]["chat_template_sha256"]
    assert config["model"]["dtype"] == "bfloat16"
    assert config["data"]["loss"] == "assistant_only"
    assert config["data"]["truncation"] is False
    assert config["data"]["packing"] is False
    assert stats["tokens"]["max"] <= config["data"]["max_sequence_length"]

    optimization = config["optimization"]
    steps = math.ceil(
        config["data"]["records"] / optimization["gradient_accumulation_steps"]
    ) * optimization["epochs"]
    assert steps == optimization["expected_optimizer_steps"] == 50
    assert optimization["warmup_steps"] == 5


def test_lora_regex_targets_all_language_projections_and_no_frozen_towers():
    config = _load("training_config.json")
    pattern = re.compile(config["lora"]["target_modules_regex"])
    names = []
    for layer in range(32):
        attention = (
            ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
            if (layer + 1) % 4 == 0
            else [
                "linear_attn.in_proj_qkv",
                "linear_attn.in_proj_z",
                "linear_attn.in_proj_b",
                "linear_attn.in_proj_a",
                "linear_attn.out_proj",
            ]
        )
        projections = attention + ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
        names.extend(f"model.language_model.layers.{layer}.{name}" for name in projections)

    matched = [name for name in names if pattern.fullmatch(name)]
    assert len(matched) == config["lora"]["expected_target_module_count"] == 248
    assert not pattern.fullmatch("model.visual.blocks.0.attn.q_proj")
    assert not pattern.fullmatch("mtp.layers.0.self_attn.q_proj")
    assert not pattern.fullmatch("model.language_model.layers.0.linear_attn.conv1d")


def test_preprocessing_report_proves_every_record_is_supervised_without_truncation():
    config = _load("training_config.json")
    report = _load("training_preprocess_report.json")
    assert report["status"] == "pass"
    assert report["dataset_sha256"] == config["data"]["dataset_sha256"]
    assert report["records"] == 100
    assert report["tokens"]["total"] == 791_929
    assert report["tokens"]["max"] == 9_616
    assert report["tokens"]["max"] <= config["data"]["max_sequence_length"]
    assert report["supervision"]["total_tokens"] == 46_989
    assert report["supervision"]["records_with_supervision"] == 100
    assert report["supervision"]["tool_call_messages"] == 272
    assert report["supervision"]["tool_calls"] == 286


class _FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tools, tokenize, add_generation_prompt,
                            enable_thinking):
        assert tools and tokenize is False and enable_thinking is False
        text = "".join(
            (
                "<assistant><think></think>" + message.get("content", "") + "</assistant>"
                if message["role"] == "assistant"
                else f"<{message['role']}>{message.get('content', '')}</{message['role']}>"
            )
            for message in messages
        )
        if add_generation_prompt:
            text += "<assistant><think></think>"
        return text

    def encode(self, value, *, add_special_tokens):
        assert add_special_tokens is False
        return list(value.encode())


def test_assistant_only_mask_starts_after_generation_prefix():
    row = {
        "messages": [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "answer"},
        ],
        "tools": [{"type": "function", "function": {"name": "act"}}],
        "metadata": {"task_id": "probe"},
    }
    example = build_training_example(_FakeTokenizer(), row, 1024)
    first = next(index for index, label in enumerate(example["labels"]) if label != IGNORE_INDEX)
    rendered_prefix = _FakeTokenizer().apply_chat_template(
        row["messages"][:2],
        tools=row["tools"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert first == len(rendered_prefix.encode())
    supervised = bytes(label for label in example["labels"] if label != IGNORE_INDEX).decode()
    assert supervised == "answer</assistant>"
