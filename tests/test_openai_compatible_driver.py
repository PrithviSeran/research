"""OpenAI-compatible benchmark driver tests; no API or Mac apps are touched."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
DRIVER_PATH = REPO / "research" / "distill" / "openai_compatible_driver.py"
spec = importlib.util.spec_from_file_location("openai_compatible_driver", DRIVER_PATH)
driver = importlib.util.module_from_spec(spec)
assert spec.loader
sys.modules[spec.name] = driver
spec.loader.exec_module(driver)


TOOLS = [{
    "type": "function",
    "function": {
        "name": "list_apps",
        "description": "list apps",
        "parameters": {"type": "object", "properties": {}},
    },
}]


def response(message, *, prompt=100, completion=20, cached=60,
             finish_reason="tool_calls"):
    return {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                  "prompt_tokens_details": {"cached_tokens": cached}},
    }


def test_loads_only_requested_dotenv_value(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# local\nOTHER=do-not-return\nexport MOONSHOT_AI='moon-key'\n")
    monkeypatch.delenv("MOONSHOT_AI", raising=False)
    assert driver.load_env_value(env, "MOONSHOT_AI") == "moon-key"
    assert driver.resolve_api_key("MOONSHOT_AI", env) == "moon-key"
    assert "OTHER" not in __import__("os").environ


def test_environment_value_wins_over_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("MOONSHOT_AI=file-key\n")
    monkeypatch.setenv("MOONSHOT_AI", "process-key")
    assert driver.resolve_api_key("MOONSHOT_AI", env) == "process-key"


def test_missing_secret_error_names_variable_but_not_values(tmp_path, monkeypatch):
    monkeypatch.delenv("MOONSHOT_AI", raising=False)
    with pytest.raises(driver.DriverError, match="MOONSHOT_AI"):
        driver.resolve_api_key("MOONSHOT_AI", tmp_path / "missing")


def test_openai_tool_shape_preserves_native_schema():
    converted = driver.openai_tools([{
        "name": "snapshot", "description": "read", "input_schema": {
            "type": "object", "properties": {"app": {"type": "string"}}},
    }])
    assert converted == [{"type": "function", "function": {
        "name": "snapshot", "description": "read", "parameters": {
            "type": "object", "properties": {"app": {"type": "string"}}}}}]


def test_preserves_complete_assistant_message_across_tool_turns():
    assistant = {
        "role": "assistant",
        "reasoning_content": "I need to inspect the running apps first.",
        "content": "",
        "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "list_apps", "arguments": "{}"},
        }],
    }
    final = {"role": "assistant", "reasoning_content": "The tool answered.",
             "content": "Done.", "tool_calls": []}
    replies = iter([response(assistant),
                    response(final, prompt=140, completion=10, cached=100,
                             finish_reason="stop")])
    bodies, calls, events = [], [], []

    def chat(body):
        bodies.append(body)
        return next(replies)

    loop = driver.TeacherLoop(
        chat, lambda name, args: (calls.append((name, args)) or "Finder, Notes", False),
        TOOLS, events.append)
    result = loop.run("List apps", system="real system", model="kimi-k3",
                      reasoning_effort="low", max_turns=4,
                      max_completion_tokens=1000)

    assert not result.aborted and result.text == "Done." and result.turns == 2
    assert calls == [("list_apps", {})]
    assert bodies[0]["tool_choice"] == "required"
    assert "tool_choice" not in bodies[1]
    assert bodies[1]["messages"][2] == assistant
    assert bodies[1]["messages"][3] == {
        "role": "tool", "tool_call_id": "call_1", "content": "Finder, Notes"}
    assert result.messages[1] == assistant
    assert result.usage == {
        "input_tokens": 240, "output_tokens": 30,
        "cached_input_tokens": 160, "uncached_input_tokens": 80,
        "tool_calls": 1, "malformed_tool_calls": 0,
        "rejected_tool_calls": 0,
    }
    # Reasoning is retained in the conversation but never printed as a progress block.
    assert all("reasoning_content" not in json.dumps(event) for event in events)


def test_multiple_native_calls_get_matching_tool_results():
    first = {"role": "assistant", "content": None, "reasoning_content": "two reads",
             "tool_calls": [
                 {"id": "a", "function": {"name": "list_apps", "arguments": "{}"}},
                 {"id": "b", "function": {"name": "list_apps", "arguments": {}}},
             ]}
    replies = iter([response(first), response(
        {"role": "assistant", "content": "done", "reasoning_content": "ok"},
        finish_reason="stop")])
    loop = driver.TeacherLoop(lambda body: next(replies),
                              lambda name, args: ("ok", False), TOOLS, lambda event: None)
    result = loop.run("x", system="s", model="kimi-k3", reasoning_effort="low",
                      max_turns=3, max_completion_tokens=100)
    tool_messages = [message for message in result.messages if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in tool_messages] == ["a", "b"]


def test_malformed_and_unknown_calls_are_not_dispatched():
    first = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "bad", "function": {"name": "list_apps", "arguments": "[1]"}},
        {"id": "nope", "function": {"name": "shell", "arguments": "{}"}},
    ]}
    replies = iter([response(first), response(
        {"role": "assistant", "content": "could not act"}, finish_reason="stop")])
    dispatched = []
    loop = driver.TeacherLoop(lambda body: next(replies),
                              lambda n, a: (dispatched.append((n, a)) or "ok", False),
                              TOOLS, lambda event: None)
    result = loop.run("x", system="s", model="kimi-k3", reasoning_effort="low",
                      max_turns=3, max_completion_tokens=100)
    assert dispatched == []
    assert result.usage["malformed_tool_calls"] == 1
    assert result.usage["rejected_tool_calls"] == 1


def test_max_turn_cutoff_is_an_error_with_complete_trace():
    tool_message = {"role": "assistant", "content": "", "reasoning_content": "again",
                    "tool_calls": [{"id": "x", "function": {
                        "name": "list_apps", "arguments": "{}"}}]}
    loop = driver.TeacherLoop(lambda body: response(tool_message),
                              lambda name, args: ("same", False), TOOLS,
                              lambda event: None)
    result = loop.run("x", system="s", model="kimi-k3", reasoning_effort="low",
                      max_turns=2, max_completion_tokens=100)
    assert result.aborted and result.stop_reason == "max_turns" and result.turns == 2
    assert len(result.messages) == 5  # user + 2 assistant/tool pairs


def test_trace_elides_image_payload_but_runtime_history_keeps_it():
    url = "data:image/png;base64," + "A" * 100
    original = [{"role": "tool", "content": [{"type": "image_url",
                                                  "image_url": {"url": url}}]}]
    safe = driver._trace_safe(original)
    assert original[0]["content"][0]["image_url"]["url"] == url
    assert safe[0]["content"][0]["image_url"]["url"].startswith("<elided")


def test_ordered_system_suffixes_are_joined_once(tmp_path):
    policy = tmp_path / "policy.md"
    examples = tmp_path / "examples.md"
    policy.write_text("  narrow shared reads  \n")
    examples.write_text("worked example\n")
    assert driver.load_system_suffix([policy, examples]) == (
        "narrow shared reads\n\nworked example")
    assert driver.load_system_suffix(None) == ""


def test_tool_result_event_records_native_dispatch_window():
    first = {"role": "assistant", "content": "", "tool_calls": [{
        "id": "timed", "function": {"name": "list_apps", "arguments": "{}"},
    }]}
    replies = iter([response(first), response(
        {"role": "assistant", "content": "done"}, finish_reason="stop")])
    events = []
    loop = driver.TeacherLoop(lambda body: next(replies),
                              lambda name, args: ("Notes", False),
                              TOOLS, events.append)
    loop.run("x", system="s", model="kimi-k3", reasoning_effort="low",
             max_turns=3, max_completion_tokens=100)
    event = next(item for item in events if item["type"] == "tool_result")
    assert event["tool_call_id"] == "timed"
    assert event["dispatched"] is True and event["is_error"] is False
    assert event["dispatch_started_monotonic_s"] <= event["dispatch_ended_monotonic_s"]
    assert event["dispatch_duration_s"] >= 0


def test_cost_uses_cached_uncached_and_output_rates():
    usage = {"uncached_input_tokens": 20_000, "cached_input_tokens": 30_000,
             "output_tokens": 6_000}
    prices = {"input": 3.0, "cached_input": 0.3, "output": 15.0}
    assert driver.estimate_cost(usage, prices) == 0.159


def test_connection_check_has_no_tools_and_small_output_budget():
    bodies = []
    client = lambda body: (bodies.append(body) or response(
        {"role": "assistant", "content": "OK"}, finish_reason="stop"))
    driver.connection_check(client, "kimi-k3", "low")
    assert "tools" not in bodies[0]
    assert bodies[0]["max_completion_tokens"] == 64


def test_provider_error_sanitizes_secret_and_account_identifiers():
    raw = ('{"error":{"message":"account org-abc123<ak-keyid987> rejected '
           'secret-value"}}')
    detail = driver._sanitized_http_detail(raw, "secret-value")
    assert "secret-value" not in detail
    assert "org-abc123" not in detail
    assert "ak-keyid987" not in detail
    assert "<redacted>" in detail and "<org>" in detail and "<key-id>" in detail


def test_cross_process_pacer_waits_to_enforce_rpm(tmp_path, monkeypatch):
    times = iter([100.0, 100.0, 100.0, 100.0, 120.25])
    sleeps = []
    monkeypatch.setattr(driver.time, "time", lambda: next(times))
    monkeypatch.setattr(driver.time, "sleep", sleeps.append)
    pacer = driver.RequestPacer(3, "provider", tmp_path)
    pacer.wait()
    pacer.wait()
    assert sleeps == [20.25]


def test_kimi_adapter_uses_generic_portable_driver_and_secret_name():
    versioned = REPO / "research/distill/hunch-kimi.adapter.json"
    runtime = REPO / "bench/adapters/hunch-kimi.json"
    if runtime.exists():
        assert runtime.read_bytes() == versioned.read_bytes()
    adapter = json.loads(versioned.read_text())
    command = adapter["driver"]["command"]
    assert command[:2] == ["{python}",
                           "{repo}/research/distill/openai_compatible_driver.py"]
    assert command[command.index("--api-key-env") + 1] == "MOONSHOT_AI"
    assert command[command.index("--model") + 1] == "{model}"
    suffix = command[command.index("--system-suffix-file") + 1]
    assert suffix == "{repo}/research/distill/teacher_collection_policy.md"
    policy = REPO / "research/distill/teacher_collection_policy.md"
    text = policy.read_text()
    assert "Never enumerate or return all Notes" in text
    assert "Never use `send`" in text
    assert "Do not call `close`" in text
    assert "whose subject starts with taskPrefix" in text
    assert "Do not use `open_file` to inspect them" in text
    assert "set day to 1 before changing year/month/day" in text
    assert "do not rewrite" in text
    assert "external checker normalizes it" in text
    assert "{prompt}" in command
    assert not any("key=" in value.casefold() for value in command)
