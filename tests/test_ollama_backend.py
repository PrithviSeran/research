"""Ollama backend + provider tests — all around the one live transport touch-point
(_chat), faked here so no Ollama server is needed. Mirrors test_codex_backend.py."""
import json
import urllib.error

import pytest

import hunch.agent as agent_mod
from hunch.agent import AGENT_TOOLS
from hunch.backends import NAMES, get
from hunch.backends.ollama import (OllamaBackend, OllamaSessionConfig,
                                   OllamaDeadlineExceeded, OllamaTool)
from hunch.providers import provider, PROVIDERS, OllamaProvider
from hunch.errors import HunchError


class _FakeHunch:
    def __init__(self):
        self._app_id = None


def _backend(responses, dispatch=None, monkeypatch=None):
    """An OllamaBackend whose _chat pops canned responses and whose tool dispatch is
    recorded (default: every tool returns 'ok')."""
    b = OllamaBackend(_FakeHunch())
    b._sent = []
    seq = list(responses)
    b._chat = lambda body: (b._sent.append(body), seq.pop(0))[1]
    if monkeypatch is not None:
        calls = []
        def fake_dispatch(mac, name, args):
            calls.append((name, args))
            return (dispatch or (lambda n, a: "ok"))(name, args), False
        monkeypatch.setattr(agent_mod, "_dispatch_core", fake_dispatch)
        b._calls = calls
    return b


def _msg(content="", tool_calls=None):
    m = {"message": {"role": "assistant", "content": content}}
    if tool_calls:
        m["message"]["tool_calls"] = tool_calls
    return m


# ── registry ──────────────────────────────────────────────────────────────────
def test_registry_and_metadata():
    assert "ollama" in NAMES
    assert get("ollama") is OllamaBackend
    assert "ollama" in PROVIDERS and isinstance(provider("ollama"), OllamaProvider)
    assert (OllamaBackend.name, OllamaBackend.provider) == ("ollama", "ollama")
    assert OllamaBackend.deps_installed() is True          # stdlib-only


def test_tools_are_agent_tools_in_function_shape():
    tools = OllamaBackend._tools()
    assert [t["function"]["name"] for t in tools] == [t["name"] for t in AGENT_TOOLS]
    assert all(t["type"] == "function" for t in tools)
    assert tools[0]["function"]["parameters"] == AGENT_TOOLS[0]["input_schema"]


def test_tool_profiles_are_scoped_and_read_at_call_time(monkeypatch):
    monkeypatch.setenv("HUNCH_OLLAMA_TOOL_PROFILE", "ax")
    ax = {t["function"]["name"] for t in OllamaBackend._tools()}
    assert {"snapshot", "find", "act"} <= ax
    assert "web_open" not in ax and "applescript" not in ax
    assert "simultaneous_mode" not in ax and "request_focus" not in ax
    monkeypatch.setenv("HUNCH_OLLAMA_TOOL_PROFILE", "web")
    web = {t["function"]["name"] for t in OllamaBackend._tools()}
    assert {"web_open", "web_snapshot", "web_act"} <= web
    assert "snapshot" not in web
    monkeypatch.setenv("HUNCH_OLLAMA_TOOL_PROFILE", "native")
    native = {t["function"]["name"] for t in OllamaBackend._tools()}
    assert "request_focus" not in native and "simultaneous_mode" not in native


def test_instance_config_overrides_environment_and_adds_internal_tool(monkeypatch):
    monkeypatch.setenv("HUNCH_OLLAMA_TOOL_PROFILE", "web")
    monkeypatch.setenv("HUNCH_OLLAMA_MODEL", "wrong-model")
    delegate = OllamaTool.create("delegate", "delegate once", {
        "type": "object", "properties": {"objective": {"type": "string"}},
        "required": ["objective"]})
    config = OllamaSessionConfig(
        role="coordinator", system_prompt="route only", tool_names=frozenset(),
        allow_delegation=True, internal_tools=(delegate,), model="fixed-model")
    b = OllamaBackend(_FakeHunch(), session_config=config,
                      dispatcher=lambda name, args: ("ok", False))
    b._chat = lambda body: (_ for _ in ()).throw(AssertionError(body))
    tools = b._session_tools()
    assert [tool["function"]["name"] for tool in tools] == ["delegate"]
    assert config.model == "fixed-model" and b.session_config.system_prompt == "route only"


def test_configured_sessions_isolate_prompt_tools_and_history(monkeypatch):
    a = OllamaBackend(_FakeHunch(), session_config=OllamaSessionConfig(
        role="a", system_prompt="prompt-a", tool_names=frozenset({"list_apps"})))
    b = OllamaBackend(_FakeHunch(), session_config=OllamaSessionConfig(
        role="b", system_prompt="prompt-b", tool_names=frozenset({"snapshot"})))
    a._chat = lambda body: _msg("done-a")
    b._chat = lambda body: _msg("done-b")
    a.run("task-a")
    b.run("task-b")
    assert a.last_system == "prompt-a" and b.last_system == "prompt-b"
    assert {t["function"]["name"] for t in a._session_tools()} == {"list_apps"}
    assert {t["function"]["name"] for t in b._session_tools()} == {"snapshot"}
    assert a.messages[0]["content"] == "task-a" and b.messages[0]["content"] == "task-b"


def test_direct_native_session_loads_versioned_recipes_but_ax_does_not(monkeypatch):
    monkeypatch.setenv("HUNCH_OLLAMA_TOOL_PROFILE", "native")
    native = _backend([_msg("done")] * 3)
    native.run("task")
    assert "Verified operation recipes for Qwen" in native.last_system
    monkeypatch.setenv("HUNCH_OLLAMA_TOOL_PROFILE", "ax")
    ax = _backend([_msg("done")] * 3)
    ax.run("task")
    assert "Verified operation recipes for Qwen" not in ax.last_system
    monkeypatch.setenv("HUNCH_OLLAMA_TOOL_PROFILE", "native")
    monkeypatch.setenv("HUNCH_OLLAMA_RECIPES", "0")
    disabled = _backend([_msg("done")] * 3)
    disabled.run("task")
    assert "Verified operation recipes for Qwen" not in disabled.last_system


def test_session_config_freezes_mutable_inputs():
    names = {"snapshot"}
    tools = []
    config = OllamaSessionConfig(role="ax", system_prompt="x", tool_names=names,
                                 internal_tools=tools)
    names.add("applescript")
    tools.append(OllamaTool.create("private", "x", {"type": "object"}))
    assert config.tool_names == frozenset({"snapshot"}) and config.internal_tools == ()


def test_deadline_exception_becomes_timeout_result():
    b = OllamaBackend(_FakeHunch())
    b._chat = lambda body: (_ for _ in ()).throw(
        OllamaDeadlineExceeded("deadline"))
    result = b.run("task", deadline=10**20)
    assert result.aborted and result.stop_reason == "timeout"


def test_injected_dispatcher_handles_private_tool_without_core(monkeypatch):
    calls = []
    monkeypatch.setattr(agent_mod, "_dispatch_core", lambda *a: pytest.fail("core called"))
    tool = OllamaTool.create("delegate", "delegate", {"type": "object"})
    b = OllamaBackend(_FakeHunch(), session_config=OllamaSessionConfig(
        role="coordinator", system_prompt="route", tool_names=frozenset(),
        allow_delegation=True, internal_tools=(tool,)),
        dispatcher=lambda name, args: (calls.append((name, args)) or "receipt", False))
    b._chat = iter([
        _msg(tool_calls=[{"function": {"name": "delegate", "arguments": {"objective": "x"}}}]),
        _msg("done"),
    ]).__next__
    # _chat normally receives a body; adapt the iterator without weakening the assertion.
    next_response = b._chat
    b._chat = lambda body: next_response()
    result = b.run("route it")
    assert not result.aborted and calls == [("delegate", {"objective": "x"})]


def test_instance_scope_rejects_before_injected_dispatch():
    calls = []
    b = OllamaBackend(_FakeHunch(), session_config=OllamaSessionConfig(
        role="ax", system_prompt="ax", tool_names=frozenset({"snapshot"})),
        dispatcher=lambda name, args: (calls.append(name) or "ok", False))
    responses = iter([
        _msg(tool_calls=[{"function": {"name": "applescript", "arguments": {"script": "x"}}}]),
        _msg("done"),
    ])
    b._chat = lambda body: next(responses)
    result = b.run("x")
    assert calls == [] and result.usage["rejected_tool_calls"] == 1


# ── the loop ──────────────────────────────────────────────────────────────────
def test_run_native_tool_loop(monkeypatch):
    b = _backend([
        _msg(tool_calls=[{"function": {"name": "list_apps", "arguments": {}}}]),
        _msg("done — Notes is running."),
    ], monkeypatch=monkeypatch)
    events = []
    r = b.run("which apps run?", on_event=lambda k, d: events.append((k, d)))
    assert b._calls == [("list_apps", {})]
    assert r.text == "done — Notes is running." and not r.aborted
    assert r.stop_reason == "end_turn" and r.turns == 2
    assert ("tool", {"name": "list_apps", "input": {}}) in events
    assert events[-1] == ("done", "done — Notes is running.")
    # request shape: playbook system message, think off, explicit num_ctx, tools attached
    body = b._sent[0]
    assert body["think"] is False and body["options"]["num_ctx"] > 0
    assert body["messages"][0]["role"] == "system"
    assert "num_ctx" in body["options"] and body["tools"]
    # tool result went back as a role='tool' message with the tool's name
    tool_msgs = [m for m in b._sent[1]["messages"] if m.get("role") == "tool"]
    assert tool_msgs == [{"role": "tool", "tool_name": "list_apps", "content": "ok"}]


def test_run_parses_string_arguments(monkeypatch):
    b = _backend([
        _msg(tool_calls=[{"function": {"name": "snapshot",
                                       "arguments": json.dumps({"app": "Notes"})}}]),
        _msg("ok"),
    ], monkeypatch=monkeypatch)
    b.run("look at Notes")
    assert b._calls == [("snapshot", {"app": "Notes"})]


def test_fallback_parses_prose_tool_call(monkeypatch):
    # The measured qwen3.5 failure mode: a well-formed call in a markdown fence
    # instead of the native channel. It must be recovered AND counted.
    prose = ('I will inspect the app.\n```json\n'
             '{"name": "snapshot", "arguments": {"app": "Notes"}}\n```')
    b = _backend([_msg(prose), _msg("finished")], monkeypatch=monkeypatch)
    r = b.run("look at Notes")
    assert b._calls == [("snapshot", {"app": "Notes"})]
    assert r.usage["fallback_tool_calls"] == 1 and r.text == "finished"


def test_fallback_ignores_non_tool_json(monkeypatch):
    # 3 identical responses: the no-tool-call reply is nudged twice, then accepted.
    resp = _msg('the config is ```json\n{"volume": 11}\n``` — done')
    b = _backend([resp, resp, resp], monkeypatch=monkeypatch)
    r = b.run("read config")
    assert b._calls == [] and r.stop_reason == "end_turn"
    assert r.usage["fallback_tool_calls"] == 0 and r.usage["continue_nudges"] == 2


def test_strict_disables_fallback(monkeypatch):
    monkeypatch.setenv("HUNCH_OLLAMA_STRICT", "1")
    prose = '```json\n{"name": "list_apps", "arguments": {}}\n```'
    resp = _msg(prose)
    b = _backend([resp, resp, resp], monkeypatch=monkeypatch)
    r = b.run("apps?")
    assert b._calls == [] and r.stop_reason == "invalid_tool_channel" and r.aborted
    assert r.usage["prose_tool_calls"] == 3
    assert r.usage["continue_nudges"] == 2


def test_malformed_native_arguments_are_rejected_not_dispatched(monkeypatch):
    b = _backend([
        _msg(tool_calls=[{"function": {"name": "snapshot", "arguments": "{"}}]),
        _msg("done"),
    ], monkeypatch=monkeypatch)
    r = b.run("look")
    assert b._calls == [] and r.usage["malformed_tool_calls"] == 1
    result = [m for m in b.messages if m.get("role") == "tool"][0]
    assert "malformed tool arguments" in result["content"]


def test_out_of_profile_call_is_rejected(monkeypatch):
    monkeypatch.setenv("HUNCH_OLLAMA_TOOL_PROFILE", "ax")
    b = _backend([
        _msg(tool_calls=[{"function": {"name": "applescript", "arguments": {"script": "x"}}}]),
        _msg("done"),
    ], monkeypatch=monkeypatch)
    r = b.run("look")
    assert b._calls == [] and r.usage["rejected_tool_calls"] == 1
    assert "not available in the ax profile" in b.messages[-2]["content"]


def test_three_distinct_applescript_compile_errors_abort_early(monkeypatch):
    calls = [{"function": {"name": "applescript",
                            "arguments": {"script": f"broken {index}"}}}
             for index in range(3)]
    error = "AppleScript error: probe.scpt: script error: Expected expression. (-2741)"
    b = _backend([_msg(tool_calls=[call]) for call in calls],
                 dispatch=lambda name, args: error, monkeypatch=monkeypatch)
    result = b.run("run a script")
    assert result.aborted and result.stop_reason == "applescript_compile_loop"
    assert result.turns == 3 and len(b._calls) == 3
    assert result.usage["applescript_compile_errors"] == 3
    assert result.usage["applescript_compile_aborts"] == 1
    assert any("APPLESCRIPT_COMPILE_LOOP" in message.get("content", "")
               for message in b.messages if message.get("role") == "tool")


def test_applescript_access_error_does_not_advance_compile_cutoff(monkeypatch):
    calls = [{"function": {"name": "applescript",
                            "arguments": {"script": f"probe {index}"}}}
             for index in range(3)]
    access_error = ("AppleScript error: Not authorized to send Apple events to "
                    "System Events. (-1723)")
    b = _backend([*[_msg(tool_calls=[call]) for call in calls], _msg("blocked")],
                 dispatch=lambda name, args: access_error, monkeypatch=monkeypatch)
    result = b.run("run a script")
    assert not result.aborted and result.stop_reason == "end_turn"
    assert result.usage["applescript_compile_errors"] == 0
    assert result.usage["applescript_compile_aborts"] == 0


def test_announce_without_act_is_nudged(monkeypatch):
    # The measured 04/12 failure: a plan narration with no tool call must be pushed
    # back into acting, and the nudge counted.
    b = _backend([
        _msg("I'll read the file and copy the total."),
        _msg(tool_calls=[{"function": {"name": "list_apps", "arguments": {}}}]),
        _msg("done"),
    ], monkeypatch=monkeypatch)
    r = b.run("do it")
    assert b._calls == [("list_apps", {})]
    assert r.usage["continue_nudges"] == 1 and r.text == "done" and not r.aborted
    # once a tool has run, a no-call response ends the episode (no further nudge)
    assert [m["content"] for m in b.messages if m.get("role") == "user"][1].startswith(
        "You have not called any tool")


def test_screenshot_bytes_ride_images_channel(monkeypatch):
    b = _backend([
        _msg(tool_calls=[{"function": {"name": "screenshot", "arguments": {}}}]),
        _msg("saw it"),
    ], dispatch=lambda n, a: b"\x89PNG...", monkeypatch=monkeypatch)
    b.run("what is on screen?")
    tool_msg = [m for m in b._sent[1]["messages"] if m.get("role") == "tool"][0]
    assert tool_msg["images"] and "screenshot" in tool_msg["content"]


def test_max_turns_aborts(monkeypatch):
    call = _msg(tool_calls=[{"function": {"name": "list_apps", "arguments": {}}}])
    b = _backend([call, call], monkeypatch=monkeypatch)
    r = b.run("loop forever", max_turns=2)
    assert r.aborted and r.stop_reason == "max_turns" and r.turns == 2


def test_identical_action_and_result_warns_then_aborts(monkeypatch):
    call = _msg(tool_calls=[{"function": {"name": "list_apps", "arguments": {}}}])
    b = _backend([call, call, call], monkeypatch=monkeypatch)
    r = b.run("do not loop", max_turns=5)
    assert r.aborted and r.stop_reason == "no_progress" and r.turns == 3
    assert r.usage["no_progress_warnings"] == 1
    assert r.usage["no_progress_aborts"] == 1
    assert len(b._calls) == 3


def test_context_is_compacted_but_full_transcript_is_retained(monkeypatch):
    monkeypatch.setenv("HUNCH_OLLAMA_HISTORY_CHARS", "4000")
    monkeypatch.setenv("HUNCH_OLLAMA_TOOL_RESULT_CHARS", "1000")
    b = OllamaBackend(_FakeHunch())
    b.messages = ([{"role": "user", "content": "old" * 2000},
                   {"role": "assistant", "content": "reply" * 1000}]
                  + [{"role": "user", "content": "current task"}])
    usage = {"context_compactions": 0, "context_messages_pruned": 0,
             "tool_results_truncated": 0, "max_context_chars": 0}
    context = b._context_messages(usage)
    assert len(b.messages) == 3
    assert context[0]["role"] == "system" and "Earlier history compacted" in context[0]["content"]
    assert context[-1]["content"] == "current task"
    assert usage["context_messages_pruned"] == 2


def test_usage_accumulates_and_reset_clears():
    resp = {**_msg("hi"), "prompt_eval_count": 100, "eval_count": 7}
    b = _backend([resp, resp, resp])       # nudged twice, so three requests total
    r = b.run("hello")
    assert (r.usage["input_tokens"], r.usage["output_tokens"]) == (300, 21)
    assert b.messages          # conversation kept for continuation
    b.reset()
    assert b.messages == []


def test_chat_retries_transient_http_errors(monkeypatch):
    # qwen3.5's tool-template parse flake: Ollama 500s once, then the retry lands.
    import io
    attempts = []
    def flaky_urlopen(req, timeout=None):
        attempts.append(1)
        if len(attempts) < 3:
            raise urllib.error.HTTPError(
                "u", 500, "err", {},
                io.BytesIO(b'{"error": "XML syntax error on line 5"}'))
        class _R:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b'{"message": {"role": "assistant", "content": "hi"}}'
        return _R()
    monkeypatch.setattr("urllib.request.urlopen", flaky_urlopen)
    out = OllamaBackend(_FakeHunch())._chat({"model": "m"})
    assert out["message"]["content"] == "hi" and len(attempts) == 3


def test_chat_gives_up_after_retries(monkeypatch):
    import io
    def always_500(req, timeout=None):
        raise urllib.error.HTTPError("u", 500, "err", {},
                                     io.BytesIO(b'{"error": "XML syntax error"}'))
    monkeypatch.setattr("urllib.request.urlopen", always_500)
    with pytest.raises(HunchError) as e:
        OllamaBackend(_FakeHunch())._chat({"model": "m"})
    assert "XML syntax error" in str(e.value)


def test_no_server_raises_friendly_hint(monkeypatch):
    b = OllamaBackend(_FakeHunch())
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: (_ for _ in ()).throw(urllib.error.URLError("refused")))
    with pytest.raises(HunchError) as e:
        b.run("hi")
    assert "ollama serve" in str(e.value)
    assert OllamaBackend.available() is False


# ── provider: reachability-as-auth ────────────────────────────────────────────
def test_provider_status_and_noop_auth(monkeypatch):
    p = OllamaProvider()
    monkeypatch.setattr(OllamaBackend, "available", classmethod(lambda cls, app_id=None: True))
    st = p.status()
    assert (st.provider, st.logged_in, st.method) == ("ollama", True, "local")
    assert p.login().logged_in is True
    assert p.logout() is None
    monkeypatch.setattr(OllamaBackend, "available", classmethod(lambda cls, app_id=None: False))
    assert p.status().logged_in is False
    with pytest.raises(HunchError):
        p.login()


def test_provider_make_backend():
    assert isinstance(OllamaProvider().make_backend(_FakeHunch()), OllamaBackend)
