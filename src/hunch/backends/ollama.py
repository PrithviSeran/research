"""backends/ollama.py — the 'ollama' backend: a LOCAL model drives this Mac.

Runs the same agent loop as the 'api' backend — HUNCH_PLAYBOOK as the system
prompt, AGENT_TOOLS as native tool schemas, tools executed in-process against the
caller's live Hunch instance via _dispatch_core — but the model is served by a
local Ollama daemon (http://127.0.0.1:11434), so there is no account, no API key,
and no per-token cost. Stdlib-only: the transport is urllib against Ollama's
/api/chat, no optional package to install.

Knobs (env):
  HUNCH_OLLAMA_MODEL    model tag (default 'qwen3.5:9b')
  HUNCH_OLLAMA_HOST     server base URL (default 'http://127.0.0.1:11434')
  HUNCH_OLLAMA_NUM_CTX  context window (default 32768 — Ollama's own default can
                        be far smaller than the model supports, which would
                        silently truncate the playbook + AX snapshots)
  HUNCH_OLLAMA_STRICT   '1' requires native tool calls and rejects prose calls
  HUNCH_OLLAMA_TOOL_PROFILE  'all', 'ax', 'web', or 'native' (default 'all')
  HUNCH_OLLAMA_HISTORY_CHARS approximate replay budget (default 48000)
  HUNCH_OLLAMA_TOOL_RESULT_CHARS per-result replay cap (default 12000)

Two measured facts about small local models shape this file (probe 2026-08-07,
qwen3.5:9b on the real playbook + a real Settings snapshot):

  • Thinking is ON by default and burned ~900 decode tokens per turn — pure
    latency in an agent loop. We send "think": false on every request.
  • With the playbook as system prompt the model sometimes IMITATES the playbook's
    prose tool descriptions instead of using the native tool channel — it prints a
    well-formed JSON call inside a markdown fence. The loop recovers those with a
    conservative fallback parser and COUNTS them (usage['fallback_tool_calls']),
    so a bench run reports how often the native channel was missed instead of
    scoring a formatting miss as a task failure. HUNCH_OLLAMA_STRICT=1 turns the
    fallback off to measure the native channel alone.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import FrozenSet, Tuple

from .base import Backend
from ..errors import HunchError

__all__ = ["OllamaBackend", "OllamaDeadlineExceeded", "OllamaSessionConfig",
           "OllamaTool", "OllamaTransportError"]

DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_NUM_CTX = 32768
DEFAULT_HISTORY_CHARS = 48000
DEFAULT_TOOL_RESULT_CHARS = 12000

TOOL_PROFILES = {
    "ax": {"snapshot", "find", "act", "list_apps", "launch_app", "quit_app"},
    "web": {"web_open", "web_login", "web_snapshot", "web_screenshot", "web_act",
            "web_restart", "web_tabs", "web_switch_tab", "list_credentials",
            "web_fill_login", "web_fill_secret", "clipboard_get", "clipboard_set",
            "notify_user"},
    "native": {"file_op", "trash", "open_file", "reveal_in_finder", "clipboard_get",
               "clipboard_set", "applescript", "list_apps", "launch_app", "quit_app",
               "notify_user"},
}

PROFILE_GUIDANCE = {
    "ax": "Use only the exposed AX tools. Do not use AppleScript, web/CDP, screenshots, "
          "pixel clicks, keyboard shortcuts, or Settings search as a shortcut.",
    "web": "Use only the exposed web/CDP tools. Read with web_snapshot and act with "
           "web_act; do not switch to native AX or AppleScript.",
    "native": "Use the exposed direct OS/API tools. Prefer file_op, trash, open_file, "
              "clipboard, and AppleScript; do not navigate UI when a direct operation exists.",
}

# One fenced or bare JSON object candidate in assistant prose (non-greedy; the
# parser json.loads-validates every candidate, so a loose match is harmless).
_JSON_CANDIDATE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```|(\{[^`]*\})", re.DOTALL)


@dataclass(frozen=True)
class OllamaTool:
    """Deeply immutable tool definition used by private Ollama sessions."""

    name: str
    description: str
    input_schema_json: str

    @classmethod
    def create(cls, name, description, input_schema):
        return cls(name=name, description=description,
                   input_schema_json=json.dumps(input_schema, sort_keys=True,
                                                separators=(",", ":")))

    def function_shape(self):
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": json.loads(self.input_schema_json)}}


@dataclass(frozen=True)
class OllamaSessionConfig:
    """Immutable settings for one logical Ollama conversation.

    Passing this object opts out of all request-time HUNCH_OLLAMA_* environment
    reads. The normal direct backend remains environment-driven for compatibility.
    """

    role: str
    system_prompt: str
    tool_names: FrozenSet[str]
    allow_delegation: bool = False
    internal_tools: Tuple[OllamaTool, ...] = ()
    model: str = DEFAULT_MODEL
    host: str = DEFAULT_HOST
    num_ctx: int = DEFAULT_NUM_CTX
    history_chars: int = DEFAULT_HISTORY_CHARS
    tool_result_chars: int = DEFAULT_TOOL_RESULT_CHARS
    strict: bool = True
    request_timeout: float = 600.0
    transport_attempts: int = 1

    def __post_init__(self):
        object.__setattr__(self, "tool_names", frozenset(self.tool_names))
        object.__setattr__(self, "internal_tools", tuple(self.internal_tools))
        if not self.role.strip():
            raise ValueError("Ollama session role cannot be empty")
        if self.num_ctx <= 0 or self.history_chars <= 0 or self.tool_result_chars <= 0:
            raise ValueError("Ollama context and result limits must be positive")
        if self.request_timeout <= 0:
            raise ValueError("Ollama request timeout must be positive")
        if not isinstance(self.transport_attempts, int) or self.transport_attempts < 1:
            raise ValueError("Ollama transport attempts must be positive")
        internal_names = {tool.name for tool in self.internal_tools}
        if len(internal_names) != len(self.internal_tools):
            raise ValueError("duplicate internal Ollama tool name")
        if internal_names & set(self.tool_names):
            raise ValueError("internal and physical Ollama tool names overlap")
        if "delegate" in internal_names and not self.allow_delegation:
            raise ValueError("delegate requires allow_delegation=True")

    @classmethod
    def from_environment(cls, *, role, system_prompt, tool_names,
                         allow_delegation=False, internal_tools=()):
        return cls(
            role=role, system_prompt=system_prompt, tool_names=frozenset(tool_names),
            allow_delegation=allow_delegation, internal_tools=tuple(internal_tools),
            model=os.environ.get("HUNCH_OLLAMA_MODEL", DEFAULT_MODEL),
            host=os.environ.get("HUNCH_OLLAMA_HOST", DEFAULT_HOST).rstrip("/"),
            num_ctx=int(os.environ.get("HUNCH_OLLAMA_NUM_CTX", DEFAULT_NUM_CTX)),
            history_chars=int(os.environ.get("HUNCH_OLLAMA_HISTORY_CHARS",
                                             DEFAULT_HISTORY_CHARS)),
            tool_result_chars=int(os.environ.get("HUNCH_OLLAMA_TOOL_RESULT_CHARS",
                                                 DEFAULT_TOOL_RESULT_CHARS)),
            strict=os.environ.get("HUNCH_OLLAMA_STRICT") == "1")


class OllamaTransportError(HunchError):
    retryable = True


class OllamaDeadlineExceeded(HunchError):
    pass


class OllamaBackend(Backend):
    """A local Ollama-served model as the agent loop, tools in-process."""

    name = "ollama"
    provider = "ollama"
    extra = ""                      # stdlib-only; nothing extra to install
    default_model = DEFAULT_MODEL

    def __init__(self, hunch, *, auth=None, app_id=None, can_use_tool=None,
                 session_config=None, dispatcher=None):
        super().__init__(hunch, auth=auth, app_id=app_id, can_use_tool=can_use_tool)
        if session_config is not None and not isinstance(session_config, OllamaSessionConfig):
            raise TypeError("session_config must be an OllamaSessionConfig or None")
        if dispatcher is not None and not callable(dispatcher):
            raise TypeError("dispatcher must be callable or None")
        self.session_config = session_config
        self._dispatcher = dispatcher
        self.messages = []          # conversation state; run() continues, reset() clears
        self._abort = False         # between-turns cancel flag (see interrupt())
        self._request_timeout = None
        self._deadline = None

    # ── availability ────────────────────────────────────────────────────────────
    @classmethod
    def deps_installed(cls):
        return True                 # urllib is the whole transport

    @classmethod
    def available(cls, app_id=None):
        """True when the local Ollama server answers. A network probe, kept short so
        `hunch doctor` and 'auto' resolution stay snappy when nothing is running."""
        try:
            with urllib.request.urlopen(cls._host() + "/api/tags", timeout=1.5):
                return True
        except Exception:
            return False

    @staticmethod
    def _host():
        return os.environ.get("HUNCH_OLLAMA_HOST", DEFAULT_HOST).rstrip("/")

    def _session_host(self):
        return (self.session_config.host.rstrip("/") if self.session_config is not None
                else self._host())

    # ── request plumbing ─────────────────────────────────────────────────────────
    @staticmethod
    def _profile():
        value = os.environ.get("HUNCH_OLLAMA_TOOL_PROFILE", "all").strip().lower()
        return value if value in (*TOOL_PROFILES, "all") else "all"

    @classmethod
    def _tools(cls):
        """AGENT_TOOLS in Ollama's (OpenAI-style) function-tool shape — same names and
        schemas, so every HUNCH_PLAYBOOK reference resolves for this model too."""
        from ..agent import AGENT_TOOLS
        allowed = TOOL_PROFILES.get(cls._profile())
        return [{"type": "function",
                 "function": {"name": t["name"], "description": t["description"],
                              "parameters": t["input_schema"]}}
                for t in AGENT_TOOLS if allowed is None or t["name"] in allowed]

    @classmethod
    def _allowed_tools(cls):
        return {t["function"]["name"] for t in cls._tools()}

    def _session_tools(self):
        if self.session_config is None:
            return self._tools()
        from ..agent import AGENT_TOOLS
        requested = set(self.session_config.tool_names)
        by_name = {tool["name"]: tool for tool in AGENT_TOOLS}
        unknown = requested - set(by_name)
        if unknown:
            raise HunchError(f"unknown Ollama session tool(s): {sorted(unknown)}")
        physical = [OllamaTool.create(name, by_name[name]["description"],
                                      by_name[name]["input_schema"]).function_shape()
                    for name in [tool["name"] for tool in AGENT_TOOLS] if name in requested]
        return physical + [tool.function_shape() for tool in self.session_config.internal_tools]

    def _session_allowed_tools(self):
        return {tool["function"]["name"] for tool in self._session_tools()}

    @staticmethod
    def _message_size(message):
        return len(json.dumps(message, ensure_ascii=False, separators=(",", ":")))

    def _context_messages(self, usage):
        """Bound inference replay while retaining self.messages verbatim for audit/training.

        Assistant tool calls and their following tool results are kept as one group so
        compaction never creates an orphan tool message rejected by Ollama.
        """
        if self.session_config is None:
            budget = max(4000, int(os.environ.get("HUNCH_OLLAMA_HISTORY_CHARS",
                                                  DEFAULT_HISTORY_CHARS)))
            result_cap = max(1000, int(os.environ.get("HUNCH_OLLAMA_TOOL_RESULT_CHARS",
                                                      DEFAULT_TOOL_RESULT_CHARS)))
        else:
            budget = max(4000, self.session_config.history_chars)
            result_cap = max(1000, self.session_config.tool_result_chars)
        projected = []
        for message in self.messages:
            item = dict(message)
            if item.get("role") == "tool" and isinstance(item.get("content"), str):
                content = item["content"]
                if len(content) > result_cap:
                    item["content"] = content[:result_cap] + "\n[tool result compacted]"
                    usage["tool_results_truncated"] += 1
            projected.append(item)

        groups = []
        for message in projected:
            if message.get("role") == "tool" and groups:
                groups[-1].append(message)
            else:
                groups.append([message])

        kept, size = [], 0
        for group in reversed(groups):
            group_size = sum(self._message_size(m) for m in group)
            if kept and size + group_size > budget:
                break
            kept.append(group)
            size += group_size
        kept.reverse()
        context = [m for group in kept for m in group]
        pruned = len(projected) - len(context)
        if pruned:
            context.insert(0, {"role": "system", "content":
                               f"Earlier history compacted ({pruned} messages). Continue the "
                               "current task from the retained state; do not repeat completed actions."})
            usage["context_compactions"] += 1
            usage["context_messages_pruned"] += pruned
        usage["max_context_chars"] = max(usage["max_context_chars"], size)
        return context

    def _chat(self, body, _attempts=None):
        """POST one /api/chat turn and return the decoded response — the single live
        transport touch-point (override in tests). Timeout is generous: a cold model
        load plus a long prefill on Apple Silicon can take minutes. Transient HTTP
        errors are retried: qwen3.5's tool-call XML template occasionally fails to
        parse a single generation and Ollama 500s ('element <function> closed by
        </parameter>', seen 2026-08-08) — a per-request fluke that must not kill the
        whole episode."""
        attempts = (_attempts if _attempts is not None else
                    self.session_config.transport_attempts
                    if self.session_config is not None else 3)
        for attempt in range(attempts):
            if self._deadline is not None:
                remaining = self._deadline - time.monotonic()
                if remaining <= 0:
                    raise OllamaDeadlineExceeded("Ollama episode deadline exhausted")
                configured = (self.session_config.request_timeout
                              if self.session_config is not None else 600.0)
                self._request_timeout = max(0.001, min(configured, remaining))
            req = urllib.request.Request(
                self._session_host() + "/api/chat", data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            try:
                timeout = self._request_timeout or (
                    self.session_config.request_timeout if self.session_config else 600)
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = json.loads(e.read()).get("error", "")
                except Exception:
                    pass
                if "not found" in detail.lower():
                    raise HunchError(f"Ollama has no model {body['model']!r} — pull it "
                                     f"with: ollama pull {body['model']}") from e
                if attempt < attempts - 1:
                    continue
                raise OllamaTransportError(f"ollama request failed: {detail or e}") from e
            except urllib.error.URLError as e:
                raise OllamaTransportError(
                    f"no Ollama server at {self._session_host()} — start it with "
                    "`ollama serve` (or install: brew install ollama)") from e
            except TimeoutError as e:
                if self._deadline is not None and time.monotonic() >= self._deadline:
                    raise OllamaDeadlineExceeded("Ollama episode deadline exhausted") from e
                raise OllamaTransportError("Ollama request timed out") from e

    def _body(self, model, system, usage):
        num_ctx = (self.session_config.num_ctx if self.session_config is not None else
                   int(os.environ.get("HUNCH_OLLAMA_NUM_CTX", DEFAULT_NUM_CTX)))
        return {"model": model, "stream": False, "think": False,
                "tools": self._session_tools(),
                "options": {"num_ctx": num_ctx},
                "messages": [{"role": "system", "content": system}]
                            + self._context_messages(usage)}

    # ── tool-call recovery (the playbook-imitation failure mode) ─────────────────
    @staticmethod
    def _fallback_calls(text, known=None):
        """Recover tool calls a model wrote as prose/markdown JSON instead of using the
        native channel. Conservative: a candidate must json-parse to an object naming a
        real AGENT_TOOLS tool with dict arguments. Returns [{'name', 'arguments'}, ...]."""
        if known is None:
            from ..agent import AGENT_TOOLS
            known = {t["name"] for t in AGENT_TOOLS}
        calls = []
        for m in _JSON_CANDIDATE.finditer(text or ""):
            try:
                obj = json.loads(m.group(1) or m.group(2))
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(obj, dict):
                continue
            name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
            args = obj.get("arguments") or obj.get("input") or obj.get("parameters")
            if name in known and isinstance(args, (dict, type(None))):
                calls.append({"name": name, "arguments": args or {}})
        return calls

    @staticmethod
    def _call_args(call):
        """A native tool_call's arguments as a dict (Ollama sends a parsed object, but
        some models/versions return a JSON string)."""
        args = (call.get("function") or {}).get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return None
        return args if isinstance(args, dict) else None

    @staticmethod
    def _tool_error(name, text, emit):
        emit("tool", {"name": name, "input": {}})
        emit("tool_result", text)
        return {"role": "tool", "tool_name": name or "invalid_tool_call", "content": text}

    @staticmethod
    def _fingerprint(name, args, result):
        return json.dumps([name, args, result.get("content", "")], sort_keys=True,
                          ensure_ascii=False, separators=(",", ":"))

    def _run_tool(self, name, args, emit):
        """Dispatch one call and return its role='tool' message. PNG bytes ride in
        Ollama's images channel (base64) with a text placeholder."""
        import base64
        emit("tool", {"name": name, "input": args})
        if self._dispatcher is None:
            from .. import agent as _agent
            value, is_error = _agent._dispatch_core(self._h, name, args)
        else:
            value, is_error = self._dispatcher(name, args)
        if isinstance(value, bytes):
            emit("tool_result", "[image]")
            return {"role": "tool", "tool_name": name, "content": "(screenshot attached)",
                    "images": [base64.b64encode(value).decode()]}
        text = ("error: " if is_error and not str(value).startswith("error") else "") + str(value)
        emit("tool_result", text[:200])
        return {"role": "tool", "tool_name": name, "content": text}

    # ── lifecycle ────────────────────────────────────────────────────────────────
    def interrupt(self):
        self._abort = True          # checked between turns

    def reset(self):
        self.messages = []

    # ── run ──────────────────────────────────────────────────────────────────────
    def run(self, task, model=None, max_turns=40, on_event=None,
            system_suffix="", effort=None, max_tokens=16000, deadline=None):
        from ..playbook import HUNCH_PLAYBOOK
        from ..agent import AGENT_ADDENDUM, AgentResult
        emit = on_event or (lambda kind, data: None)
        if self.session_config is None:
            model = model or os.environ.get("HUNCH_OLLAMA_MODEL", self.default_model)
            system = HUNCH_PLAYBOOK + "\n" + AGENT_ADDENDUM
            profile = self._profile()
            if profile in PROFILE_GUIDANCE:
                system += ("\n\n── Active tool profile: " + profile + " ──\n"
                           + PROFILE_GUIDANCE[profile])
            strict = os.environ.get("HUNCH_OLLAMA_STRICT") == "1"
        else:
            model = model or self.session_config.model
            system = self.session_config.system_prompt
            profile = self.session_config.role
            strict = self.session_config.strict
        if system_suffix:
            system += "\n" + system_suffix
        self.last_system = system
        allowed = self._session_allowed_tools()

        self.messages.append({"role": "user", "content": task})
        usage = {"input_tokens": 0, "output_tokens": 0, "fallback_tool_calls": 0,
                 "continue_nudges": 0, "prose_tool_calls": 0,
                 "malformed_tool_calls": 0, "rejected_tool_calls": 0,
                 "no_progress_warnings": 0, "no_progress_aborts": 0,
                 "context_compactions": 0, "context_messages_pruned": 0,
                 "tool_results_truncated": 0, "max_context_chars": 0,
                 "tool_profile": profile}
        stop_reason, final_text, aborted, turn = "max_turns", "", True, 0
        acted = False                  # any tool call dispatched this run
        attempted = False              # includes calls rejected before dispatch
        repeated = {}
        self._abort = False
        self._deadline = deadline

        for turn in range(1, max_turns + 1):
            now = time.monotonic()
            if self._abort or (deadline is not None and now >= deadline):
                stop_reason = "interrupted" if self._abort else "timeout"
                aborted = True
                emit("error", stop_reason)
                break
            if deadline is not None:
                configured = (self.session_config.request_timeout
                              if self.session_config is not None else 600.0)
                self._request_timeout = max(0.001, min(configured, deadline - now))
            else:
                self._request_timeout = None
            try:
                resp = self._chat(self._body(model, system, usage))
            except OllamaDeadlineExceeded:
                stop_reason, aborted = "timeout", True
                emit("error", "timeout")
                break
            usage["input_tokens"] += resp.get("prompt_eval_count") or 0
            usage["output_tokens"] += resp.get("eval_count") or 0
            msg = resp.get("message") or {}
            text = (msg.get("content") or "").strip()
            if text:
                emit("text", text)
            self.messages.append(msg)             # replay verbatim next turn

            calls = [{"name": (c.get("function") or {}).get("name", ""),
                      "arguments": self._call_args(c), "native": True}
                     for c in (msg.get("tool_calls") or [])]
            prose_calls = self._fallback_calls(text, allowed) if not calls else []
            if not calls and prose_calls and not strict:
                calls = [{**c, "native": False} for c in prose_calls]
                usage["fallback_tool_calls"] += len(calls)
            elif not calls and prose_calls and strict:
                usage["prose_tool_calls"] += len(prose_calls)
                if usage["continue_nudges"] < 2:
                    usage["continue_nudges"] += 1
                    self.messages.append({"role": "user", "content":
                                          "That tool call was written as prose and was not "
                                          "executed. Call it through the native tool channel now."})
                    continue
                stop_reason, aborted = "invalid_tool_channel", True
                emit("error", "aborted: model repeatedly wrote tool calls as prose")
                break
            if calls:
                attempted = True
                # ALL results appended before the next request, mirroring ApiBackend
                for c in calls:
                    name, args = c["name"], c["arguments"]
                    if args is None:
                        usage["malformed_tool_calls"] += 1
                        self.messages.append(self._tool_error(
                            name, "error: malformed tool arguments; send a JSON object", emit))
                        continue
                    if name not in allowed:
                        usage["rejected_tool_calls"] += 1
                        self.messages.append(self._tool_error(
                            name, f"error: tool {name!r} is not available in the {profile} profile", emit))
                        continue
                    acted = True
                    result = self._run_tool(name, args, emit)
                    fp = self._fingerprint(name, args, result)
                    repeated[fp] = repeated.get(fp, 0) + 1
                    if repeated[fp] == 2:
                        usage["no_progress_warnings"] += 1
                        result["content"] += ("\n\nNO_PROGRESS: this exact action returned the exact "
                                              "same state twice. Choose a different action or route.")
                    self.messages.append(result)
                    if repeated[fp] >= 3:
                        usage["no_progress_aborts"] += 1
                        stop_reason, aborted = "no_progress", True
                        emit("error", "aborted: identical action and result repeated three times")
                        break
                if stop_reason == "no_progress":
                    break
                continue

            # Announce-without-act: small local models sometimes narrate the plan and
            # stop before any tool call (also seen when Ollama drops a generation's
            # malformed tool-call XML, leaving only prose). Ending there scores a
            # formatting fluke as a task failure, so push back — bounded and COUNTED
            # (usage['continue_nudges']) to keep the measurement honest.
            if not acted and not attempted and usage["continue_nudges"] < 2:
                usage["continue_nudges"] += 1
                self.messages.append({"role": "user", "content":
                                      "You have not called any tool yet. Do not describe "
                                      "the plan — act now by calling a tool."})
                continue

            stop_reason, final_text, aborted = "end_turn", text, False
            emit("done", final_text)
            break
        else:
            emit("error", f"aborted after max_turns={max_turns}")

        self._deadline = None
        return AgentResult(text=final_text, turns=turn, stop_reason=stop_reason,
                           usage=usage, aborted=aborted)
