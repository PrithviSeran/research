"""Execution lease, permissions, deadlines, and physical-operation accounting."""
import asyncio
import inspect
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from ..agent import _dispatch_core


class RetryableTransportError(Exception):
    """A transport failed before the specialist received or acted on a response."""


class ExecutionLease:
    """Serialize whole specialist episodes, not individual tool calls."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    @contextmanager
    def hold(self):
        self._lock.acquire()
        with self._state_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            yield
        finally:
            with self._state_lock:
                self.active -= 1
            self._lock.release()


@dataclass
class PhysicalOperationBudget:
    limit: int
    used: int = 0
    _lock: object = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        if not isinstance(self.limit, int) or self.limit < 0:
            raise ValueError("physical-operation limit must be a non-negative integer")

    @staticmethod
    def cost(name, args):
        args = args or {}
        if name in {"act", "web_act"}:
            return max(1, len(args.get("actions") or []))
        if name == "file_op" and args.get("batch") is not None:
            return max(1, len(args.get("batch") or []))
        if name in {"trash", "reveal_in_finder"}:
            return max(1, len(args.get("paths") or []))
        return 1

    def available(self, cost):
        with self._lock:
            return self.used + cost <= self.limit

    def reserve(self, cost):
        with self._lock:
            if self.used + cost > self.limit:
                return False
            self.used += cost
            return True


def _run_awaitable(awaitable):
    """Resolve the host SDK's async permission result from this synchronous loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    result, failure = [], []

    def run():
        try:
            result.append(asyncio.run(awaitable))
        except BaseException as exc:  # surfaced on the caller thread
            failure.append(exc)

    thread = threading.Thread(target=run, name="hunch-ollama-permission", daemon=True)
    thread.start()
    thread.join()
    if failure:
        raise failure[0]
    return result[0]


def permission_allowed(callback, name, args):
    if callback is None:
        return True
    result = callback(name, args, None)
    if inspect.isawaitable(result):
        result = _run_awaitable(result)
    if isinstance(result, bool):
        return result
    if isinstance(result, dict):
        return result.get("behavior") == "allow" or result.get("allow") is True
    class_name = type(result).__name__.lower()
    if class_name.endswith("permissionresultallow"):
        return True
    behavior = getattr(result, "behavior", None)
    return behavior == "allow"


class ScopedDispatcher:
    """The only path from a specialist tool call to physical Hunch dispatch."""

    def __init__(self, hunch, allowed_tools, budget, *, deadline=None,
                 can_use_tool=None, on_dispatch=None, interrupted=None):
        self.hunch = hunch
        self.allowed_tools = frozenset(allowed_tools)
        self.budget = budget
        self.deadline = deadline
        self.can_use_tool = can_use_tool
        self.on_dispatch = on_dispatch or (lambda event: None)
        self.interrupted = interrupted or (lambda: False)

    def __call__(self, name, args):
        if name not in self.allowed_tools:
            return f"error: tool {name!r} is outside this specialist's immutable scope", True
        if self.interrupted():
            return "REFUSED: specialist interrupted before dispatch", False
        if self.deadline is not None and time.monotonic() >= self.deadline:
            return "REFUSED: specialist deadline exhausted before dispatch", False
        cost = self.budget.cost(name, args)
        if not self.budget.available(cost):
            return (f"REFUSED: physical-operation budget exhausted "
                    f"({self.budget.used}/{self.budget.limit})"), False
        try:
            allowed = permission_allowed(self.can_use_tool, name, args)
        except Exception as exc:
            self.on_dispatch({"tool": name, "cost": 0, "permission": "error"})
            return f"REFUSED: host permission check failed: {exc}", False
        if not allowed:
            self.on_dispatch({"tool": name, "cost": 0, "permission": "denied"})
            return "REFUSED: host denied this tool call; do not retry it", False
        if self.interrupted():
            return "REFUSED: specialist interrupted during permission check", False
        if self.deadline is not None and time.monotonic() >= self.deadline:
            return "REFUSED: specialist deadline expired during permission check", False
        if not self.budget.reserve(cost):
            return "REFUSED: physical-operation budget was concurrently exhausted", False
        value, is_error = _dispatch_core(self.hunch, name, args)
        self.on_dispatch({"tool": name, "cost": cost, "permission": "allowed",
                          "result_class": "error" if is_error else "content"})
        return value, is_error
