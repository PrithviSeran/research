"""Immutable contracts crossing coordinator/specialist boundaries."""
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


MAX_OBJECTIVE_CHARS = 1200
MAX_CONTEXT_ITEMS = 12
MAX_CONTEXT_VALUE_CHARS = 500
MAX_CRITERIA = 8
MAX_CRITERION_CHARS = 500
MAX_SUMMARY_CHARS = 1200
MAX_EVIDENCE_ITEMS = 12
MAX_EVIDENCE_CHARS = 500
_KEY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class SpecialistKind(StrEnum):
    APPLESCRIPT = "applescript"
    NATIVE = "native"
    AX = "ax"
    WEB = "web"


class SpecialistStatus(StrEnum):
    REPORTED_COMPLETE = "reported_complete"
    VERIFIED_COMPLETE = "verified_complete"
    VERIFICATION_FAILED = "verification_failed"
    BLOCKED = "blocked"
    FAILED = "failed"
    TIMEOUT = "timeout"


class TrustedEvidenceSource(StrEnum):
    BENCHMARK_CHECKER = "benchmark_checker"
    RUNTIME_VERIFIER = "runtime_verifier"


@dataclass(frozen=True)
class TrustedEvidence:
    """Allowlisted verifier facts; never arbitrary model or tool-result prose."""

    source: TrustedEvidenceSource
    facts: object

    def __post_init__(self):
        object.__setattr__(self, "source", TrustedEvidenceSource(self.source))
        facts = dict(self.facts)
        if not facts:
            raise ValueError("trusted evidence facts must be a non-empty dict")
        if any(not isinstance(value, (bool, int)) for value in facts.values()):
            raise TypeError("trusted evidence values must be booleans or integers")
        object.__setattr__(self, "facts", _freeze_mapping(
            facts, "trusted_evidence", value_limit=32))

    @classmethod
    def create(cls, source, facts):
        if not isinstance(facts, dict):
            raise TypeError("trusted evidence facts must be a dict")
        return cls(source, facts)

    def labels(self):
        return tuple(f"{self.source.value}:{key}={str(value).lower()}"
                     for key, value in self.facts.items())

    @property
    def passed(self):
        """Only an explicit boolean pass can promote a model report to verified."""
        return self.facts.get("passed") is True


def _bounded_text(value, name, limit, *, empty=False):
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    value = value.strip()
    if not empty and not value:
        raise ValueError(f"{name} cannot be empty")
    if len(value) > limit:
        raise ValueError(f"{name} exceeds {limit} characters")
    return value


def _freeze_mapping(value, name, *, allowed_keys=None, value_limit=MAX_CONTEXT_VALUE_CHARS,
                    max_items=MAX_CONTEXT_ITEMS):
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dict")
    if len(value) > max_items:
        raise ValueError(f"{name} has too many entries")
    frozen = {}
    for key, item in sorted(value.items()):
        if not isinstance(key, str) or not _KEY.fullmatch(key):
            raise ValueError(f"invalid {name} key {key!r}")
        if allowed_keys is not None and key not in allowed_keys:
            raise ValueError(f"{name} key {key!r} is not allowed")
        if not isinstance(item, (str, int, float, bool, type(None))):
            raise TypeError(f"{name}[{key!r}] must be a bounded scalar")
        if isinstance(item, str):
            item = _bounded_text(item, f"{name}[{key!r}]", value_limit, empty=True)
        frozen[key] = item
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class DelegationRequest:
    specialist: SpecialistKind
    objective: str
    context: object
    acceptance_criteria: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "specialist", SpecialistKind(self.specialist))
        object.__setattr__(self, "objective", _bounded_text(
            self.objective, "objective", MAX_OBJECTIVE_CHARS))
        object.__setattr__(self, "context", _freeze_mapping(
            dict(self.context), "context"))
        criteria = tuple(_bounded_text(item, "acceptance criterion",
                                       MAX_CRITERION_CHARS)
                         for item in self.acceptance_criteria)
        if len(criteria) > MAX_CRITERIA:
            raise ValueError("too many acceptance criteria")
        object.__setattr__(self, "acceptance_criteria", criteria)

    @classmethod
    def create(cls, specialist, objective, context=None, acceptance_criteria=(),
               *, allowed_context_keys=None):
        specialist = SpecialistKind(specialist)
        objective = _bounded_text(objective, "objective", MAX_OBJECTIVE_CHARS)
        context = _freeze_mapping(context or {}, "context",
                                  allowed_keys=allowed_context_keys)
        criteria = tuple(_bounded_text(item, "acceptance criterion",
                                       MAX_CRITERION_CHARS)
                         for item in acceptance_criteria)
        if len(criteria) > MAX_CRITERIA:
            raise ValueError("too many acceptance criteria")
        return cls(specialist, objective, context, criteria)

    def to_dict(self):
        return {"specialist": self.specialist.value, "objective": self.objective,
                "context": dict(self.context),
                "acceptance_criteria": list(self.acceptance_criteria)}


@dataclass(frozen=True)
class SpecialistReceipt:
    delegation_id: str
    specialist: SpecialistKind
    status: SpecialistStatus
    summary: str
    evidence: tuple[str, ...]
    stop_reason: str
    turns: int
    tool_calls: int
    physical_operations: int
    usage: object
    trace_id: str
    disturbance_delta: object
    trusted_evidence: object = None

    def __post_init__(self):
        object.__setattr__(self, "delegation_id", _bounded_text(
            self.delegation_id, "delegation_id", 128))
        object.__setattr__(self, "trace_id", _bounded_text(
            self.trace_id, "trace_id", 128))
        object.__setattr__(self, "stop_reason", _bounded_text(
            self.stop_reason, "stop_reason", 128))
        object.__setattr__(self, "specialist", SpecialistKind(self.specialist))
        object.__setattr__(self, "status", SpecialistStatus(self.status))
        if (self.status is SpecialistStatus.VERIFIED_COMPLETE
                and not isinstance(self.trusted_evidence, TrustedEvidence)):
            raise ValueError("VERIFIED_COMPLETE requires trusted evidence")
        if (self.trusted_evidence is not None
                and not isinstance(self.trusted_evidence, TrustedEvidence)):
            raise TypeError("trusted_evidence must be a TrustedEvidence or None")
        object.__setattr__(self, "summary", _bounded_text(
            self.summary, "summary", MAX_SUMMARY_CHARS))
        evidence = tuple(_bounded_text(item, "evidence", MAX_EVIDENCE_CHARS)
                         for item in self.evidence)
        if len(evidence) > MAX_EVIDENCE_ITEMS:
            raise ValueError("too many evidence items")
        object.__setattr__(self, "evidence", evidence)
        for field_name in ("turns", "tool_calls", "physical_operations"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        object.__setattr__(self, "usage", _freeze_mapping(
            dict(self.usage), "usage", value_limit=200, max_items=24))
        object.__setattr__(self, "disturbance_delta", _freeze_mapping(
            dict(self.disturbance_delta), "disturbance_delta", value_limit=100))

    @classmethod
    def create(cls, *, delegation_id, specialist, status, summary, evidence=(),
               stop_reason, turns, tool_calls, physical_operations, usage,
               trace_id, disturbance_delta, trusted_evidence=None):
        status = SpecialistStatus(status)
        if (status is SpecialistStatus.VERIFIED_COMPLETE
                and not isinstance(trusted_evidence, TrustedEvidence)):
            raise ValueError("VERIFIED_COMPLETE requires trusted evidence")
        if trusted_evidence is not None and not isinstance(trusted_evidence, TrustedEvidence):
            raise TypeError("trusted_evidence must be a TrustedEvidence or None")
        evidence = tuple(_bounded_text(item, "evidence", MAX_EVIDENCE_CHARS)
                         for item in evidence)
        if len(evidence) > MAX_EVIDENCE_ITEMS:
            raise ValueError("too many evidence items")
        for field_name, value in (("turns", turns), ("tool_calls", tool_calls),
                                  ("physical_operations", physical_operations)):
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        return cls(
            _bounded_text(delegation_id, "delegation_id", 128),
            SpecialistKind(specialist), status,
            _bounded_text(summary, "summary", MAX_SUMMARY_CHARS), evidence,
            _bounded_text(stop_reason, "stop_reason", 128), turns, tool_calls,
            physical_operations, _freeze_mapping(
                dict(usage), "usage", value_limit=200, max_items=24),
            _bounded_text(trace_id, "trace_id", 128),
            _freeze_mapping(dict(disturbance_delta), "disturbance_delta",
                            value_limit=100), trusted_evidence)

    def to_dict(self):
        return {
            "delegation_id": self.delegation_id,
            "specialist": self.specialist.value,
            "status": self.status.value,
            "summary": self.summary,
            "evidence": list(self.evidence),
            "stop_reason": self.stop_reason,
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "physical_operations": self.physical_operations,
            "usage": dict(self.usage),
            "trace_id": self.trace_id,
            "disturbance_delta": dict(self.disturbance_delta),
            "trusted_evidence": self.trusted_evidence is not None,
        }

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"))
