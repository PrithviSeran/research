import json

import pytest

from hunch.agent import _DISPATCH
from hunch.orchestration.contracts import (DelegationRequest, SpecialistKind,
                                           SpecialistReceipt, SpecialistStatus,
                                           TrustedEvidence)
from hunch.orchestration.specialists import (SPECIALISTS, canonical_recipe_bytes,
                                              recipe_sha256, validate_registry)


def test_request_is_bounded_typed_and_context_allowlisted():
    request = DelegationRequest.create(
        "web", "submit the form", {"url": "https://example.test"}, ["confirmation visible"],
        allowed_context_keys={"url"})
    assert request.specialist is SpecialistKind.WEB
    assert request.to_dict()["context"]["url"] == "https://example.test"
    with pytest.raises(ValueError, match="not allowed"):
        DelegationRequest.create("web", "x", {"secret": "value"},
                                 allowed_context_keys={"url"})
    with pytest.raises(ValueError, match="exceeds"):
        DelegationRequest.create("web", "x" * 1201)


def test_contracts_are_deeply_immutable():
    request = DelegationRequest.create("ax", "toggle", {"app": "Settings"},
                                       allowed_context_keys={"app"})
    with pytest.raises(TypeError):
        request.context["app"] = "Other"
    with pytest.raises(Exception):
        request.objective = "changed"


def test_verified_receipt_requires_trusted_evidence_and_serializes_stably():
    kwargs = dict(
        delegation_id="d-1", specialist="ax", status="verified_complete",
        summary="completed with trusted checker", evidence=("checker:pass",),
        stop_reason="end_turn", turns=2, tool_calls=1, physical_operations=1,
        usage={"input_tokens": 10}, trace_id="t-1",
        disturbance_delta={"pixel_clicks": 0})
    with pytest.raises(ValueError, match="trusted evidence"):
        SpecialistReceipt.create(**kwargs)
    evidence = TrustedEvidence.create("benchmark_checker", {"passed": True})
    receipt = SpecialistReceipt.create(**kwargs, trusted_evidence=evidence)
    assert receipt.to_json() == json.dumps(receipt.to_dict(), sort_keys=True,
                                           ensure_ascii=False, separators=(",", ":"))
    assert "messages" not in receipt.to_dict() and "tool_results" not in receipt.to_dict()


def test_trusted_evidence_rejects_arbitrary_prose():
    with pytest.raises(TypeError, match="booleans or integers"):
        TrustedEvidence.create("benchmark_checker", {"detail": "secret text"})
    with pytest.raises(TypeError, match="booleans or integers"):
        TrustedEvidence("benchmark_checker", {"detail": "direct bypass"})


def test_registry_tools_are_real_and_delegate_is_private():
    validate_registry()
    assert all(spec.tool_names <= set(_DISPATCH) for spec in SPECIALISTS.values())
    assert "delegate" not in _DISPATCH


def test_recipe_corpus_is_stable_and_complete():
    corpus = canonical_recipe_bytes()
    for kind in SpecialistKind:
        assert f"## {kind.value}\n".encode() in corpus
    assert len(recipe_sha256()) == 64
    assert recipe_sha256() == recipe_sha256()
