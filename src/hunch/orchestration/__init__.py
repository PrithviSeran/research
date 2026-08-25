"""Private local-model orchestration experiment.

Nothing in this package is registered as an MCP tool or exposed by ``Hunch.agent``.
"""

ORCHESTRATOR_VERSION = "experiment-v1"

from .contracts import (DelegationRequest, SpecialistKind, SpecialistReceipt,
                        SpecialistStatus, TrustedEvidence, TrustedEvidenceSource)
from .specialists import SPECIALISTS, SpecialistSpec

__all__ = ["DelegationRequest", "SPECIALISTS", "SpecialistKind",
           "SpecialistReceipt", "SpecialistSpec", "SpecialistStatus"]
__all__ += ["TrustedEvidence", "TrustedEvidenceSource"]
__all__ += ["ORCHESTRATOR_VERSION"]
