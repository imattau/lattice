"""Typed interfaces between Lattice components. Spec §4."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import torch


class DecisionType(str, Enum):
    ROUTING = "routing"
    URGENCY = "urgency"
    ESCALATION = "escalation"
    RELEVANCE = "relevance"


class Verdict(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"


class Stakes(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class EncoderOutput:
    """Spec §4.1 — Encoder → Reasoner."""
    latent_state: torch.Tensor          # [B, hidden_dim]
    attention_mask: torch.Tensor        # [B, seq_len] bool
    metadata: dict = field(default_factory=dict)


@dataclass
class ReadoutOutput:
    """Spec §4.3 — Readout Head → Controller."""
    decision_type: DecisionType
    distribution: torch.Tensor          # [B, num_options], sums to 1
    confidence: torch.Tensor            # [B], in [0, 1]
    temperature: torch.Tensor           # [B]
    ood_score: torch.Tensor             # [B]
    rationale_vector: Optional[torch.Tensor] = None


@dataclass
class ProposedAction:
    """Spec §4.4 — Controller → Constraint Layer."""
    action_type: str
    parameters: dict
    decision_type: DecisionType
    distribution: torch.Tensor
    confidence: torch.Tensor
    stakes: Stakes


@dataclass
class ConstraintResult:
    """Spec §4.5 — Constraint Layer → Controller."""
    verdict: Verdict
    modified_action: Optional[ProposedAction] = None
    trace: list = field(default_factory=list)
    policy_hash: str = ""


@dataclass
class GenerationRequest:
    """Spec §4.6 — Controller → Talker."""
    task: str                           # draft_response | summarize | explain
    latent_context: torch.Tensor        # [B, seq_len, hidden_dim]
    style: dict = field(default_factory=dict)
    constraint_trace: list = field(default_factory=list)
