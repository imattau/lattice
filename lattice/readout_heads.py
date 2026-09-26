'''Typed probabilistic readout heads. Spec §3.3.

Each head is a thin module over a shared representation. It outputs a
bounded, calibrated distribution over a declared option set.
'''
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from interfaces import DecisionType, ReadoutOutput


class ReadoutHead(nn.Module):
    '''One typed decision head.

    Args:
        hidden_dim: shared representation dimension.
        num_options: bounded output space (<= 255 per spec).
        decision_type: declared type.
        rationale_dim: optional rationale vector dimension (auditing aid only).
    '''

    def __init__(
        self,
        hidden_dim: int,
        num_options: int,
        decision_type: DecisionType,
        rationale_dim: int = 0,
        init_temperature: float = 1.0,
    ):
        super().__init__()
        assert num_options <= 255, 'Spec caps option space at 255'
        self.hidden_dim = hidden_dim
        self.num_options = num_options
        self.decision_type = decision_type
        self.rationale_dim = rationale_dim

        # Logit head
        self.logit_head = nn.Linear(hidden_dim, num_options)
        # Learnable temperature, initialized to init_temperature (log-space)
        self.log_temperature = nn.Parameter(
            torch.tensor(float(torch.log(torch.tensor(init_temperature))))
        )
        # Optional rationale vector — auditing aid, not a causal explanation
        self.rationale_head = (
            nn.Linear(hidden_dim, rationale_dim) if rationale_dim > 0 else None
        )

    def forward(self, pooled_state: torch.Tensor) -> ReadoutOutput:
        '''
        Args:
            pooled_state: [B, hidden_dim]
        Returns:
            ReadoutOutput with calibrated distribution.
        '''
        logits = self.logit_head(pooled_state)          # [B, C]
        temperature = torch.exp(self.log_temperature).clamp(min=1e-3)
        scaled = logits / temperature
        distribution = F.softmax(scaled, dim=-1)

        confidence = distribution.max(dim=-1).values    # [B]
        # OOD proxy: negative max logit. Replace with a fitted estimator later.
        ood_score = -logits.max(dim=-1).values

        rationale = None
        if self.rationale_head is not None:
            rationale = self.rationale_head(pooled_state)

        return ReadoutOutput(
            decision_type=self.decision_type,
            distribution=distribution,
            confidence=confidence,
            temperature=temperature.expand_as(confidence),
            ood_score=ood_score,
            rationale_vector=rationale,
        )


class ReadoutBank(nn.Module):
    '''A bank of readout heads sharing one representation. Spec §3.3.'''

    def __init__(self, hidden_dim: int, specs: dict[DecisionType, int],
                 rationale_dim: int = 0):
        super().__init__()
        self.heads = nn.ModuleDict({
            dt.value: ReadoutHead(hidden_dim, n, dt, rationale_dim)
            for dt, n in specs.items()
        })

    def forward(self, pooled_state: torch.Tensor
                ) -> dict[DecisionType, ReadoutOutput]:
        return {
            DecisionType(k): head(pooled_state)
            for k, head in self.heads.items()
        }
