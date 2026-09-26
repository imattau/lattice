"""Training objectives for the JEPA Core. Spec §6.

Primary:   JEPA latent prediction (cosine, EMA target, stop-grad).
Auxiliary: NextLat, MLM, STP regularizer.
Optional:  Functorial consistency loss (see §6.4; experimental).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPAPredictor(nn.Module):
    """Lightweight predictor mapping context latent → target latent. Spec §3.1."""

    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or 2 * dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim),
        )

    def forward(self, context_latent: torch.Tensor) -> torch.Tensor:
        return self.net(context_latent)


def jepa_loss(
    predicted: torch.Tensor,      # [B, T_tgt, D]
    target: torch.Tensor,         # [B, T_tgt, D], stop-grad applied
) -> torch.Tensor:
    """Cosine similarity loss. Spec §6.1."""
    pred = F.normalize(predicted, dim=-1)
    tgt = F.normalize(target, dim=-1).detach()
    return -(pred * tgt).sum(dim=-1).mean()


def next_latent_loss(
    z_t: torch.Tensor,            # [B, T-1, D]
    z_next_pred: torch.Tensor,    # [B, T-1, D]
) -> torch.Tensor:
    """Next-latent prediction loss. Spec §6.2."""
    return F.mse_loss(z_next_pred, z_t.detach())


def mlm_loss(
    logits: torch.Tensor,         # [B, T, V]
    labels: torch.Tensor,         # [B, T], -100 for ignored
) -> torch.Tensor:
    """Light token reconstruction. Spec §6.3."""
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100,
    )


def stp_regularizer(
    hidden_trajectory: torch.Tensor,  # [B, T, D]
) -> torch.Tensor:
    """Semantic Tube Prediction regularizer. Spec §5.6.

    Encourages hidden states to lie near the geodesic (linear
    interpolation) between endpoints on the semantic manifold.
    """
    B, T, D = hidden_trajectory.shape
    if T < 3:
        return torch.zeros((), device=hidden_trajectory.device)
    start = hidden_trajectory[:, 0]
    end = hidden_trajectory[:, -1]
    t = torch.linspace(0, 1, T, device=hidden_trajectory.device)
    t = t.view(1, T, 1)
    geodesic = start.unsqueeze(1) * (1 - t) + end.unsqueeze(1) * t
    return ((hidden_trajectory - geodesic) ** 2).mean()


def functorial_consistency_loss(
    z_a: torch.Tensor,           # [B, D]
    z_b: torch.Tensor,           # [B, D]
    z_ab: torch.Tensor,          # [B, D] representation of composition
    compose_fn: nn.Module,       # monoidal composition on latents
) -> torch.Tensor:
    """Functorial consistency. Spec §5.4, §6.4.

    Enforces z(a ∘ b) ≈ compose_fn(z(a), z(b)) up to a learned
    natural transformation. This is the experimental functoriality
    constraint; results are reported, not assumed.
    """
    composed = compose_fn(z_a, z_b)
    return F.mse_loss(z_ab, composed)


def combined_loss(
    *,
    jepa: torch.Tensor,
    next_lat: torch.Tensor | None = None,
    mlm: torch.Tensor | None = None,
    stp: torch.Tensor | None = None,
    functorial: torch.Tensor | None = None,
    alpha: float = 0.1,
    beta: float = 0.1,
    gamma: float = 0.01,
    delta: float = 0.01,
) -> torch.Tensor:
    """Spec §6.5. All auxiliary terms optional."""
    loss = jepa
    if next_lat is not None:
        loss = loss + alpha * next_lat
    if mlm is not None:
        loss = loss + beta * mlm
    if stp is not None:
        loss = loss + gamma * stp
    if functorial is not None:
        loss = loss + delta * functorial
    return loss
