'''Combined JEPA training objective with per-term switches.

Every auxiliary term is gated by a config flag so that the ablation
matrix in the verified training plan is a matter of flipping booleans,
not editing code.
'''
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from lattice.config import LossWeights


def jepa_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    '''Cosine loss between predicted and stop-gradient target latents.

    Args:
        predicted: [N, D] latents from the predictor.
        target:    [N, D] latents from the EMA target encoder.
    '''
    pred = F.normalize(predicted, dim=-1)
    tgt = F.normalize(target, dim=-1).detach()
    return -(pred * tgt).sum(dim=-1).mean()


def next_latent_loss(z: torch.Tensor) -> torch.Tensor:
    '''Predict z[:, t+1] from z[:, t]. Cheap recurrent inductive bias.'''
    if z.size(1) < 2:
        return torch.zeros((), device=z.device)
    return F.mse_loss(z[:, :-1], z[:, 1:].detach())


def mlm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )


def stp_regularizer(hidden: torch.Tensor) -> torch.Tensor:
    '''Simplified Semantic Tube proxy: penalize deviation from the
    linear interpolation between sequence endpoints. The real STP uses
    a geodesic on the semantic manifold; this is a differentiable proxy
    suitable for reference implementation and per-task validation.
    '''
    B, T, D = hidden.shape
    if T < 3:
        return torch.zeros((), device=hidden.device)
    t = torch.linspace(0, 1, T, device=hidden.device).view(1, T, 1)
    start = hidden[:, 0].unsqueeze(1)
    end = hidden[:, -1].unsqueeze(1)
    geodesic = start * (1 - t) + end * t
    return ((hidden - geodesic) ** 2).mean()


class MonoidalCompose(nn.Module):
    '''Placeholder monoidal composition operator on latents.

    NOTE: The spec's functoriality requirement has no established
    implementation for transformers on language. This is a learned
    bilinear form used only for the functoriality ablation. It is not
    claimed to be a true monoidal functor.
    '''
    def __init__(self, dim: int):
        super().__init__()
        self.bilinear = nn.Bilinear(dim, dim, dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.bilinear(a, b)


def functorial_consistency_loss(z_a, z_b, z_ab, compose_fn) -> torch.Tensor:
    '''Soft constraint: z(a ∘ b) ≈ compose(z(a), z(b)).'''
    return F.mse_loss(z_ab, compose_fn(z_a, z_b))


@dataclass
class ObjectiveOutputs:
    total: torch.Tensor
    jepa: torch.Tensor
    nextlat: torch.Tensor | None = None
    mlm: torch.Tensor | None = None
    stp: torch.Tensor | None = None
    functorial: torch.Tensor | None = None
    aux: dict | None = None


def combined_objective(
    *,
    jepa: torch.Tensor,
    weights: LossWeights,
    nextlat: torch.Tensor | None = None,
    mlm: torch.Tensor | None = None,
    stp: torch.Tensor | None = None,
    functorial: torch.Tensor | None = None,
) -> ObjectiveOutputs:
    total = jepa
    if nextlat is not None:
        total = total + weights.alpha_nextlat * nextlat
    if mlm is not None:
        total = total + weights.beta_mlm * mlm
    if stp is not None:
        total = total + weights.gamma_stp * stp
    if functorial is not None:
        total = total + weights.delta_functorial * functorial
    return ObjectiveOutputs(
        total=total, jepa=jepa, nextlat=nextlat, mlm=mlm,
        stp=stp, functorial=functorial,
    )
