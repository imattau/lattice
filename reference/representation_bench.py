"""Representation benchmark. Spec §9.5.

Four components: linear probing, geometry, compositional
generalization, cross-modal transfer.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Component 1: Linear probing -----------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def linear_probe_accuracy(
    features: torch.Tensor,       # [N, D]
    labels: torch.Tensor,         # [N]
    num_classes: int,
    epochs: int = 200,
    lr: float = 1e-3,
) -> float:
    """Freeze features, train a linear classifier. Spec §9.5."""
    probe = LinearProbe(features.size(-1), num_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        logits = probe(features)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (probe(features).argmax(dim=-1) == labels).float().mean()
    return acc.item()


# ---- Component 2: Geometry -----------------------------------------------

def uniformity(features: torch.Tensor) -> float:
    """Log of average pairwise cosine similarity. Lower is better."""
    x = F.normalize(features, dim=-1)
    sim = x @ x.T
    n = sim.size(0)
    off_diag = sim[~torch.eye(n, dtype=torch.bool, device=sim.device)]
    return torch.log(off_diag.abs().mean() + 1e-8).item()


def alignment(features_a: torch.Tensor, features_b: torch.Tensor) -> float:
    """Cosine similarity between semantic equivalents. Higher is better."""
    a = F.normalize(features_a, dim=-1)
    b = F.normalize(features_b, dim=-1)
    return (a * b).sum(dim=-1).mean().item()


def spectral_decay(features: torch.Tensor) -> float:
    """Ratio of first to last singular value. Lower decay is richer."""
    x = features - features.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x)
    return (s[0] / (s[-1] + 1e-8)).item()


def effective_rank(features: torch.Tensor, threshold: float = 0.9) -> int:
    """Number of dimensions capturing `threshold` of variance."""
    x = features - features.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x) ** 2
    cum = torch.cumsum(s, dim=0) / s.sum()
    return int((cum < threshold).sum().item()) + 1


def geometry_report(features: torch.Tensor) -> dict:
    return {
        "uniformity": uniformity(features),
        "spectral_decay": spectral_decay(features),
        "effective_rank": effective_rank(features),
    }
