'''Latent Probing Suite. Four components, all empirical.'''
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearProbe(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def linear_probe_accuracy(features, labels, num_classes,
                          epochs=200, lr=1e-3) -> float:
    probe = LinearProbe(features.size(-1), num_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(probe(features), labels)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (probe(features).argmax(-1) == labels).float().mean().item()
    return acc


def uniformity(features: torch.Tensor) -> float:
    x = F.normalize(features, dim=-1)
    sim = x @ x.T
    n = sim.size(0)
    off = sim[~torch.eye(n, dtype=torch.bool, device=sim.device)]
    return float(torch.log(off.abs().mean() + 1e-8))


def spectral_decay(features: torch.Tensor) -> float:
    x = features - features.mean(0, keepdim=True)
    s = torch.linalg.svdvals(x)
    return float(s[0] / (s[-1] + 1e-8))


def effective_rank(features: torch.Tensor, threshold: float = 0.9) -> int:
    x = features - features.mean(0, keepdim=True)
    s = torch.linalg.svdvals(x) ** 2
    cum = torch.cumsum(s, 0) / s.sum()
    return int((cum < threshold).sum().item()) + 1


@dataclass
class LinearProbeSuite:
    feature_dim: int
    tasks: list[str]

    def report(self, features: torch.Tensor, labels: dict) -> dict:
        out: dict = {'geometry': {
            'uniformity': uniformity(features),
            'spectral_decay': spectral_decay(features),
            'effective_rank': effective_rank(features),
        }}
        for task in self.tasks:
            if task in labels:
                out[task] = linear_probe_accuracy(
                    features, labels[task], labels[task].max().item() + 1,
                )
        return out
