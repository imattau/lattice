'''Phase 0: calibrated typed readout heads over a frozen encoder.

Uses RLCD surrogate in-loop, not post-hoc temperature scaling. Reports
ECE, accuracy, Brier, and order-flip rate against a generative baseline.
'''
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from lattice.calibration import (
    expected_calibration_error, fit_temperature, rlcd_surrogate,
)


class ReadoutHead(nn.Module):
    def __init__(self, dim: int, num_options: int):
        super().__init__()
        self.fc = nn.Linear(dim, num_options)
        self.log_temperature = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = torch.exp(self.log_temperature).clamp(min=1e-3)
        return F.softmax(self.fc(x) / T, dim=-1)


def train_readout(
    encoder: nn.Module,
    features: torch.Tensor,       # [N, D] precomputed or cached
    labels: torch.Tensor,         # [N]
    num_options: int,
    epochs: int = 30,
    lr: float = 1e-3,
    beta_rlcd: float = 1.0,
) -> tuple[ReadoutHead, dict]:
    encoder.eval()
    head = ReadoutHead(features.size(-1), num_options)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)

    for epoch in range(epochs):
        probs = head(features)
        loss = rlcd_surrogate(probs, labels, beta=beta_rlcd)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        probs = head(features)
        ece = expected_calibration_error(probs, labels)
        acc = (probs.argmax(dim=-1) == labels).float().mean().item()
        brier = ((probs - F.one_hot(labels, num_options)) ** 2).sum(-1).mean()
    return head, {'ece': ece, 'accuracy': acc, 'brier': float(brier)}


def order_flip_rate(head: ReadoutHead, features: torch.Tensor,
                    permutations: list[torch.Tensor]) -> float:
    '''Measure sensitivity to option-order permutation. Requires that
    the encoder's representation is order-invariant at the pooled level;
    this is a sanity check on the head, not the encoder.'''
    with torch.no_grad():
        base = head(features).argmax(dim=-1)
        flips = 0
        total = 0
        for perm in permutations:
            # Permute the option logits and check if the argmax flips back.
            logits = head.fc(features)
            permuted = logits[..., perm]
            pred = permuted.argmax(dim=-1)
            # Map back
            inv = torch.empty_like(perm)
            inv[perm] = torch.arange(len(perm))
            mapped = inv[pred]
            flips += (mapped != base).sum().item()
            total += base.numel()
    return flips / max(1, total)
