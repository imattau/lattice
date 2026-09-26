"""Calibration utilities. Spec §9.1 and §6.5.

Phase 0 uses temperature/Platt scaling. RLCD is a Phase 1+ upgrade.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """Standard ECE for multiclass classification.

    Args:
        probs: [N, C] predicted probabilities.
        labels: [N] true class indices.
        n_bins: number of confidence bins.
    """
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros((), device=probs.device)
    bin_edges = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        prop = in_bin.float().mean()
        if prop.item() > 0:
            acc = accuracies[in_bin].float().mean()
            conf = confidences[in_bin].mean()
            ece += (acc - conf).abs() * prop
    return ece.item()


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 200,
    lr: float = 0.01,
) -> float:
    """Fit a single scalar temperature by NLL minimization.

    Returns the temperature value. Used to calibrate a readout head
    post-hoc without retraining the encoder.
    """
    log_T = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([log_T], max_iter=max_iter, lr=lr)

    def closure():
        opt.zero_grad()
        T = torch.exp(log_T).clamp(min=1e-3)
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return torch.exp(log_T).item()


def rlcd_loss(
    probs: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Calibrated-decision loss (RLCD proxy).

    Combines NLL with a confidence-accuracy alignment term.
    Full RLCD optimizes honest calibration under reward; this is a
    differentiable surrogate for Phase 1.
    """
    nll = F.nll_loss(torch.log(probs.clamp(min=1e-8)), labels)
    conf = probs.max(dim=-1).values
    correct = probs.argmax(dim=-1).eq(labels).float()
    # Penalize confident-wrong and under-confident-correct
    alignment = ((conf - correct) ** 2).mean()
    return nll + beta * alignment
