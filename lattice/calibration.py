'''Calibration utilities. RLCD surrogate is first-class from Phase 0.

The verified training plan made calibration a training objective, not a
post-hoc fix. The temperature scaling path remains for post-hoc
correction; the RLCD surrogate is used during readout-head training.
'''
from __future__ import annotations
import torch
import torch.nn.functional as F


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor,
                               n_bins: int = 15) -> float:
    conf, pred = probs.max(dim=-1)
    acc = pred.eq(labels)
    ece = torch.zeros((), device=probs.device)
    edges = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        prop = in_bin.float().mean()
        if prop.item() > 0:
            ece += (acc[in_bin].float().mean() - conf[in_bin].mean()).abs() * prop
    return ece.item()


def rlcd_surrogate(probs: torch.Tensor, labels: torch.Tensor,
                   beta: float = 1.0) -> torch.Tensor:
    '''Differentiable surrogate for the RLCD objective.

    Combines NLL with a confidence-accuracy alignment penalty. This is
    what is used in-loop. Full RLCD requires a reward signal and is not
    implemented here.
    '''
    nll = F.nll_loss(torch.log(probs.clamp(min=1e-8)), labels)
    conf = probs.max(dim=-1).values
    correct = probs.argmax(dim=-1).eq(labels).float()
    alignment = ((conf - correct) ** 2).mean()
    return nll + beta * alignment


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor,
                    max_iter: int = 200, lr: float = 0.01) -> float:
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


# Backwards-compatible alias used by component reference code.
rlcd_loss = rlcd_surrogate
