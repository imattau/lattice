"""Multi-phase scaling law fitting. Spec §5.1, §9.6.

Fits:
    L(D) = sum_k 1[D in D_k] * (E_k + B_k / D^beta_k)

Phase boundaries are detected via a change-point algorithm on the
derivative dL/dlogD, then coefficients are fit per phase.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class PhaseFit:
    D_min: float
    D_max: float
    E: float
    B: float
    beta: float
    r2: float


def _power_law(D: np.ndarray, E: float, B: float, beta: float) -> np.ndarray:
    return E + B / np.power(D, beta)


def detect_phases(
    D: np.ndarray,
    L: np.ndarray,
    window: int = 5,
    threshold: float = 0.15,
) -> list[int]:
    """Detect phase boundaries via derivative of loss w.r.t. log D.

    Returns indices where the local slope changes sharply.
    """
    logD = np.log(D)
    dLdlogD = np.gradient(L, logD)
    # Smooth
    if len(dLdlogD) >= window:
        kernel = np.ones(window) / window
        smooth = np.convolve(dLdlogD, kernel, mode="same")
    else:
        smooth = dLdlogD
    # Second derivative (curvature)
    curvature = np.abs(np.gradient(smooth, logD))
    if curvature.max() <= 0:
        return []
    boundaries = []
    for i in range(1, len(curvature) - 1):
        if curvature[i] > threshold * curvature.max():
            if not boundaries or i - boundaries[-1] > window:
                boundaries.append(i)
    return boundaries


def fit_multi_phase(
    D: np.ndarray,
    L: np.ndarray,
    boundaries: list[int] | None = None,
) -> list[PhaseFit]:
    """Fit a per-phase power law. Spec §5.1."""
    boundaries = boundaries or detect_phases(D, L)
    edges = [0] + boundaries + [len(D)]
    fits: list[PhaseFit] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi - lo < 3:
            continue
        Dk, Lk = D[lo:hi], L[lo:hi]
        try:
            popt, _ = curve_fit(
                _power_law, Dk, Lk,
                p0=[Lk.min(), Lk.max() - Lk.min(), 1.0],
                bounds=([0, 0, 0], [np.inf, np.inf, 10]),
                maxfev=10000,
            )
            pred = _power_law(Dk, *popt)
            ss_res = np.sum((Lk - pred) ** 2)
            ss_tot = np.sum((Lk - Lk.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            fits.append(PhaseFit(
                D_min=float(Dk.min()), D_max=float(Dk.max()),
                E=float(popt[0]), B=float(popt[1]), beta=float(popt[2]),
                r2=float(r2),
            ))
        except RuntimeError:
            continue
    return fits


def data_efficiency_ratio(
    jepa_fit: PhaseFit, llm_fit: PhaseFit,
) -> float:
    """Spec §9.6 — beta_JEPA / beta_LLM."""
    if llm_fit.beta == 0:
        return float("inf")
    return jepa_fit.beta / llm_fit.beta
