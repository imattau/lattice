'''Exploratory scaling analysis. Does NOT assume multi-phase scaling.

Fits both a single-phase power law and a multi-phase piecewise power
law, then chooses between them with BIC. The verification flagged that
the multi-phase result is for two-layer networks; the data must decide.
'''
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


def single_phase(D: np.ndarray, E: float, B: float, beta: float) -> np.ndarray:
    return E + B / np.power(D, beta)


def multi_phase(D: np.ndarray, L: np.ndarray, boundaries: list[int]
                ) -> tuple[list[tuple[float, float, float, float]], float]:
    '''Returns per-phase (E, B, beta, r2) and total residual sum of squares.'''
    edges = [0] + list(boundaries) + [len(D)]
    fits = []
    rss = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi - lo < 3:
            continue
        Dk, Lk = D[lo:hi], L[lo:hi]
        try:
            popt, _ = curve_fit(
                single_phase, Dk, Lk,
                p0=[Lk.min(), max(Lk.max() - Lk.min(), 1e-6), 1.0],
                bounds=([0, 0, 0], [np.inf, np.inf, 10]),
                maxfev=20000,
            )
        except RuntimeError:
            continue
        pred = single_phase(Dk, *popt)
        resid = float(np.sum((Lk - pred) ** 2))
        ss_tot = float(np.sum((Lk - Lk.mean()) ** 2))
        r2 = 1 - resid / ss_tot if ss_tot > 0 else 0.0
        fits.append((float(popt[0]), float(popt[1]), float(popt[2]), r2))
        rss += resid
    return fits, rss


def bic(rss: float, n: int, k: int) -> float:
    return n * np.log(max(rss / n, 1e-12)) + k * np.log(n)


def detect_phase_boundaries(D: np.ndarray, L: np.ndarray,
                            window: int = 5, threshold: float = 0.15
                            ) -> list[int]:
    '''Derivative-of-loss diagnostics. Returns candidate change points.'''
    logD = np.log(D)
    dL = np.gradient(L, logD)
    if len(dL) >= window:
        kernel = np.ones(window) / window
        smooth = np.convolve(dL, kernel, mode='same')
    else:
        smooth = dL
    curvature = np.abs(np.gradient(smooth, logD))
    if curvature.max() <= 0:
        return []
    picks = []
    for i in range(1, len(curvature) - 1):
        if curvature[i] > threshold * curvature.max():
            if not picks or i - picks[-1] > window:
                picks.append(i)
    return picks


@dataclass
class ScalingChoice:
    model: str                    # 'single_phase' | 'multi_phase' | 'none'
    single_bic: float | None
    multi_bic: float | None
    single_fit: tuple | None
    multi_fits: list | None
    boundaries: list[int]


def choose_scaling_model(D: np.ndarray, L: np.ndarray) -> ScalingChoice:
    '''Fit both models; return the one with lower BIC.'''
    n = len(D)
    # Single-phase fit
    try:
        popt, _ = curve_fit(
            single_phase, D, L,
            p0=[L.min(), max(L.max() - L.min(), 1e-6), 1.0],
            bounds=([0, 0, 0], [np.inf, np.inf, 10]),
            maxfev=20000,
        )
        pred = single_phase(D, *popt)
        single_rss = float(np.sum((L - pred) ** 2))
        single_b = bic(single_rss, n, k=3)
    except RuntimeError:
        single_b = None
        popt = None

    boundaries = detect_phase_boundaries(D, L)
    if boundaries:
        multi_fits, multi_rss = multi_phase(D, L, boundaries)
        k = 3 * len(multi_fits)
        multi_b = bic(multi_rss, n, k) if multi_fits else None
    else:
        multi_fits, multi_b = [], None

    single_fit = tuple(popt) if popt is not None else None
    if single_b is None and multi_b is None:
        return ScalingChoice('none', None, None, None, None, boundaries)
    if multi_b is None or (single_b is not None and single_b <= multi_b):
        return ScalingChoice('single_phase', single_b, multi_b,
                             single_fit, multi_fits, boundaries)
    return ScalingChoice('multi_phase', single_b, multi_b,
                         single_fit, multi_fits, boundaries)
