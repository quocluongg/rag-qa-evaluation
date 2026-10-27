"""Statistical helpers for paired RAG evaluation.

The functions here are deliberately dependency-light (numpy + scipy) and are
covered by unit tests so the thesis numbers are reproducible.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
from scipy import stats


def bootstrap_ci(
    values: Iterable[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean, lower, upper) percentile bootstrap CI."""
    arr = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(arr.mean())
    if arr.size == 1:
        return (mean, mean, mean)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (mean, float(lo), float(hi))


def mcnemar_test(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict[str, float]:
    """Exact / continuity-corrected McNemar test for paired binary outcomes.

    b = A correct & B wrong; c = A wrong & B correct.
    """
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same length")
    b_count = int(np.sum(a & ~b))
    c_count = int(np.sum(~a & b))
    n = b_count + c_count
    if n == 0:
        p_value = 1.0
        statistic = 0.0
    elif n < 25:
        statistic = float(min(b_count, c_count))
        p_value = min(1.0, 2.0 * float(stats.binom.cdf(min(b_count, c_count), n, 0.5)))
    else:
        statistic = float((abs(b_count - c_count) - 1) ** 2 / n)
        p_value = float(stats.chi2.sf(statistic, df=1))
    return {
        "b": float(b_count),
        "c": float(c_count),
        "statistic": statistic,
        "p_value": float(p_value),
    }


def holm_correction(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, preserving the input order."""
    pvals = np.asarray([float(p) if p is not None else float("nan") for p in p_values])
    adjusted = np.full(pvals.shape, np.nan, dtype=float)
    valid_idx = np.where(~np.isnan(pvals))[0]
    if valid_idx.size == 0:
        return adjusted.tolist()
    order = valid_idx[np.argsort(pvals[valid_idx])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, pvals[idx] * (m - rank))
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sided Wilcoxon signed-rank p-value with a safe zero-difference fallback."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError("paired arrays must have the same length")
    diff = a_arr - b_arr
    if np.allclose(diff, 0):
        return 1.0
    try:
        return float(stats.wilcoxon(a_arr, b_arr, zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0
