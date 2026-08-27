"""REFERENCE SOLUTION — statistics. Do not distribute to students.

Statistics for Parts 2, 3 and 5. numpy ONLY — scipy/statsmodels implementations are not accepted.

Every function below is a stub. The resampling scheme is fixed so that the grader's recomputation
matches yours:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))          # B resamples of n item indices, drawn ONCE
    stat = c[idx].mean(axis=1)                     # one statistic per resample (row); NOT a Python loop
    ci = np.percentile(stat, [2.5, 97.5])
Paired CI (same items, e.g. two scorers): the SAME idx is applied to both arrays:
    stat = a[idx].mean(axis=1) - b[idx].mean(axis=1)
Unpaired CI (two DISJOINT item sets, e.g. CS vs CA): draw idx_a and THEN idx_b, in that order, from the
same rng:
    idx_a = rng.integers(0, n_a, size=(B, n_a))
    idx_b = rng.integers(0, n_b, size=(B, n_b))
    stat = a[idx_a].mean(1) - b[idx_b].mean(1)
The point estimate is always the plain (unresampled) mean / difference of means. Test yourself against
reference/stats_toy.json (tests/test_stats.py).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

B_DEFAULT = 2000


def _arr(x) -> np.ndarray:
    a = np.asarray(x)
    return a.astype(np.float64) if a.dtype != bool else a.astype(np.float64)


def bootstrap_ci(correct, B: int = B_DEFAULT, seed: int = 0):
    c = _arr(correct); n = len(c)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    accs = c[idx].mean(axis=1)
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return float(c.mean()), float(lo), float(hi)


def paired_bootstrap_ci(correct_a, correct_b, B: int = B_DEFAULT, seed: int = 0):
    a, b = _arr(correct_a), _arr(correct_b)
    assert len(a) == len(b), "paired arrays must have equal length"
    n = len(a)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def unpaired_bootstrap_ci(correct_a, correct_b, B: int = B_DEFAULT, seed: int = 0):
    a, b = _arr(correct_a), _arr(correct_b)
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(B, len(a)))
    ib = rng.integers(0, len(b), size=(B, len(b)))
    diffs = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def discordant_counts(correct_a, correct_b):
    a, b = np.asarray(correct_a).astype(bool), np.asarray(correct_b).astype(bool)
    return int((a & ~b).sum()), int((~a & b).sum())


def agreement_rate(preds_a: Sequence[Optional[str]], preds_b: Sequence[Optional[str]]) -> float:
    assert len(preds_a) == len(preds_b)
    return float(np.mean([x == y for x, y in zip(preds_a, preds_b)]))


def cohens_kappa(preds_a: Sequence[Optional[str]], preds_b: Sequence[Optional[str]]) -> float:
    assert len(preds_a) == len(preds_b)
    n = len(preds_a)
    cats = sorted({str(x) for x in preds_a} | {str(x) for x in preds_b})
    po = sum(x == y for x, y in zip(preds_a, preds_b)) / n
    pe = sum((sum(str(x) == c for x in preds_a) / n) * (sum(str(y) == c for y in preds_b) / n) for c in cats)
    return float(1.0 if pe == 1.0 else (po - pe) / (1 - pe))


def mdd_95(n_a_only: int, n_b_only: int, n: int) -> float:
    return float(1.96 * np.sqrt(n_a_only + n_b_only) / n)
