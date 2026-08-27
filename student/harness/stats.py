"""Statistics for Parts 2, 3 and 5. numpy ONLY — scipy/statsmodels implementations are not accepted.

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


def bootstrap_ci(correct: np.ndarray, B: int = B_DEFAULT, seed: int = 0) -> tuple[float, float, float]:
    """Item-level percentile bootstrap. Returns (accuracy, ci_low, ci_high)."""
    raise NotImplementedError("TODO (Part 2)")


def paired_bootstrap_ci(correct_a: np.ndarray, correct_b: np.ndarray, B: int = B_DEFAULT, seed: int = 0) -> tuple[float, float, float]:
    """Paired bootstrap of mean(a) - mean(b) over the SAME items. Returns (diff, ci_low, ci_high)."""
    raise NotImplementedError("TODO (Part 2)")


def unpaired_bootstrap_ci(correct_a: np.ndarray, correct_b: np.ndarray, B: int = B_DEFAULT, seed: int = 0) -> tuple[float, float, float]:
    """Independent resampling of two DISJOINT item sets (e.g. CS vs CA). Returns (diff, ci_low, ci_high)."""
    raise NotImplementedError("TODO (Part 2)")


def discordant_counts(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[int, int]:
    """(n_a_only_correct, n_b_only_correct) over paired items."""
    raise NotImplementedError("TODO (Part 2)")


def agreement_rate(preds_a: Sequence[Optional[str]], preds_b: Sequence[Optional[str]]) -> float:
    """Fraction of paired items with identical predicted letter (None counts as a letter of its own)."""
    raise NotImplementedError("TODO (Part 2)")


def cohens_kappa(preds_a: Sequence[Optional[str]], preds_b: Sequence[Optional[str]]) -> float:
    """Cohen's kappa between two label sequences; None is its own category."""
    raise NotImplementedError("TODO (Part 1)")


def mdd_95(n_a_only: int, n_b_only: int, n: int) -> float:
    """Minimum detectable paired difference at 95%: 1.96 * sqrt(b + c) / n (Wald approximation)."""
    raise NotImplementedError("TODO (Part 5)")
