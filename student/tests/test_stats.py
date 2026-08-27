"""Tests for harness.stats against hand-computable cases and reference/stats_toy.json."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from harness import stats
from tests.conftest import reference_path


def test_bootstrap_ci_all_ones():
    assert stats.bootstrap_ci(np.ones(30, dtype=int)) == (1.0, 1.0, 1.0)


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(1)
    c = rng.integers(0, 2, size=200)
    acc, lo, hi = stats.bootstrap_ci(c)
    assert acc == pytest.approx(c.mean())
    assert lo <= acc <= hi and 0.0 <= lo and hi <= 1.0
    # deterministic in (B, seed)
    assert stats.bootstrap_ci(c) == (acc, lo, hi)


def test_discordant_counts():
    a = np.array([1, 1, 0, 0, 1, 0, 1, 1])
    b = np.array([1, 0, 1, 0, 0, 0, 1, 0])
    assert stats.discordant_counts(a, b) == (3, 1)
    assert stats.discordant_counts(b, a) == (1, 3)


def test_paired_self_is_zero():
    c = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])
    assert stats.paired_bootstrap_ci(c, c) == (0.0, 0.0, 0.0)


def test_mdd_95_formula():
    assert stats.mdd_95(10, 10, 400) == pytest.approx(1.96 * math.sqrt(20) / 400, abs=1e-12)


def test_agreement_and_kappa_simple():
    a = ["A", "B", "C", "D", None, "A"]
    assert stats.agreement_rate(a, list(a)) == 1.0
    assert stats.cohens_kappa(a, list(a)) == pytest.approx(1.0)
    b = ["A", "B", "C", "D", "A", None]
    assert stats.agreement_rate(a, b) == pytest.approx(4 / 6)
    # kappa for a 2-category, hand-computed case: po = 0.5, pe = 0.5 -> 0
    x = ["A", "A", "B", "B"]
    y = ["A", "B", "A", "B"]
    assert stats.cohens_kappa(x, y) == pytest.approx(0.0)


def _load_toy():
    with open(reference_path("stats_toy.json"), encoding="utf-8") as f:
        return json.load(f)


def test_reference_bootstrap_ci():
    d = _load_toy()
    for key in ("a", "b"):
        got = stats.bootstrap_ci(np.array(d[f"correct_{key}"]), B=2000, seed=0)
        exp = d["expected"][f"bootstrap_ci_{key}"]
        assert got[0] == pytest.approx(exp[0], abs=1e-9)
        assert got[1:] == pytest.approx(exp[1:], abs=1e-6)


def test_reference_paired_and_unpaired():
    d = _load_toy()
    a, b = np.array(d["correct_a"]), np.array(d["correct_b"])
    got = stats.paired_bootstrap_ci(a, b, B=2000, seed=0)
    exp = d["expected"]["paired_bootstrap_ci"]
    assert got[0] == pytest.approx(exp[0], abs=1e-9)
    assert got[1:] == pytest.approx(exp[1:], abs=1e-6)
    got = stats.unpaired_bootstrap_ci(a, b, B=2000, seed=0)
    exp = d["expected"]["unpaired_bootstrap_ci"]
    assert got[0] == pytest.approx(exp[0], abs=1e-9)
    assert got[1:] == pytest.approx(exp[1:], abs=1e-6)


def test_reference_counts_and_agreement():
    d = _load_toy()
    a, b = np.array(d["correct_a"]), np.array(d["correct_b"])
    assert list(stats.discordant_counts(a, b)) == list(d["expected"]["discordant_counts"])
    assert stats.agreement_rate(d["preds_a"], d["preds_b"]) == pytest.approx(d["expected"]["agreement_rate"], abs=1e-9)
    assert stats.cohens_kappa(d["preds_a"], d["preds_b"]) == pytest.approx(d["expected"]["cohens_kappa"], abs=1e-9)
    bc = d["expected"]["discordant_counts"]
    assert stats.mdd_95(bc[0], bc[1], len(a)) == pytest.approx(d["expected"]["mdd_95"], abs=1e-9)
