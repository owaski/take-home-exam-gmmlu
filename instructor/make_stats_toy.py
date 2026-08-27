"""Generate student/reference/stats_toy.json from the reference stats implementation (numpy only, deterministic)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "solution"))
from harness import stats as S  # noqa: E402  (instructor/solution/harness/stats.py)

rng = np.random.default_rng(12345)
n = 30
a = (rng.random(n) < 0.6).astype(int)
b = (rng.random(n) < 0.45).astype(int)
letters = ["A", "B", "C", "D", None]
pa = [letters[i] for i in rng.integers(0, 4, n)]
pb = [pa[i] if rng.random() < 0.5 else letters[j] for i, j in zip(range(n), rng.integers(0, 5, n))]
bc = S.discordant_counts(a, b)
out = {
    "correct_a": a.tolist(), "correct_b": b.tolist(), "preds_a": pa, "preds_b": pb,
    "B": 2000, "seed": 0,
    "expected": {
        "bootstrap_ci_a": list(S.bootstrap_ci(a)), "bootstrap_ci_b": list(S.bootstrap_ci(b)),
        "paired_bootstrap_ci": list(S.paired_bootstrap_ci(a, b)),
        "unpaired_bootstrap_ci": list(S.unpaired_bootstrap_ci(a, b)),
        "discordant_counts": list(bc), "agreement_rate": S.agreement_rate(pa, pb),
        "cohens_kappa": S.cohens_kappa(pa, pb), "mdd_95": S.mdd_95(bc[0], bc[1], n),
    },
}
path = os.path.join(os.path.dirname(__file__), "..", "student", "reference", "stats_toy.json")
json.dump(out, open(path, "w"), indent=1)
print("wrote", path, out["expected"])
