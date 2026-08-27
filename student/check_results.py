#!/usr/bin/env python
"""check_results.py -- recompute every accuracy / count in the submitted CSVs from predictions/*.jsonl.

This is the SAME script the grader runs. Run it from the student/ directory:

    python check_results.py [--predictions predictions] [--results results.csv] [--paired paired_vs_en.csv]
                            [--permutations permutations.csv] [--native native_instruction.csv]
                            [--comparison comparison.csv] [--cs-ca cs_ca_gap.csv]
                            [--primary qwen2.5] [--contrast qwen3]

Which prediction file backs a table row (same rule as analysis.py's select()): the file of the PREFERRED part is
used when it exists (Part 1 tables -> part-1 files; Part 2/3/5 tables -> the part-2 file for v1_en LETTER runs,
part-3 for perm_*/v2_native, part-5 for the contrast model), otherwise any file with the same
(model, lang, scorer, variant). When several files exist and their predictions differ, a WARN is printed.

Checks (tolerance 0.001 on rates, exact on counts):
  results.csv        one row per (part, model, lang, scorer, prompt_variant, subset in {all, CS, CA}) whose n and
                     accuracy match the predictions; parse_fail_rate (GEN rows) = fraction of null preds.
                     Missing rows, extra rows and mismatches are reported. The `model` column may hold either the
                     full Hub id (as in the records) or its alias (qwen2.5 / qwen3 / ...); the script says which.
  GOLD-NOT-REMAPPED  any perm_* file within 0.03 of chance (0.25) while the same (model, lang) perm_ABCD
                     (or v1_en LETTER) file is above 0.35.
  permutations.csv   acc_ABCD..acc_DABC, mean, consistency_rate (fraction of items whose predicted option TEXT --
                     the pred letter mapped back through the order to the original option index -- is identical
                     across the four permutation files; the v1_en LETTER file stands in for a missing perm_ABCD).
                     std and frac_pred_* are checked loosely (warnings only).
  paired_vs_en.csv   n, acc_en, acc_lang, diff = acc_lang - acc_en, discordant counts, agreement_rate.
  native_instruction.csv  acc_v1_en, acc_v2_native, diff = acc_v2_native - acc_v1_en, discordant counts.
  comparison.csv     n, acc_qwen25, acc_qwen3, diff = acc_qwen3 - acc_qwen25, discordant counts,
                     mdd_95 = 1.96*sqrt(b+c)/n.
  cs_ca_gap.csv      n_cs, n_ca, acc_cs, acc_ca, gap = acc_cs - acc_ca from the qwen2.5 LETTER v1_en (part-2) files.
Every diff column is SECOND MINUS FIRST (acc_lang - acc_en, acc_v2_native - acc_v1_en, acc_qwen3 - acc_qwen25,
acc_cs - acc_ca); a flipped sign is a FAILURE, not a warning.
Confidence intervals are NOT checked here (the grader recomputes them separately).
Optional CSVs that are absent are reported as SKIP; results.csv is required. Exit 1 on any failure.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from harness import data as D
from harness import schema as S
from harness.model import MODELS
from harness.prompts import PERM_ORDERS

TOL = 0.001 + 1e-9
PRIMARY_DEFAULT, CONTRAST_DEFAULT = "qwen2.5", "qwen3"
ALIAS_OF = {hub: alias for alias, (hub, _rev) in MODELS.items()}
HUB_OF = {alias: hub for alias, (hub, _rev) in MODELS.items()}


# ---------------------------------------------------------------- helpers
class Report:
    def __init__(self):
        self.fails: list[str] = []
        self.warns: list[str] = []

    def fail(self, section: str, msg: str):
        self.fails.append(f"[{section}] {msg}")

    def warn(self, section: str, msg: str):
        self.warns.append(f"[{section}] {msg}")


def _num(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v


def _close(a, b, tol=TOL) -> bool:
    a, b = _num(a), _num(b)
    if math.isnan(a) or math.isnan(b):
        return False
    return abs(a - b) <= tol


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")


class Run:
    """One prediction file."""

    def __init__(self, path: str, recs: list[dict]):
        self.path, self.recs = path, recs
        r = recs[0]
        self.part, self.model, self.lang = str(r["part"]), r["model"], r["lang"]
        self.scorer, self.variant, self.benchmark = r["scorer"], r["prompt_variant"], r["benchmark"]
        self.by_id = {x["sample_id"]: x for x in recs}

    @property
    def key(self):
        return (self.part, self.model, self.lang, self.scorer, self.variant)

    def subset(self, name: str) -> list[dict]:
        if name == "all":
            return self.recs
        return [x for x in self.recs if x["cs_label"] == name]

    def acc(self, name: str = "all") -> float:
        rs = self.subset(name)
        return float(np.mean([x["pred"] == x["gold"] for x in rs])) if rs else math.nan


def load_runs(pred_dir: str, rep: Report) -> list[Run]:
    runs = []
    for p in sorted(glob.glob(os.path.join(pred_dir, "*.jsonl"))):
        try:
            recs = S.read_jsonl(p)
        except Exception as e:
            rep.fail("predictions", f"{p}: unreadable ({e})")
            continue
        if not recs:
            rep.fail("predictions", f"{p}: empty")
            continue
        if any(set(S.PRED_FIELDS) - set(r) for r in recs):
            rep.fail("predictions", f"{p}: records missing schema fields (run validate_predictions.py)")
            continue
        runs.append(Run(p, recs))
    return runs


def find_run(runs: list[Run], model: str, lang: str, scorer: str, variant: str, rep: Report | None = None,
             section: str = "", part: str | None = None) -> Run | None:
    """Pick the run for (model, lang, scorer, variant): the file of the preferred `part` when present, otherwise
    the first matching file in name order (the same rule as analysis.py's select()). Files of other parts with
    the same protocol are accepted (a Part 1 `en` LETTER v1_en run IS the Part 2 run); if the duplicates disagree
    on their predictions a WARN is issued (never a failure)."""
    cands = [r for r in runs if (r.model, r.lang, r.scorer, r.variant) == (model, lang, scorer, variant)
             and r.benchmark == "global_mmlu_lite"]
    if not cands:
        return None
    cands.sort(key=lambda r: r.path)
    pick = cands[0]
    if part is not None:
        for r in cands:
            if r.part == str(part):
                pick = r
                break
    if rep is not None and len(cands) > 1:
        preds = [tuple(sorted((k, v["pred"]) for k, v in r.by_id.items())) for r in cands]
        if any(p != preds[0] for p in preds):
            key = f"({model}, {lang}, {scorer}, {variant})"
            if key not in _WARNED_DUPES:
                _WARNED_DUPES.add(key)
                rep.warn(section, f"several files for {key} with differing predictions "
                                  f"({', '.join(os.path.basename(r.path) for r in cands)}); using "
                                  f"{os.path.basename(pick.path)} (preferred part {part})")
    return pick


_WARNED_DUPES: set = set()


def paired(a: Run, b: Run, rep: Report, section: str, label: str):
    """Align two runs on sample_id. Returns (ids, correct_a, correct_b) or None."""
    ids = sorted(set(a.by_id) & set(b.by_id))
    if not ids:
        rep.fail(section, f"{label}: no common sample_ids")
        return None
    if len(ids) != len(a.by_id) or len(ids) != len(b.by_id):
        rep.warn(section, f"{label}: item sets differ ({len(a.by_id)} vs {len(b.by_id)}); using {len(ids)} common items")
    ca = np.array([a.by_id[i]["pred"] == a.by_id[i]["gold"] for i in ids], dtype=bool)
    cb = np.array([b.by_id[i]["pred"] == b.by_id[i]["gold"] for i in ids], dtype=bool)
    return ids, ca, cb


def discordant(ca: np.ndarray, cb: np.ndarray) -> tuple[int, int]:
    return int((ca & ~cb).sum()), int((~ca & cb).sum())


def check_cols(df: pd.DataFrame, cols: list[str], rep: Report, section: str) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        rep.fail(section, f"missing columns {missing} (have {list(df.columns)})")
        return False
    return True


def cmp(rep: Report, section: str, where: str, col: str, got, want, exact=False):
    ok = (_num(got) == _num(want)) if exact else _close(got, want)
    if not ok:
        rep.fail(section, f"{where}: {col}={got!r} but recomputed {want:.4f}" if not exact
                 else f"{where}: {col}={got!r} but recomputed {want}")


# ---------------------------------------------------------------- results.csv
def check_results(runs: list[Run], path: str, rep: Report):
    sec = "results.csv"
    if not os.path.exists(path):
        rep.fail(sec, f"{path} not found")
        return
    df = read_csv(path)
    if not check_cols(df, S.RESULTS_COLUMNS, rep, sec):
        return
    expected = {}
    for r in runs:
        for sub in ("all", "CS", "CA"):
            rs = r.subset(sub)
            if not rs:
                continue
            row = {"n": len(rs), "accuracy": float(np.mean([x["pred"] == x["gold"] for x in rs])),
                   "benchmark": r.benchmark}
            if r.scorer == "GEN":
                row["parse_fail_rate"] = float(np.mean([x["pred"] is None for x in rs]))
            expected[r.key + (sub,)] = row
    seen = set()
    model_forms = set()
    for i, row in df.iterrows():
        model = row["model"].strip()
        if model in HUB_OF:                       # alias given instead of the full Hub id: accept, remember
            model_forms.add("alias")
            model = HUB_OF[model]
        elif model in ALIAS_OF:
            model_forms.add("hub id")
        key = (row["part"].strip(), model, row["lang"].strip(), row["scorer"].strip(),
               row["prompt_variant"].strip(), row["subset"].strip())
        where = f"row {i + 2} {key}"
        if key in seen:
            rep.fail(sec, f"{where}: duplicate row")
        seen.add(key)
        exp = expected.get(key)
        if exp is None:
            rep.fail(sec, f"{where}: no prediction file backs this row (extra row)")
            continue
        cmp(rep, sec, where, "n", row["n"], exp["n"], exact=True)
        cmp(rep, sec, where, "accuracy", row["accuracy"], exp["accuracy"])
        if row["benchmark"].strip() != exp["benchmark"]:
            rep.fail(sec, f"{where}: benchmark {row['benchmark']!r} != {exp['benchmark']!r}")
        if "parse_fail_rate" in exp:
            cmp(rep, sec, where, "parse_fail_rate", row["parse_fail_rate"], exp["parse_fail_rate"])
        elif row["parse_fail_rate"].strip() not in ("", "nan", "NaN"):
            rep.warn(sec, f"{where}: parse_fail_rate should be empty for non-GEN rows")
    for key in sorted(expected):
        if key not in seen:
            rep.fail(sec, f"missing row {key}")
    if model_forms:
        form = " and ".join(sorted(model_forms))
        print(f"results.csv: model column holds the {form}" + (" (mixing both is allowed but untidy)" if len(model_forms) > 1 else ""))
        if len(model_forms) > 1:
            rep.warn(sec, "model column mixes aliases and full Hub ids")
    print(f"results.csv: {len(df)} rows, {len(expected)} expected, {len(seen & set(expected))} matched by key")


# ---------------------------------------------------------------- gold-not-remapped flag
def check_remap(runs: list[Run], rep: Report):
    sec = "GOLD-NOT-REMAPPED"
    n_checked = 0
    for r in runs:
        if not r.variant.startswith("perm_"):
            continue
        ref = find_run(runs, r.model, r.lang, r.scorer, "perm_ABCD") or find_run(runs, r.model, r.lang, "LETTER", "v1_en")
        if ref is None or ref is r:
            continue
        n_checked += 1
        acc, ref_acc = r.acc(), ref.acc()
        if abs(acc - 0.25) <= 0.03 and ref_acc > 0.35:
            rep.fail(sec, f"{r.path}: accuracy {acc:.3f} is at chance while {ref.path} has {ref_acc:.3f} "
                          f"-- gold letter probably not remapped under the permutation")
    print(f"gold-remap flag: {n_checked} permutation file(s) checked")


# ---------------------------------------------------------------- permutations.csv
def orig_index(pred: str | None, order: str) -> int | None:
    """Original option index shown at the predicted letter under `order` (None for a null prediction)."""
    if pred is None:
        return None
    return D.LETTERS.index(order[D.LETTERS.index(pred)])


def check_permutations(runs: list[Run], path: str, model: str, rep: Report):
    sec = "permutations.csv"
    if not os.path.exists(path):
        print(f"{sec}: SKIP (not present)")
        return
    df = read_csv(path)
    cols = ["lang"] + [f"acc_{o}" for o in PERM_ORDERS] + ["mean", "std", "consistency_rate"] + \
           [f"frac_pred_{L}" for L in D.LETTERS]
    if not check_cols(df, cols, rep, sec):
        return
    for i, row in df.iterrows():
        lang = row["lang"].strip()
        where = f"row {i + 2} lang={lang}"
        files = {}
        for o in PERM_ORDERS:
            r = find_run(runs, model, lang, "LETTER", f"perm_{o}", rep, sec, part="3")
            if r is None and o == "ABCD":
                r = find_run(runs, model, lang, "LETTER", "v1_en", rep, sec, part="2")
            if r is None:
                if math.isnan(_num(row[f"acc_{o}"])):
                    rep.warn(sec, f"{where}: no prediction file for perm_{o} and acc_{o} is NaN (Part 3 run still missing)")
                else:
                    rep.fail(sec, f"{where}: no prediction file for perm_{o} but acc_{o}={row[f'acc_{o}']!r}")
            files[o] = r
        accs = []
        for o, r in files.items():
            if r is None:
                continue
            accs.append(r.acc())
            cmp(rep, sec, where, f"acc_{o}", row[f"acc_{o}"], r.acc())
        if len(accs) == 4:
            cmp(rep, sec, where, "mean", row["mean"], float(np.mean(accs)))
            std0, std1 = float(np.std(accs)), float(np.std(accs, ddof=1))
            if not (_close(row["std"], std0) or _close(row["std"], std1)):
                rep.fail(sec, f"{where}: std={row['std']!r} but recomputed {std0:.4f} (ddof=0) / {std1:.4f} (ddof=1)")
            # consistency: original option index identical across the four files
            ids = set.intersection(*(set(r.by_id) for r in files.values() if r is not None))
            if len(ids) != min(len(r.by_id) for r in files.values()):
                rep.warn(sec, f"{where}: permutation files do not share the same items; using {len(ids)} common")
            if ids:
                same = [len({orig_index(files[o].by_id[s]["pred"], o) for o in PERM_ORDERS}) == 1 for s in sorted(ids)]
                cmp(rep, sec, where, "consistency_rate", row["consistency_rate"], float(np.mean(same)))
            # letter frequencies pooled over the four files (warning only: pooling convention may differ)
            pooled = [x["pred"] for r in files.values() for x in r.recs]
            for L in D.LETTERS:
                want = float(np.mean([p == L for p in pooled]))
                if not _close(row[f"frac_pred_{L}"], want):
                    rep.warn(sec, f"{where}: frac_pred_{L}={row[f'frac_pred_{L}']!r} but pooled recomputation {want:.4f}")
    print(f"{sec}: {len(df)} rows checked")


# ---------------------------------------------------------------- paired CSVs
def check_pairwise(runs: list[Run], path: str, rep: Report, sec: str, cols: list[str],
                   pick_a, pick_b, col_a: str, col_b: str, n_a_col: str, n_b_col: str,
                   diff_is_b_minus_a: bool, extra=None):
    """Generic checker: per row, run a = pick_a(lang), run b = pick_b(lang)."""
    if not os.path.exists(path):
        print(f"{sec}: SKIP (not present)")
        return
    df = read_csv(path)
    if not check_cols(df, cols, rep, sec):
        return
    for i, row in df.iterrows():
        lang = row["lang"].strip()
        where = f"row {i + 2} lang={lang}"
        a, b = pick_a(lang), pick_b(lang)
        if a is None or b is None:
            rep.fail(sec, f"{where}: missing prediction file for {col_a if a is None else col_b}")
            continue
        got = paired(a, b, rep, sec, where)
        if got is None:
            continue
        ids, ca, cb = got
        n = len(ids)
        if "n" in cols:
            cmp(rep, sec, where, "n", row["n"], n, exact=True)
        acc_a, acc_b = float(ca.mean()), float(cb.mean())
        cmp(rep, sec, where, col_a, row[col_a], acc_a)
        cmp(rep, sec, where, col_b, row[col_b], acc_b)
        want_diff = (acc_b - acc_a) if diff_is_b_minus_a else (acc_a - acc_b)
        if not _close(row["diff"], want_diff):
            if _close(row["diff"], -want_diff) and abs(want_diff) > TOL:
                rep.fail(sec, f"{where}: diff={row['diff']!r} has the WRONG SIGN: diff must be {col_b} - {col_a} "
                              f"(second minus first) = {want_diff:+.4f}")
            else:
                rep.fail(sec, f"{where}: diff={row['diff']!r} but recomputed {want_diff:.4f}")
        n_a, n_b = discordant(ca, cb)
        cmp(rep, sec, where, n_a_col, row[n_a_col], n_a, exact=True)
        cmp(rep, sec, where, n_b_col, row[n_b_col], n_b, exact=True)
        if extra is not None:
            extra(row, where, ids, a, b, ca, cb)
    print(f"{sec}: {len(df)} rows checked")


def check_paired_vs_en(runs, path, model, rep):
    sec = "paired_vs_en.csv"
    cols = ["lang", "n", "acc_en", "acc_lang", "diff", "paired_ci_low", "paired_ci_high",
            "n_en_only_correct", "n_lang_only_correct", "agreement_rate"]

    def extra(row, where, ids, a, b, ca, cb):
        agree = float(np.mean([a.by_id[s]["pred"] == b.by_id[s]["pred"] for s in ids]))
        cmp(rep, sec, where, "agreement_rate", row["agreement_rate"], agree)

    check_pairwise(runs, path, rep, sec, cols,
                   lambda lang: find_run(runs, model, "en", "LETTER", "v1_en", rep, sec, part="2"),
                   lambda lang: find_run(runs, model, lang, "LETTER", "v1_en", rep, sec, part="2"),
                   "acc_en", "acc_lang", "n_en_only_correct", "n_lang_only_correct",
                   diff_is_b_minus_a=True, extra=extra)


def check_native(runs, path, model, rep):
    sec = "native_instruction.csv"
    cols = ["lang", "acc_v1_en", "acc_v2_native", "diff", "paired_ci_low", "paired_ci_high",
            "n_v1_only_correct", "n_v2_only_correct"]
    check_pairwise(runs, path, rep, sec, cols,
                   lambda lang: find_run(runs, model, lang, "LETTER", "v1_en", rep, sec, part="2"),
                   lambda lang: find_run(runs, model, lang, "LETTER", "v2_native", rep, sec, part="3"),
                   "acc_v1_en", "acc_v2_native", "n_v1_only_correct", "n_v2_only_correct",
                   diff_is_b_minus_a=True)


def check_comparison(runs, path, primary, contrast, rep):
    sec = "comparison.csv"
    cols = ["lang", "n", "acc_qwen25", "acc_qwen3", "diff", "paired_ci_low", "paired_ci_high", "unpaired_overlap",
            "n_qwen25_only_correct", "n_qwen3_only_correct", "mdd_95"]

    def extra(row, where, ids, a, b, ca, cb):
        n_a, n_b = discordant(ca, cb)
        mdd = 1.96 * math.sqrt(n_a + n_b) / len(ids)
        cmp(rep, sec, where, "mdd_95", row["mdd_95"], mdd)

    check_pairwise(runs, path, rep, sec, cols,
                   lambda lang: find_run(runs, primary, lang, "LETTER", "v1_en", rep, sec, part="2"),
                   lambda lang: find_run(runs, contrast, lang, "LETTER", "v1_en", rep, sec, part="5"),
                   "acc_qwen25", "acc_qwen3", "n_qwen25_only_correct", "n_qwen3_only_correct",
                   diff_is_b_minus_a=True, extra=extra)     # diff = acc_qwen3 - acc_qwen25 (second minus first)


# ---------------------------------------------------------------- cs_ca_gap.csv
def check_cs_ca_gap(runs, path, model, rep):
    """n_cs, n_ca, acc_cs, acc_ca and gap = acc_cs - acc_ca from the (part-2) LETTER v1_en file of each language.
    The unpaired CI columns and `detectable` are not checked here."""
    sec = "cs_ca_gap.csv"
    cols = ["lang", "n_cs", "n_ca", "acc_cs", "acc_ca", "gap", "unpaired_ci_low", "unpaired_ci_high", "detectable"]
    if not os.path.exists(path):
        print(f"{sec}: SKIP (not present)")
        return
    df = read_csv(path)
    if not check_cols(df, cols, rep, sec):
        return
    for i, row in df.iterrows():
        lang = row["lang"].strip()
        where = f"row {i + 2} lang={lang}"
        r = find_run(runs, model, lang, "LETTER", "v1_en", rep, sec, part="2")
        if r is None:
            rep.fail(sec, f"{where}: no LETTER v1_en prediction file for {lang}")
            continue
        cs, ca = r.subset("CS"), r.subset("CA")
        if not cs or not ca:
            rep.fail(sec, f"{where}: file has no CS or no CA items ({len(cs)} CS, {len(ca)} CA)")
            continue
        acc_cs, acc_ca = r.acc("CS"), r.acc("CA")
        cmp(rep, sec, where, "n_cs", row["n_cs"], len(cs), exact=True)
        cmp(rep, sec, where, "n_ca", row["n_ca"], len(ca), exact=True)
        cmp(rep, sec, where, "acc_cs", row["acc_cs"], acc_cs)
        cmp(rep, sec, where, "acc_ca", row["acc_ca"], acc_ca)
        want = acc_cs - acc_ca
        if not _close(row["gap"], want):
            if _close(row["gap"], -want) and abs(want) > TOL:
                rep.fail(sec, f"{where}: gap={row['gap']!r} has the WRONG SIGN: gap must be acc_cs - acc_ca = {want:+.4f}")
            else:
                rep.fail(sec, f"{where}: gap={row['gap']!r} but recomputed {want:.4f}")
    print(f"{sec}: {len(df)} rows checked")


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", default="predictions")
    ap.add_argument("--results", default="results.csv")
    ap.add_argument("--paired", default="paired_vs_en.csv")
    ap.add_argument("--permutations", default="permutations.csv")
    ap.add_argument("--native", default="native_instruction.csv")
    ap.add_argument("--comparison", default="comparison.csv")
    ap.add_argument("--cs-ca", default="cs_ca_gap.csv")
    ap.add_argument("--primary", default=PRIMARY_DEFAULT, help="alias of the primary model (Parts 1-4)")
    ap.add_argument("--contrast", default=CONTRAST_DEFAULT, help="alias of the Part 5 contrast model")
    args = ap.parse_args(argv)

    rep = Report()
    runs = load_runs(args.predictions, rep)
    print(f"loaded {len(runs)} prediction file(s) from {args.predictions}/")
    if not runs:
        rep.fail("predictions", "no prediction files")
    primary, contrast = MODELS[args.primary][0], MODELS[args.contrast][0]

    check_results(runs, args.results, rep)
    check_remap(runs, rep)
    check_permutations(runs, args.permutations, primary, rep)
    check_paired_vs_en(runs, args.paired, primary, rep)
    check_native(runs, args.native, primary, rep)
    check_comparison(runs, args.comparison, primary, contrast, rep)
    check_cs_ca_gap(runs, args.cs_ca, primary, rep)

    print()
    for w in rep.warns:
        print("WARN  " + w)
    for f in rep.fails:
        print("FAIL  " + f)
    print(f"\n{'ALL CHECKS PASSED' if not rep.fails else f'{len(rep.fails)} FAILURE(S)'}; {len(rep.warns)} warning(s)")
    return 1 if rep.fails else 0


if __name__ == "__main__":
    sys.exit(main())
