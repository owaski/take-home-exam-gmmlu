#!/usr/bin/env python
"""grade.py -- held-back grading script (INSTRUCTOR ONLY; never shipped to students).

Usage:
    python instructor/grade.py <submission_dir> [--package-root <repo root>] [--reference instructor/reference_results]
                               [--rerun-en] [--rerun-device auto|cpu|mps|cuda] [--out grade_report.md]
                               [--n-expected N]   (DEBUG ONLY: validate files produced with `--n N`)

What it does (every step is isolated; one failure never stops the report):
  1. runs the INSTRUCTOR copies of validate_predictions.py / check_results.py against the submission
     (cwd = submission, sys.path = instructor overlay first, so a tampered validator/harness in the submission
     is irrelevant for the schema check);
  2. recomputes every CSV from the submission's predictions/ with instructor/solution/analysis.py (same
     harness.stats functions, B=2000, seed=0) and diffs against the submitted CSVs
     (rates 0.001, CIs 0.01, counts exact);
  3. plausibility bands (reference/bands.json: {"alias|lang|scorer|variant": {"acc","lo","hi"}}; when smoke_ok.json
     says lite_mode = true the "lite:alias|lang|scorer|variant" key is used first, falling back to the unprefixed one);
  4. gold-not-remapped flag + mean-over-permutations vs the reference mean (tolerance 0.015);
  5. tokenizer.csv within 10 % of reference/tokenizer.csv;
  6. probe.jsonl: ids per (model, lang) are a subset of harness.data.probe_ids(student_id) -- always drawn from the
     full 400-item en list -- with exactly 20 ids in full mode or |probe ids INTERSECT LITE subset| in LITE mode;
     per-item agreement and score-vector Pearson correlation against reference/predictions/ (fabrication flag);
  7. run_log.txt sanity (items/s vs device, all Part 2/3/5 runs logged, identical wall-clocks);
  8. integrity.txt / smoke_ok.json / README / PDFs present, LITE-mode consistency, frozen files unchanged,
     scipy/statsmodels absent from stats.py;
  9. optional --rerun-en: re-run the submission's OWN harness on Part 2 `en` in a temp copy and compare.
Then prints/writes a Markdown report with a pre-filled rubric table (automatic lines scored, manual lines ___).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PACKAGE_ROOT = os.path.dirname(HERE)

CORE_LANGS = ["en", "de", "zh", "ar", "hi", "id", "sw", "yo"]
PERM_LANGS = ["en", "zh", "hi", "sw"]
NATIVE_LANGS = ["zh", "hi", "sw"]
PART1_LANGS = ["en", "zh", "sw"]
PART1_SCORERS = ["LETTER", "CONT", "CONT_TOKNORM", "CONT_CHARNORM", "GEN"]
PERM_ORDERS = ["ABCD", "BCDA", "CDAB", "DABC"]
PRIMARY, CONTRAST = "qwen2.5", "qwen3"
CSV_FILES = ["results.csv", "paired_vs_en.csv", "permutations.csv", "native_instruction.csv", "comparison.csv",
             "cs_ca_gap.csv", "part1_scorers.csv"]
FIGURES = ["accuracy_by_language.png", "position_bias.png", "fertility_vs_accuracy.png"]
FROZEN_HARNESS_FILES = ["data.py", "model.py", "schema.py", "cli.py", "__main__.py"]
TOL_RATE, TOL_CI, TOL_PERM_MEAN, TOL_TOKENIZER_REL = 0.001, 0.01, 0.015, 0.10
IPS_FLAG = {"cpu": 20.0, "mps": 200.0}


# ============================================================================================ report plumbing
class Report:
    """Collects sections, flags and automatic rubric scores; renders Markdown."""

    def __init__(self):
        self.sections: list[tuple[str, list[str]]] = []
        self.flags: list[str] = []
        self.errors: list[str] = []
        self.scores: dict[str, tuple[Optional[float], float, str]] = {}   # rubric line -> (auto pts | None, max, note)
        self.facts: dict[str, Any] = {}

    def section(self, title: str) -> list[str]:
        lines: list[str] = []
        self.sections.append((title, lines))
        return lines

    def flag(self, msg: str) -> None:
        self.flags.append(msg)

    def score(self, line: str, pts: Optional[float], mx: float, note: str = "") -> None:
        self.scores[line] = (pts, mx, note)

    def render(self, submission: str) -> str:
        out = [f"# Grade report: `{submission}`", "", f"generated {time.strftime('%Y-%m-%d %H:%M:%S')} by instructor/grade.py", ""]
        if self.flags:
            out += ["## FLAGS (read these first)", ""] + [f"- **{f}**" for f in self.flags] + [""]
        else:
            out += ["## FLAGS: none", ""]
        if self.errors:
            out += ["## Grader-internal errors (steps that could not run)", ""] + [f"- {e}" for e in self.errors] + [""]
        for title, lines in self.sections:
            out += [f"## {title}", ""] + (lines or ["(nothing to report)"]) + [""]
        out += ["## Rubric (auto lines pre-filled; manual lines = ___)", "",
                "| # | Criterion | Max | Auto | Final | Notes |", "|---|---|---|---|---|---|"]
        total_auto, total_max = 0.0, 0.0
        for line, (pts, mx, note) in self.scores.items():
            crit = RUBRIC_TITLES.get(line, "")
            auto = "___" if pts is None else f"{pts:g}"
            if pts is not None:
                total_auto += pts
            total_max += mx
            out.append(f"| {line} | {crit} | {mx:g} | {auto} | ___ | {note} |")
        out += ["", f"Automatic subtotal: **{total_auto:g}** of the {total_max:g} rubric points "
                    f"(manual lines and the manual parts of mixed lines are not included).", "",
                "Penalties to apply by hand: late (-10/day), validator failures after the grace window (-5/file), "
                "disallowed model / altered frozen template (part = 0), missing integrity.txt or smoke_ok.json "
                "(not graded until supplied), report claims contradicted by CSVs (-1 each, max -5).", ""]
        return "\n".join(out)


RUBRIC_TITLES = {
    "P1-code": "Three scorers", "P1-analysis": "Scorer comparison", "P2-runs": "Sweep, results.csv",
    "P2-figure": "accuracy_by_language.png", "P2-stats": "Statistics reasoning", "P2-parallel": "Parallel-item analysis",
    "P3-runs": "Permutation + native runs", "P3-figure": "Position bias", "P3-analysis": "Sensitivity reasoning",
    "P4-csv": "tokenizer.csv", "P4-figure": "Fertility plot", "P4-analysis": "Fertility reasoning",
    "P5-runs": "Qwen3 sweep", "P5-stats": "Paired statistics", "P5-analysis": "Model comparison",
    "P6-report": "Report", "P6-reflection": "Reflection", "P6-repro": "Reproducibility", "Bonus": "S1-S5",
}
RUBRIC_MAX = {"P1-code": 8, "P1-analysis": 7, "P2-runs": 8, "P2-figure": 2, "P2-stats": 5, "P2-parallel": 5,
              "P3-runs": 7, "P3-figure": 3, "P3-analysis": 5, "P4-csv": 4, "P4-figure": 1, "P4-analysis": 5,
              "P5-runs": 5, "P5-stats": 5, "P5-analysis": 5, "P6-report": 15, "P6-reflection": 5, "P6-repro": 5,
              "Bonus": 10}


def step(rep: Report, title: str) -> Callable:
    """Decorator: run a grading step, catch everything, record the traceback in the report."""
    def deco(fn):
        def run(*a, **k):
            lines = rep.section(title)
            try:
                return fn(lines, *a, **k)
            except Exception as e:  # noqa: BLE001
                tb = traceback.format_exc().strip().splitlines()[-1]
                rep.errors.append(f"{title}: {type(e).__name__}: {e}")
                lines.append(f"- grader step failed: `{tb}`")
                return None
        return run
    return deco


# ============================================================================================ overlay + imports
def build_overlay(package_root: str, tmp: str) -> str:
    """student/ + instructor/solution overlay -> a working instructor harness dir (validators, analysis, harness)."""
    student = os.path.join(package_root, "student")
    sol = os.path.join(package_root, "instructor", "solution")
    dst = os.path.join(tmp, "overlay")
    shutil.copytree(student, dst, ignore=shutil.ignore_patterns("predictions", "figures", "__pycache__", "*.jsonl",
                                                                 "*.csv", "*.pdf", "run_log.txt", "smoke_ok.json"))
    for f in glob.glob(os.path.join(sol, "harness", "*.py")):
        shutil.copy(f, os.path.join(dst, "harness", os.path.basename(f)))
    for f in ("analysis.py", "tokenizer_stats.py"):
        if os.path.exists(os.path.join(sol, f)):
            shutil.copy(os.path.join(sol, f), os.path.join(dst, f))
    return dst


def import_from(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # type: ignore[union-attr]
    return mod


def run_subprocess(cmd: list[str], cwd: str, env_extra: Optional[dict] = None, timeout: int = 1800) -> tuple[int, str]:
    env = dict(os.environ)
    env.update(env_extra or {})
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + ("\n" + p.stderr if p.stderr.strip() else ""))
    except subprocess.TimeoutExpired as e:
        return 124, f"TIMEOUT after {timeout}s\n{e.stdout or ''}\n{e.stderr or ''}"


def read_csv_str(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def fnum(x) -> float:
    try:
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none", "null"):
            return math.nan
        if s.lower() in ("true", "false"):
            return 1.0 if s.lower() == "true" else 0.0
        return float(s)
    except Exception:  # noqa: BLE001
        return math.nan


def tail(text: str, n: int = 40) -> list[str]:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) > n:
        lines = [f"... ({len(lines) - n} lines omitted) ..."] + lines[-n:]
    return ["```"] + lines + ["```"]


# ============================================================================================ step 1: validators
def step1_validators(lines, rep: Report, sub: str, overlay: str, n_expected: Optional[int]) -> dict:
    env = {"PYTHONPATH": overlay, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1")}
    res = {}
    cmd = [sys.executable, os.path.join(overlay, "validate_predictions.py"), "--dir", "predictions"]
    if n_expected:
        cmd += ["--n-expected", str(n_expected)]
        lines.append(f"- DEBUG: `--n-expected {n_expected}` passed to the validator (never for real grading)")
    rc, out = run_subprocess(cmd, cwd=sub, env_extra=env)
    res["validate_rc"] = rc
    n_fail = len(re.findall(r"^FAIL ", out, flags=re.M))
    n_ok = len(re.findall(r"^OK ", out, flags=re.M))
    res["validate_ok"], res["validate_fail"] = n_ok, n_fail
    lines.append(f"- validate_predictions.py (instructor copy): exit {rc}; {n_ok} OK, {n_fail} FAIL")
    if rc != 0:
        rep.flag(f"validate_predictions.py failed on {n_fail} file(s) (exit {rc})")
        lines += tail(out, 30)
    cmd = [sys.executable, os.path.join(overlay, "check_results.py")]
    rc, out = run_subprocess(cmd, cwd=sub, env_extra=env)
    res["check_rc"] = rc
    fails = [ln for ln in out.splitlines() if ln.startswith("FAIL ")]
    warns = [ln for ln in out.splitlines() if ln.startswith("WARN ")]
    res["check_fails"], res["check_warns"] = fails, warns
    res["remap_flags"] = [ln for ln in fails if "GOLD-NOT-REMAPPED" in ln]
    lines.append(f"- check_results.py (instructor copy): exit {rc}; {len(fails)} FAIL, {len(warns)} WARN")
    if rc != 0:
        lines += tail("\n".join(fails + warns), 40)
    for f in res["remap_flags"]:
        rep.flag("GOLD NOT REMAPPED: " + f)
    return res


# ============================================================================================ step 2: recompute
def diff_tables(sub_df: Optional[pd.DataFrame], ref_df: pd.DataFrame, keys: list[str], name: str,
                lines: list[str], col_tol: Optional[dict] = None) -> dict:
    """Compare a submitted CSV with the recomputed one row-by-row on `keys`. Returns counts."""
    out = {"present": sub_df is not None, "missing_rows": 0, "extra_rows": 0, "mismatch": 0, "ci_mismatch": 0,
           "checked": 0, "missing_cols": []}
    if sub_df is None:
        lines.append(f"- **{name}: not submitted**")
        return out
    if not len(ref_df):
        lines.append(f"- {name}: nothing to recompute (no prediction files for this table); submitted {len(sub_df)} rows")
        return out
    missing_cols = [c for c in ref_df.columns if c not in sub_df.columns]
    out["missing_cols"] = missing_cols
    if missing_cols:
        lines.append(f"- {name}: submitted file lacks columns {missing_cols}")
    if any(k not in sub_df.columns for k in keys):
        lines.append(f"- **{name}: key columns {keys} missing; cannot compare**")
        out["mismatch"] = len(ref_df)
        return out
    sub_idx = {tuple(str(r[k]).strip() for k in keys): r for _, r in sub_df.iterrows()}
    ref_idx = {tuple(str(r[k]).strip() for k in keys): r for _, r in ref_df.iterrows()}
    out["extra_rows"] = len(set(sub_idx) - set(ref_idx))
    problems = []
    for key, rr in ref_idx.items():
        sr = sub_idx.get(key)
        if sr is None:
            out["missing_rows"] += 1
            problems.append(f"missing row {key}")
            continue
        for col in ref_df.columns:
            if col in keys or col in missing_cols or col in ("wall_clock_sec",):
                continue
            want, got = fnum(rr[col]), fnum(sr[col])
            if math.isnan(want) and math.isnan(got):
                continue
            tol, kind = tolerance_for(col, col_tol)
            out["checked"] += 1
            bad = (math.isnan(want) != math.isnan(got)) or (kind == "rel" and abs(got - want) > tol * max(abs(want), 1e-9)) \
                or (kind != "rel" and abs(got - want) > tol + 1e-9)
            if bad:
                out["mismatch"] += 1
                if "ci" in col.lower():
                    out["ci_mismatch"] += 1
                if len(problems) < 25:
                    problems.append(f"{key} {col}: submitted {sr[col]!s} vs recomputed {rr[col]!s} (tol {tol}{'%' if kind == 'rel' else ''})")
    status = "OK" if not (out["missing_rows"] or out["mismatch"]) else "MISMATCH"
    lines.append(f"- {name}: {status} — {len(ref_df)} recomputed rows, {out['missing_rows']} missing, {out['extra_rows']} extra, "
                 f"{out['mismatch']} value mismatches ({out['ci_mismatch']} in CI columns) of {out['checked']} checked")
    for p in problems[:25]:
        lines.append(f"    - {p}")
    return out


def tolerance_for(col: str, col_tol: Optional[dict]) -> tuple[float, str]:
    if col_tol and col in col_tol:
        return col_tol[col]
    c = col.lower()
    if c == "n" or c.startswith("n_") or c in ("detectable", "unpaired_overlap") or c.startswith("n_items"):
        return 0.0, "abs"
    if "ci_" in c or c.endswith("_ci") or c in ("ci_low", "ci_high"):
        return TOL_CI, "abs"
    if c == "std":
        return 0.005, "abs"
    return TOL_RATE, "abs"


def step2_recompute(lines, rep: Report, sub: str, overlay: str, tmp: str) -> dict:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    sys.path.insert(0, overlay)
    A = import_from(os.path.join(overlay, "analysis.py"), "instructor_analysis")
    res: dict = {"A": A}
    pred_dir = os.path.join(sub, "predictions")
    preds = A.load_predictions(pred_dir) if os.path.isdir(pred_dir) else {}
    res["preds"] = preds
    lines.append(f"- {len(preds)} readable prediction files in predictions/")
    if not preds:
        rep.flag("no prediction files")
        return res
    run_log = A.load_run_log(os.path.join(sub, "run_log.txt"))
    ref = {
        "results.csv": A.build_results(preds, 2000, 0, run_log),
        "paired_vs_en.csv": A.build_paired_vs_en(preds, 2000, 0),
        "permutations.csv": A.build_permutations(preds),
        "native_instruction.csv": A.build_native_instruction(preds, 2000, 0),
        "comparison.csv": A.build_comparison(preds, 2000, 0),
        "cs_ca_gap.csv": A.build_cs_ca_gap(preds, 2000, 0),
        "part1_scorers.csv": A.build_part1_scorers(preds, 2000, 0),
    }
    outdir = os.path.join(tmp, "recomputed")
    os.makedirs(outdir, exist_ok=True)
    for name, df in ref.items():
        df.to_csv(os.path.join(outdir, name), index=False)
    res["ref"] = ref
    keys = {"results.csv": ["part", "model", "lang", "scorer", "prompt_variant", "subset"],
            "paired_vs_en.csv": ["lang"], "permutations.csv": ["lang"], "native_instruction.csv": ["lang"],
            "comparison.csv": ["lang"], "cs_ca_gap.csv": ["lang"], "part1_scorers.csv": ["scorer", "lang"]}
    diffs = {}
    for name in CSV_FILES:
        sub_df = read_csv_str(os.path.join(sub, name))
        if name == "results.csv" and sub_df is not None and "part" in sub_df.columns:
            sub_df["part"] = sub_df["part"].astype(str).str.strip()
        diffs[name] = diff_tables(sub_df, ref[name], keys[name], name, lines)
    res["diffs"] = diffs
    lines.append(f"- recomputed CSVs written to `{outdir}` (kept only while the grader runs unless --keep-tmp)")
    return res


# ============================================================================================ helpers on preds
def acc_of(rows) -> float:
    return float(np.mean([r["pred"] == r["gold"] for r in rows])) if rows else math.nan


def run_index(A, preds: dict) -> dict[tuple[str, str, str, str], list[dict]]:
    """(alias, lang, scorer, variant) -> rows (first file wins; part ignored)."""
    idx = {}
    for rows in preds.values():
        if rows[0].get("benchmark", "global_mmlu_lite") != "global_mmlu_lite":
            continue
        k = A.run_key(rows)[1:]
        idx.setdefault(k, rows)
    return idx


# ============================================================================================ step 3: bands
def load_bands(reference: str) -> dict:
    p = os.path.join(reference, "bands.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def band_for(bands: dict, key: str, lite: bool) -> Optional[dict]:
    """Band for "alias|lang|scorer|variant": in LITE mode the "lite:<key>" entry (tolerance 0.065, n=200 reference)
    is preferred and the unprefixed full-mode key is the fallback; in full mode only the unprefixed key is used."""
    if lite:
        b = bands.get("lite:" + key)
        if b is not None:
            return b
    return bands.get(key)


def step3_bands(lines, rep: Report, A, preds: dict, bands: dict, lite: bool = False) -> dict:
    res = {"in_band": {}, "out_of_band": [], "no_band": [], "lite": lite}
    if not bands:
        lines.append("- reference bands.json not found: band checks skipped (every band line is scored as 'present')")
    elif lite:
        n_lite = sum(1 for k in bands if k.startswith("lite:"))
        lines.append(f"- LITE submission: 'lite:' bands used where available ({n_lite} in bands.json), full-mode bands otherwise")
    idx = run_index(A, preds)
    for (alias, lang, scorer, variant), rows in sorted(idx.items()):
        key = f"{alias}|{lang}|{scorer}|{variant}"
        acc = acc_of(rows)
        b = band_for(bands, key, lite)
        if b is None:
            res["no_band"].append(key)
            continue
        ok = b["lo"] - 1e-9 <= acc <= b["hi"] + 1e-9
        res["in_band"][key] = ok
        mark = "in band" if ok else "**OUT OF BAND**"
        used = "lite band" if (lite and ("lite:" + key) in bands) else "band"
        lines.append(f"- {key}: acc {acc:.3f} (n={len(rows)}) vs {used} [{b['lo']:.3f}, {b['hi']:.3f}] (ref {b['acc']:.3f}) — {mark}")
        if not ok:
            res["out_of_band"].append(key)
    if res["out_of_band"]:
        rep.flag(f"{len(res['out_of_band'])} run(s) outside the plausibility band: {', '.join(res['out_of_band'])}")
    if res["no_band"] and bands:
        lines.append(f"- no band for: {', '.join(res['no_band'])}")
    return res


# ============================================================================================ step 4: permutations
def step4_permutations(lines, rep: Report, A, preds: dict, bands: dict, remap_flags: list, lite: bool = False) -> dict:
    res = {"langs_ok": [], "langs_bad": [], "langs_missing": [], "n_perm_runs": 0, "n_native_runs": 0}
    idx = run_index(A, preds)
    for lang in PERM_LANGS:
        runs = A.perm_runs(preds, lang, PRIMARY)
        n_new = sum(1 for o in PERM_ORDERS[1:] if o in runs)
        res["n_perm_runs"] += n_new
        if len(runs) < 4:
            res["langs_missing"].append(lang)
            lines.append(f"- {lang}: only {len(runs)}/4 permutation runs present ({sorted(runs)})")
            continue
        accs = {o: acc_of(runs[o]) for o in PERM_ORDERS}
        mean = float(np.mean(list(accs.values())))
        ref_accs = []
        for o in PERM_ORDERS:
            b = band_for(bands, f"{PRIMARY}|{lang}|LETTER|perm_{o}", lite) or \
                (band_for(bands, f"{PRIMARY}|{lang}|LETTER|v1_en", lite) if o == "ABCD" else None)
            if b is not None:
                ref_accs.append(b["acc"])
        txt = " ".join(f"{o}={accs[o]:.3f}" for o in PERM_ORDERS)
        if len(ref_accs) == 4:
            ref_mean = float(np.mean(ref_accs))
            ok = abs(mean - ref_mean) <= TOL_PERM_MEAN
            (res["langs_ok"] if ok else res["langs_bad"]).append(lang)
            lines.append(f"- {lang}: {txt}; mean {mean:.3f} vs reference mean {ref_mean:.3f} "
                         f"({'OK' if ok else '**OFF by more than 0.015**'}); std {np.std(list(accs.values())):.3f}")
        else:
            res["langs_ok"].append(lang)
            lines.append(f"- {lang}: {txt}; mean {mean:.3f} (no reference means available; not checked)")
    for lang in NATIVE_LANGS:
        if (PRIMARY, lang, "LETTER", "v2_native") in idx:
            res["n_native_runs"] += 1
        else:
            lines.append(f"- native run missing for {lang}")
    lines.append(f"- {res['n_perm_runs']}/12 new permutation runs, {res['n_native_runs']}/3 native runs present; "
                 f"gold-not-remapped flags from check_results: {len(remap_flags)}")
    if res["langs_bad"]:
        rep.flag(f"permutation mean off the reference by > 0.015 for {res['langs_bad']}")
    return res


# ============================================================================================ step 5: tokenizer
def step5_tokenizer(lines, rep: Report, sub: str, reference: str) -> dict:
    res = {"present": False, "n_tokenizers": 0, "n_rows": 0, "n_bad": 0, "n_checked": 0, "zh_nan": None, "ref": False}
    p = os.path.join(sub, "tokenizer.csv")
    df = read_csv_str(p)
    if df is None:
        lines.append("- **tokenizer.csv not submitted**")
        return res
    res["present"] = True
    res["n_rows"] = len(df)
    need = ["tokenizer", "lang", "n_items", "mean_tokens", "tokens_per_char", "tokens_per_word", "frac_partial_char_tokens"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        lines.append(f"- tokenizer.csv lacks columns {missing}")
    if "tokenizer" in df.columns:
        toks = sorted(set(df["tokenizer"].astype(str)))
        res["n_tokenizers"] = len(toks)
        lines.append(f"- tokenizers: {toks}; {len(df)} rows")
        if "lang" in df.columns:
            for t in toks:
                langs = sorted(set(df[df.tokenizer == t]["lang"]))
                miss = [l for l in CORE_LANGS if l not in langs]
                if miss:
                    lines.append(f"- {t}: missing languages {miss}")
    if "tokens_per_word" in df.columns and "lang" in df.columns:
        zh = df[df["lang"] == "zh"]["tokens_per_word"].map(fnum)
        res["zh_nan"] = bool(len(zh)) and bool(zh.isna().all())
        lines.append(f"- zh tokens_per_word is NaN: {res['zh_nan']}")
    rp = os.path.join(reference, "tokenizer.csv")
    ref = read_csv_str(rp)
    if ref is None:
        lines.append("- reference tokenizer.csv not found: value check skipped")
        return res
    res["ref"] = True
    ref_idx = {(str(r["tokenizer"]), str(r["lang"])): r for _, r in ref.iterrows()}
    alias_map = {"Qwen/Qwen2.5-0.5B-Instruct": "qwen2.5", "Qwen/Qwen3-0.6B": "qwen3", "HuggingFaceTB/SmolLM2-360M-Instruct": "smollm2"}
    bad = []
    for _, r in df.iterrows():
        t = alias_map.get(str(r.get("tokenizer", "")), str(r.get("tokenizer", "")))
        rr = ref_idx.get((t, str(r.get("lang", ""))))
        if rr is None:
            continue
        for col in ("mean_tokens", "tokens_per_char", "tokens_per_word", "frac_partial_char_tokens"):
            if col not in df.columns:
                continue
            got, want = fnum(r[col]), fnum(rr[col])
            if math.isnan(want):
                continue
            res["n_checked"] += 1
            if math.isnan(got) or abs(got - want) > TOL_TOKENIZER_REL * max(abs(want), 1e-9):
                res["n_bad"] += 1
                if len(bad) < 15:
                    bad.append(f"{t}/{r['lang']} {col}: {got!r} vs reference {want:.4g}")
    lines.append(f"- {res['n_bad']} of {res['n_checked']} values differ from the reference by more than 10 %")
    for b in bad:
        lines.append(f"    - {b}")
    return res


# ============================================================================================ step 6: probe
def expected_probe_ids(overlay: str, student_id: str, lite: bool) -> tuple[set, int, str]:
    """(allowed probe ids, expected count per (model, lang), note). Probe ids are ALWAYS drawn from the full 400-item
    en sample_id list (harness.data.probe_ids), independent of LITE mode; a LITE submission may only contain those
    probe ids that fall inside the LITE subset, so its expected count is |probe ids INTERSECT LITE subset|."""
    from harness.data import load_gmmlu_lite, lite_subset, probe_ids
    items = load_gmmlu_lite("en")
    try:
        ids = list(probe_ids(student_id))                                  # new signature: full list loaded inside
    except TypeError:
        ids = list(probe_ids(student_id, items))                           # older signature (student_id, items)
    allowed = set(ids)
    if lite:
        lite_ids = {it.sample_id for it in lite_subset(items)}
        expected = allowed & lite_ids
        return allowed, len(expected), f"LITE mode: {len(expected)} of the {len(allowed)} probe ids lie inside the LITE subset"
    return allowed, len(allowed), f"full mode: {len(allowed)} probe ids"


def step6_probe(lines, rep: Report, sub: str, overlay: str, reference: str, A, preds: dict, bands: dict,
                student_id: Optional[str], lite: bool = False) -> dict:
    res = {"present": False, "groups": 0, "ids_ok": None, "agreement": {}, "fabrication_flags": []}
    p = os.path.join(sub, "probe.jsonl")
    if not os.path.exists(p):
        lines.append("- **probe.jsonl missing** (the reflection cannot be checked against it)")
        return res
    from harness.schema import read_jsonl
    rows = read_jsonl(p)
    res["present"] = True
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r.get("model", "?"), r.get("lang", "?")), []).append(r)
    res["groups"] = len(groups)
    lines.append(f"- probe.jsonl: {len(rows)} records in {len(groups)} (model, lang) groups")
    allowed, n_expected = None, 20
    if student_id:
        try:
            allowed, n_expected, note = expected_probe_ids(overlay, student_id, lite)
            lines.append(f"- expected probe ids: {note} (drawn from the full 400-item en list; must be a subset)")
            en_run = run_index(A, preds).get((PRIMARY, "en", "LETTER", "v1_en"))
            full_n = 200 if lite else 400
            if en_run is not None and len(en_run) < full_n:      # DEBUG submissions made with --n N
                ids_present = {r["sample_id"] for r in en_run}
                n_expected = len(allowed & ids_present)
                lines.append(f"- NOTE: en run has only {len(en_run)} items (--n debug run?); expected count reduced to "
                             f"{n_expected} (probe ids present in that run)")
        except Exception as e:  # noqa: BLE001
            lines.append(f"- could not compute the expected probe ids ({type(e).__name__}: {e})")
    else:
        lines.append("- no student_id (smoke_ok.json missing): expected probe ids not computed")
    all_ok = True
    for (model, lang), rs in sorted(groups.items()):
        ids = {r["sample_id"] for r in rs}
        msg = f"- {model} / {lang}: {len(rs)} records"
        if allowed is not None:
            extra = ids - allowed
            ok = not extra and len(ids) == n_expected and len(ids) == len(rs)
            all_ok &= ok
            if ok:
                msg += f" — ids are probe_ids(student_id) ({len(ids)}/{n_expected})"
            else:
                why = []
                if extra:
                    why.append(f"{len(extra)} id(s) not in probe_ids(student_id), e.g. {sorted(extra)[:3]}")
                if len(ids) != n_expected:
                    why.append(f"{len(ids)} distinct ids, expected {n_expected}")
                if len(ids) != len(rs):
                    why.append(f"{len(rs) - len(ids)} duplicate record(s)")
                msg += " — **ids DO NOT match probe_ids** (" + "; ".join(why) + ")"
        elif len(rs) != n_expected:
            all_ok = False
        lines.append(msg)
    res["ids_ok"] = all_ok if allowed is not None else None
    if allowed is not None and not all_ok:
        rep.flag("probe.jsonl ids do not match probe_ids(student_id) for at least one (model, lang)")
    # consistency: probe rows must equal the corresponding prediction records
    idx = run_index(A, preds)
    hub_to_alias = {hub: alias for alias, (hub, _) in A.MODELS.items()}
    n_mismatch = 0
    for (model, lang), rs in groups.items():
        run = idx.get((hub_to_alias.get(model, model), lang, "LETTER", "v1_en"))
        if run is None:
            continue
        by_id = {r["sample_id"]: r for r in run}
        for r in rs:
            pr = by_id.get(r["sample_id"])
            if pr is None or pr["pred"] != r.get("pred") or any(abs(a - b) > 1e-6 for a, b in zip(pr["scores"], r.get("scores") or [])):
                n_mismatch += 1
    if n_mismatch:
        lines.append(f"- **{n_mismatch} probe record(s) disagree with the submitted prediction files** (edited probe or predictions?)")
        rep.flag(f"probe.jsonl disagrees with predictions/ on {n_mismatch} record(s)")
    else:
        lines.append("- probe records agree with the submitted prediction files")
    lines.append("- for the reflection check: the 20 probe sample_ids per language are listed in probe.jsonl; three must be cited with matching numbers")
    # reference agreement
    ref_dir = os.path.join(reference, "predictions")
    if not os.path.isdir(ref_dir):
        lines.append("- reference predictions/ not found: agreement / correlation check skipped")
        return res
    ref_preds = A.load_predictions(ref_dir)
    ref_idx = run_index(A, ref_preds)
    for (alias, lang, scorer, variant), rows in sorted(idx.items()):
        if scorer != "LETTER" or variant != "v1_en":
            continue
        ref_rows = ref_idx.get((alias, lang, scorer, variant))
        if ref_rows is None:
            continue
        a, b = A.align(rows, ref_rows)
        if len(a) < 10:
            continue
        agree = float(np.mean([x["pred"] == y["pred"] for x, y in zip(a, b)]))
        sa = np.array([x["scores"] for x in a], dtype=float).ravel()
        sb = np.array([y["scores"] for y in b], dtype=float).ravel()
        corr = float(np.corrcoef(sa, sb)[0, 1]) if sa.std() > 0 and sb.std() > 0 else math.nan
        key = f"{alias}|{lang}|{scorer}|{variant}"
        res["agreement"][key] = (agree, corr, len(a))
        acc = acc_of(rows)
        band = band_for(bands, key, lite)
        in_band = band is not None and band["lo"] <= acc <= band["hi"]
        note = ""
        if agree < 0.90:
            note = " — **LOW AGREEMENT with the instructor run**" + (" while accuracy is in band: POSSIBLE FABRICATION" if in_band else "")
            if in_band or lang == "en":
                res["fabrication_flags"].append(key)
        lines.append(f"- {key}: per-item agreement {agree:.3f}, score-vector Pearson r {corr:.3f} (n={len(a)}){note}")
    for k in res["fabrication_flags"]:
        rep.flag(f"{k}: < 90 % agreement with the reference predictions (accuracy plausible) — possible fabrication; inspect")
    return res


# ============================================================================================ step 7: run log
def parse_run_log(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        ls = [ln.rstrip("\n") for ln in f if ln.strip()]
    if not ls:
        return []
    cols = [c.strip() for c in ls[0].split("|")]
    out = []
    for ln in ls[1:]:
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) == len(cols):
            out.append(dict(zip(cols, parts)))
    return out


def step7_run_log(lines, rep: Report, sub: str, A, smoke: dict) -> dict:
    res = {"present": False, "rows": 0, "ips_flags": [], "missing_runs": [], "identical_wall": False}
    rows = parse_run_log(os.path.join(sub, "run_log.txt"))
    if not rows:
        lines.append("- **run_log.txt missing or empty**")
        return res
    res["present"], res["rows"] = True, len(rows)
    lines.append(f"- {len(rows)} run(s) logged; devices: {sorted(set(r.get('device', '?') for r in rows))}; "
                 f"smoke_ok device: {smoke.get('device', '?')}, chip: {smoke.get('chip', '?')}")
    for r in rows:
        dev = r.get("device", "")
        ips = fnum(r.get("items_per_sec"))
        n = fnum(r.get("n_items"))
        lim = IPS_FLAG.get(dev)
        if lim is not None and not math.isnan(ips) and ips > lim and n >= 100:
            res["ips_flags"].append(f"{r.get('model')}/{r.get('lang')}/{r.get('scorer')}/{r.get('variant')}: {ips} items/s on {dev}")
    for f in res["ips_flags"]:
        lines.append(f"- **implausible throughput**: {f}")
        rep.flag("implausible throughput in run_log.txt: " + f)
    if smoke.get("device") and any(r.get("device") not in (smoke.get("device"), "") for r in rows):
        lines.append(f"- note: some runs used a device other than smoke_ok's {smoke.get('device')} (Colab fallback? must be declared in Setup)")
    walls = [r.get("wall_sec") for r in rows if fnum(r.get("n_items")) >= 100]
    if len(walls) >= 5 and len(set(walls)) == 1:
        res["identical_wall"] = True
        rep.flag("run_log.txt: identical wall_sec on every run (fabrication signal)")
    logged = {(r.get("model"), r.get("lang"), r.get("scorer", "").upper(), r.get("variant")) for r in rows}
    hub = {alias: h for alias, (h, _) in A.MODELS.items()}
    expected = [(hub[PRIMARY], l, "LETTER", "v1_en") for l in CORE_LANGS]
    expected += [(hub[PRIMARY], l, "LETTER", f"perm_{o}") for l in PERM_LANGS for o in PERM_ORDERS[1:]]
    expected += [(hub[PRIMARY], l, "LETTER", "v2_native") for l in NATIVE_LANGS]
    expected += [(hub[CONTRAST], l, "LETTER", "v1_en") for l in CORE_LANGS]
    expected += [(hub[PRIMARY], l, s, "v1_en") for l in PART1_LANGS for s in ("CONT", "GEN")]
    res["missing_runs"] = [e for e in expected if e not in logged]
    lines.append(f"- {len(expected) - len(res['missing_runs'])}/{len(expected)} expected core runs (Parts 1/2/3/5) appear in the log")
    for m in res["missing_runs"][:40]:
        lines.append(f"    - not logged: {m[0].split('/')[-1]} {m[1]} {m[2]} {m[3]}")
    return res


# ============================================================================================ step 8: files / integrity
def pdf_pages(path: str) -> Optional[int]:
    try:
        with open(path, "rb") as f:
            data = f.read()
        n = len(re.findall(rb"/Type\s*/Page[^s]", data))
        return n or None
    except Exception:  # noqa: BLE001
        return None


def step8_files(lines, rep: Report, sub: str, package_root: str, A, preds: dict, smoke: dict) -> dict:
    res = {"smoke": bool(smoke), "integrity": False, "readme": False, "report": False, "reflection": False,
           "figures": {}, "frozen_modified": [], "templates_ok": None, "scipy": False, "lite_ok": True,
           "requirements": False, "own_test": None}
    for name, key in [("integrity.txt", "integrity"), ("README.md", "readme"), ("report.pdf", "report"),
                      ("reflection.pdf", "reflection"), ("requirements.txt", "requirements")]:
        ok = os.path.exists(os.path.join(sub, name))
        res[key] = ok
        extra = ""
        if ok and name.endswith(".pdf"):
            n = pdf_pages(os.path.join(sub, name))
            extra = f" (~{n} pages by /Type /Page count; verify by eye)" if n else ""
            if n and name == "report.pdf" and n > 4:
                extra += " **over the 4-page limit?**"
            if n and name == "reflection.pdf" and n > 1:
                extra += " **over the 1-page limit?**"
        lines.append(f"- {name}: {'present' if ok else '**MISSING**'}{extra}")
    if not res["integrity"]:
        rep.flag("integrity.txt missing — report not graded until supplied, no grace window")
    if not smoke:
        rep.flag("smoke_ok.json missing — report not graded until supplied, no grace window")
    else:
        lines.append(f"- smoke_ok.json: student_id={smoke.get('student_id')}, device={smoke.get('device')}, dtype={smoke.get('dtype')}, "
                     f"lite_mode={smoke.get('lite_mode')}, torch={smoke.get('torch')}, model_sha={str(smoke.get('model_sha'))[:8]}")
    for fig in FIGURES:
        ok = os.path.exists(os.path.join(sub, "figures", fig))
        res["figures"][fig] = ok
        lines.append(f"- figures/{fig}: {'present' if ok else '**MISSING**'}")
    # LITE consistency
    lite = bool(smoke.get("lite_mode", False)) if smoke else None
    bad_lite = []
    for fname, rows in preds.items():
        if rows[0].get("benchmark", "global_mmlu_lite") != "global_mmlu_lite":
            continue
        if lite is not None and bool(rows[0].get("lite_mode")) != lite:
            bad_lite.append(fname)
    if bad_lite:
        res["lite_ok"] = False
        lines.append(f"- **LITE inconsistency**: {len(bad_lite)} file(s) have lite_mode != smoke_ok.json ({bad_lite[:5]}...)")
        rep.flag("lite_mode in predictions disagrees with smoke_ok.json")
    elif smoke:
        lines.append(f"- lite_mode consistent across predictions ({lite})")
    sids = {rows[0].get("student_id") for rows in preds.values()}
    if smoke and sids and sids != {smoke.get("student_id")}:
        lines.append(f"- **student_id in predictions {sorted(map(str, sids))} != smoke_ok.json {smoke.get('student_id')}**")
        rep.flag("student_id mismatch between predictions/ and smoke_ok.json")
    # frozen files
    for f in FROZEN_HARNESS_FILES:
        a, b = os.path.join(sub, "harness", f), os.path.join(package_root, "student", "harness", f)
        if os.path.exists(a) and os.path.exists(b):
            if open(a, "rb").read() != open(b, "rb").read():
                res["frozen_modified"].append(f)
        elif not os.path.exists(a):
            res["frozen_modified"].append(f + " (missing)")
    lines.append(f"- frozen harness files modified vs package: {res['frozen_modified'] or 'none'}")
    # templates / MODELS via subprocess in the submission's own harness
    code = ("import json,sys; sys.path.insert(0,'.'); from harness import prompts as P, model as M; "
            "print(json.dumps({'V1_EN':P.V1_EN,'NATIVE':P.NATIVE_TEMPLATES,'PREFIX':P.ASSISTANT_PREFIX,'PERM':P.PERM_ORDERS,"
            "'MODELS':{k:list(v) for k,v in M.MODELS.items()}}))")
    rc, out = run_subprocess([sys.executable, "-c", code], cwd=sub, env_extra={"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": ""}, timeout=120)
    try:
        theirs = json.loads(out.strip().splitlines()[-1])
        from harness import prompts as P   # instructor overlay
        ours = {"V1_EN": P.V1_EN, "NATIVE": P.NATIVE_TEMPLATES, "PREFIX": P.ASSISTANT_PREFIX, "PERM": P.PERM_ORDERS,
                "MODELS": {k: list(v) for k, v in A.MODELS.items()}}
        diff = [k for k in ours if ours[k] != theirs.get(k)]
        res["templates_ok"] = not diff
        lines.append(f"- frozen templates / model pins in the submission's harness: {'unchanged' if not diff else '**ALTERED: ' + ', '.join(diff) + '**'}")
        if diff:
            rep.flag(f"frozen prompt templates or model pins altered in the submission's harness: {diff} (affected part = 0)")
    except Exception:  # noqa: BLE001
        lines.append(f"- could not import the submission's harness.prompts/model (exit {rc}): " + " ".join(out.strip().splitlines()[-2:]))
    # scipy / statsmodels in student-written code
    hits = []
    for f in ("harness/stats.py", "analysis.py", "harness/scorers.py", "tokenizer_stats.py"):
        p = os.path.join(sub, f)
        if os.path.exists(p):
            src = open(p, encoding="utf-8", errors="replace").read()
            if re.search(r"^\s*(import|from)\s+(scipy|statsmodels)", src, flags=re.M):
                hits.append(f)
    res["scipy"] = bool(hits)
    if hits:
        lines.append(f"- **scipy/statsmodels imported in {hits}** (CI computation must be numpy-only: -3 on P2-runs)")
    else:
        lines.append("- no scipy/statsmodels import in stats.py / analysis.py / scorers.py / tokenizer_stats.py")
    # own test + fast pytest on the submission's own code (no model)
    tests_dir = os.path.join(sub, "tests")
    if os.path.isdir(tests_dir):
        pkg_tests = set(os.listdir(os.path.join(package_root, "student", "tests")))
        own = [f for f in os.listdir(tests_dir) if f.endswith(".py") and f not in pkg_tests]
        added_fn = 0
        for f in os.listdir(tests_dir):
            p, q = os.path.join(tests_dir, f), os.path.join(package_root, "student", "tests", f)
            if f.endswith(".py") and os.path.exists(q):
                added_fn += max(0, len(re.findall(r"^def test_", open(p, encoding="utf-8", errors="replace").read(), flags=re.M))
                                - len(re.findall(r"^def test_", open(q, encoding="utf-8", errors="replace").read(), flags=re.M)))
        res["own_test"] = bool(own) or added_fn > 0
        lines.append(f"- own tests: new test files {own or 'none'}, {added_fn} test function(s) added to instructor files")
        rc, out = run_subprocess([sys.executable, "-m", "pytest", "-q", "tests", "--no-model", "-p", "no:cacheprovider"],
                                 cwd=sub, env_extra={"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": ""}, timeout=600)
        summary = [ln for ln in out.splitlines() if re.search(r"\d+ (passed|failed|error)", ln)]
        res["pytest_rc"] = rc
        m = re.search(r"(\d+) failed", out)
        res["pytest_failed"] = int(m.group(1)) if m else (0 if rc == 0 else None)
        m = re.search(r"(\d+) passed", out)
        res["pytest_passed"] = int(m.group(1)) if m else 0
        res["parse_gen_failed"] = len(re.findall(r"FAILED .*test_parse_gen", out))
        lines.append(f"- `pytest -q tests --no-model` on the submission's own code: exit {rc}; {summary[-1] if summary else out.strip().splitlines()[-1:]}")
        if rc != 0:
            lines += tail("\n".join(ln for ln in out.splitlines() if ln.startswith("FAILED") or ln.startswith("ERROR")), 20)
    else:
        lines.append("- tests/ directory missing")
    return res


# ============================================================================================ step 9: rerun
def step9_rerun(lines, rep: Report, sub: str, tmp: str, A, preds: dict, smoke: dict, device: str, n: Optional[int]) -> dict:
    res = {"ran": False, "agreement": None, "acc_match": None}
    sid = smoke.get("student_id") or (next(iter(preds.values()))[0].get("student_id") if preds else "UNKNOWN")
    work = os.path.join(tmp, "rerun")
    shutil.copytree(sub, work, ignore=shutil.ignore_patterns("predictions", "figures", "*.pdf", "__pycache__", "*.partial"))
    os.makedirs(os.path.join(work, "predictions"), exist_ok=True)
    cmd = [sys.executable, "-m", "harness", "run", "--part", "2", "--model", PRIMARY, "--lang", "en", "--scorer", "LETTER",
           "--variant", "v1_en", "--student-id", str(sid), "--device", device, "--batch-size", "8"]
    if n:
        cmd += ["--n", str(n)]
        lines.append(f"- DEBUG: re-run truncated to --n {n}")
    lines.append(f"- command: `{' '.join(cmd[1:])}` (cwd = temp copy of the submission, its own harness)")
    env = {"PYTHONPATH": ""}
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    t0 = time.time()
    rc, out = run_subprocess(cmd, cwd=work, env_extra=env, timeout=3600)
    lines.append(f"- exit {rc} after {time.time() - t0:.0f}s")
    if rc != 0:
        rep.flag("Part 2 en re-run of the submission's harness FAILED (command does not run without edits)")
        lines += tail(out, 25)
        return res
    res["ran"] = True
    new = A.load_predictions(os.path.join(work, "predictions"))
    new_rows = A.select(new, PRIMARY, "en", "LETTER", "v1_en")
    old_rows = A.select(preds, PRIMARY, "en", "LETTER", "v1_en", part="2")
    if new_rows is None or old_rows is None:
        lines.append("- could not pair the re-run output with a submitted en LETTER v1_en file")
        return res
    a, b = A.align(old_rows, new_rows)
    agree = float(np.mean([x["pred"] == y["pred"] for x, y in zip(a, b)]))
    acc_old, acc_new = acc_of(a), acc_of(b)
    res["agreement"], res["acc_match"] = agree, round(acc_old, 3) == round(acc_new, 3)
    res["n"] = len(a)
    lines.append(f"- per-item agreement {agree:.4f} on {len(a)} items; accuracy submitted {acc_old:.3f} vs re-run {acc_new:.3f} "
                 f"({'identical to 3 decimals' if res['acc_match'] else '**DIFFERENT**'})")
    if agree < 0.99:
        rep.flag(f"re-run agreement {agree:.3f} < 0.99 with the submitted en predictions")
    return res


# ============================================================================================ rubric scoring
def fill_rubric(rep: Report, A, preds: dict, s1: dict, s2: dict, s3: dict, s4: dict, s5: dict, s6: dict, s7: dict,
                s8: dict, s9: Optional[dict], bands: dict) -> None:
    idx = run_index(A, preds)
    inb = s3.get("in_band", {}) if s3 else {}

    def band_ok(alias, lang, scorer, variant="v1_en") -> Optional[bool]:
        k = f"{alias}|{lang}|{scorer}|{variant}"
        if (alias, lang, scorer, variant) not in idx:
            return None
        return inb.get(k, True)   # no band -> present counts as OK

    diffs = (s2 or {}).get("diffs", {})

    def csv_ok(name) -> bool:
        d = diffs.get(name)
        return bool(d and d["present"] and not d["missing_rows"] and not d["mismatch"] and not d["missing_cols"])

    def csv_present(name) -> bool:
        return bool(diffs.get(name, {}).get("present"))

    # ---- P1-code (8): LETTER en 3, CONT en 3, GEN parser tests 2 ------------------------------------------------
    pts, notes = 0.0, []
    ok = band_ok(PRIMARY, "en", "LETTER")
    if ok is None:
        notes.append("no en LETTER run (0/3)")
    elif ok:
        pts += 3
    else:
        pts += 2; notes.append("LETTER en out of band (-1 if cause documented, else more)")
    cont_present = [s for s in ("CONT", "CONT_TOKNORM", "CONT_CHARNORM") if (PRIMARY, "en", s, "v1_en") in idx]
    if len(cont_present) == 3:
        ok = band_ok(PRIMARY, "en", "CONT")
        pts += 3 if ok else 2
        if not ok:
            notes.append("CONT en out of band")
    else:
        pts += max(0, len(cont_present) - 0) * 1.0 if cont_present else 0
        notes.append(f"CONT variants present: {cont_present or 'none'} (-1 per missing)")
    pg_failed = (s8 or {}).get("parse_gen_failed")
    gen_present = (PRIMARY, "en", "GEN", "v1_en") in idx
    if s8 and s8.get("pytest_rc") is not None:
        if pg_failed == 0 and gen_present:
            pts += 2
        elif gen_present:
            pts += 1; notes.append(f"{pg_failed} parse_gen test(s) failed")
        else:
            notes.append("no GEN run")
        other_failed = max(0, (s8.get("pytest_failed") or 0) - (pg_failed or 0))
        if other_failed:
            pts -= other_failed
            notes.append(f"{other_failed} other instructor test(s) failed (-1 each; model tests not run here)")
        if s8.get("own_test") is False:
            notes.append("no own test found")
    else:
        pts += 1 if gen_present else 0
        notes.append("pytest not run; GEN parser tests unscored (manual)")
    rep.score("P1-code", min(pts, 8), 8, "; ".join(notes) + "; README letter-token note: manual")
    # ---- P1-analysis (7): manual; note table completeness -------------------------------------------------------
    have = sum(1 for l in PART1_LANGS for s in PART1_SCORERS if (PRIMARY, l, s, "v1_en") in idx)
    rep.score("P1-analysis", None, 7, f"{have}/15 Part 1 (scorer, lang) runs present; part1_scorers.csv "
              f"{'recomputes' if csv_ok('part1_scorers.csv') else 'absent/mismatch (optional file)'}; 8 cases + kappa + recommendation: manual")
    # ---- P2-runs (8) ----------------------------------------------------------------------------------------------
    langs = [l for l in CORE_LANGS if (PRIMARY, l, "LETTER", "v1_en") in idx]
    pts, notes = 8.0, []
    miss = 8 - len(langs)
    if miss:
        pts -= min(4, miss); notes.append(f"{miss} language(s) missing")
    if not langs:
        pts = 0
    if s1 and s1.get("validate_rc", 1) != 0:
        pts -= 1; notes.append("validator fails")
    d = diffs.get("results.csv", {})
    if not d.get("present"):
        pts -= 3; notes.append("results.csv missing")
    else:
        non_ci = d["mismatch"] - d["ci_mismatch"] + d["missing_rows"]
        if non_ci:
            pts -= 1; notes.append(f"{non_ci} accuracy/n mismatches or missing rows")
        if d["ci_mismatch"]:
            pts -= 2; notes.append(f"{d['ci_mismatch']} CI values off by > 0.01")
    if s8 and s8.get("scipy"):
        pts -= 3; notes.append("scipy/statsmodels used")
    rep.score("P2-runs", max(0, pts), 8, "; ".join(notes) or "8 langs, validator OK, results.csv recomputes")
    fig = (s8 or {}).get("figures", {})
    rep.score("P2-figure", None if fig.get(FIGURES[0]) else 0, 2, "present: CI bars / chance line / order / axes = manual" if fig.get(FIGURES[0]) else "missing")
    rep.score("P2-stats", None, 5, "manual (paired-CI list, sqrt(p(1-p)/n) with +-5/+-7, CS-CA unpaired + detectability)")
    pts = 0.0
    a, b = csv_ok("paired_vs_en.csv"), csv_ok("cs_ca_gap.csv")
    if a and b:
        pts = 3
    elif a or csv_present("paired_vs_en.csv"):
        pts = 2 if a else 1
    rep.score("P2-parallel", pts, 5, f"auto part (3): paired_vs_en {'OK' if a else 'absent/mismatch'}, cs_ca_gap {'OK' if b else 'absent/mismatch'}; interpretation (2) manual")
    # ---- P3 -------------------------------------------------------------------------------------------------------
    pts, notes = 7.0, []
    if s4:
        n_missing_lang = len(s4["langs_missing"])
        pts -= 1.5 * n_missing_lang
        if n_missing_lang:
            notes.append(f"incomplete permutations for {s4['langs_missing']}")
        if s4["n_native_runs"] < 3:
            pts -= 2 if s4["n_native_runs"] == 0 else (3 - s4["n_native_runs"]) * 0.7
            notes.append(f"{s4['n_native_runs']}/3 native runs")
        if s4["langs_bad"]:
            pts -= 1.5; notes.append(f"perm mean off reference for {s4['langs_bad']}")
        if s4["n_perm_runs"] == 0:
            pts = 0; notes.append("no permutation runs")
    if s1 and s1.get("remap_flags"):
        pts = min(pts, 3); notes.append("GOLD-NOT-REMAPPED flag (2-3 total)")
    if not csv_ok("permutations.csv"):
        pts -= 1; notes.append("permutations.csv absent/mismatch")
    if not csv_ok("native_instruction.csv"):
        pts -= 1; notes.append("native_instruction.csv absent/mismatch")
    rep.score("P3-runs", max(0, pts), 7, "; ".join(notes) or "12 perm + 3 native runs, remapped, CSVs recompute")
    rep.score("P3-figure", None if fig.get(FIGURES[1]) else 0, 3,
              ("present; consistency_rate in permutations.csv: " + ("yes" if csv_present("permutations.csv") else "no") + "; pooled/0.25 line = manual") if fig.get(FIGURES[1]) else "missing")
    rep.score("P3-analysis", None, 5, "manual (std vs CI half-width, >35 % letter, native effect + sw caveat)")
    # ---- P4 -------------------------------------------------------------------------------------------------------
    pts, notes = 4.0, []
    if not s5 or not s5["present"]:
        pts = 0; notes.append("tokenizer.csv missing")
    else:
        nt = s5["n_tokenizers"]
        if nt < 3:
            pts -= (3 - nt); notes.append(f"{nt} tokenizer(s)")
        if s5["ref"] and s5["n_bad"]:
            pts -= 1; notes.append(f"{s5['n_bad']} values off > 10 %")
        if s5["zh_nan"] is False:
            pts -= 0.5; notes.append("zh tokens_per_word not NaN")
        if s5["n_rows"] < 8 * max(nt, 1):
            notes.append("some languages missing")
    rep.score("P4-csv", max(0, pts), 4, "; ".join(notes) + ("; zh NaN reason: manual" if pts > 0 else ""))
    rep.score("P4-figure", None if fig.get(FIGURES[2]) else 0, 1, "present; labels / no r = manual" if fig.get(FIGURES[2]) else "missing")
    rep.score("P4-analysis", None, 5, "manual (BPE mechanism, yo diacritics, cost estimate, id vs sw)")
    # ---- P5 -------------------------------------------------------------------------------------------------------
    q3 = [l for l in CORE_LANGS if (CONTRAST, l, "LETTER", "v1_en") in idx]
    pts, notes = 5.0, []
    pts -= (8 - len(q3))
    if len(q3) < 8:
        notes.append(f"{8 - len(q3)} language(s) missing")
    ok = band_ok(CONTRAST, "en", "LETTER")
    if ok is False:
        pts -= 1; notes.append("qwen3 en out of band")
    proto_bad = []
    for l in q3:
        p2 = idx.get((PRIMARY, l, "LETTER", "v1_en"))
        p5 = idx[(CONTRAST, l, "LETTER", "v1_en")]
        if p2 is not None and [r["sample_id"] for r in p2] != [r["sample_id"] for r in p5]:
            proto_bad.append(l)
        if p5 and str(p5[0].get("part")) != "5":
            notes.append(f"{l} qwen3 file has part={p5[0].get('part')}")
    if proto_bad:
        pts -= 2; notes.append(f"item set differs from Part 2 for {proto_bad}")
    rep.score("P5-runs", max(0, pts), 5, "; ".join(notes) or "8 langs, en in band, same items as Part 2")
    d = diffs.get("comparison.csv", {})
    pts, notes = 0.0, []
    if d.get("present"):
        ci_ok = d["ci_mismatch"] == 0 and d["missing_rows"] == 0
        cnt_ok = (d["mismatch"] - d["ci_mismatch"]) == 0 and d["missing_rows"] == 0
        pts += 2 if ci_ok else 1
        pts += 1 if cnt_ok else 0
        pts += 1 if cnt_ok else 0   # mdd value correct (1 of the 2); derivation in report is manual
        notes.append(f"paired CI {'OK' if ci_ok else 'mismatch'}; counts/mdd {'OK' if cnt_ok else 'mismatch'}; +1 manual for the MDD derivation in the report")
    else:
        notes.append("comparison.csv missing")
    rep.score("P5-stats", pts, 5, "; ".join(notes))
    rep.score("P5-analysis", None, 5, "manual (paired vs unpaired with a language, chance cells via CI, confound + control)")
    # ---- P6 -------------------------------------------------------------------------------------------------------
    rep.score("P6-report", None if (s8 or {}).get("report") else 0, 15, "manual" if (s8 or {}).get("report") else "report.pdf missing")
    rep.score("P6-reflection", None if (s8 or {}).get("reflection") else 0, 5,
              ("manual; probe ids " + {True: "verified", False: "MISMATCH", None: "unverified"}[(s6 or {}).get("ids_ok")]) if (s8 or {}).get("reflection") else "reflection.pdf missing")
    pts, notes = 0.0, []
    if s9 is None:
        notes.append("re-run not performed (--rerun-en): 3 pts manual")
        rerun_pts = None
    elif not s9.get("ran"):
        rerun_pts = 0; notes.append("re-run failed")
    elif s9.get("agreement", 0) >= 0.99 and s9.get("acc_match"):
        rerun_pts = 3
    elif s9.get("acc_match"):
        rerun_pts = 2; notes.append(f"agreement {s9['agreement']:.3f} < 0.99")
    else:
        rerun_pts = 1; notes.append("accuracy differs on re-run")
    other = 2.0
    if s1 and (s1.get("validate_rc", 1) != 0 or s1.get("check_rc", 1) != 0):
        other -= 1; notes.append("a validator fails")
    if s7 and (not s7["present"] or s7["missing_runs"]):
        other -= 0.5; notes.append(f"run_log incomplete ({len(s7.get('missing_runs', []))} core runs not logged)" if s7["present"] else "run_log missing")
    if not (s8 or {}).get("requirements"):
        other -= 0.5; notes.append("requirements.txt missing")
    if not (s8 or {}).get("readme"):
        notes.append("README missing (one command per part: manual)")
    total = None if rerun_pts is None else rerun_pts + max(0, other)
    rep.score("P6-repro", total, 5, "; ".join(notes) + (f"; auto part without re-run = {max(0, other):g}/2" if rerun_pts is None else ""))
    rep.score("Bonus", None, 10, "manual; stretch files present: " + (", ".join(sorted(f for f in preds if os.path.basename(f).startswith("pS"))) or "none"))


# ============================================================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission")
    ap.add_argument("--package-root", default=DEFAULT_PACKAGE_ROOT)
    ap.add_argument("--reference", default=None, help="default: <package-root>/instructor/reference_results")
    ap.add_argument("--rerun-en", action="store_true")
    ap.add_argument("--rerun-device", default="auto")
    ap.add_argument("--rerun-n", type=int, default=None, help="DEBUG ONLY: truncate the re-run")
    ap.add_argument("--n-expected", type=int, default=None, help="DEBUG ONLY: validator line-count override")
    ap.add_argument("--out", default=None, help="Markdown report path (default: <submission>/grade_report.md)")
    ap.add_argument("--keep-tmp", action="store_true")
    args = ap.parse_args(argv)

    sub = os.path.abspath(args.submission)
    package_root = os.path.abspath(args.package_root)
    reference = os.path.abspath(args.reference or os.path.join(package_root, "instructor", "reference_results"))
    out_path = args.out or os.path.join(sub, "grade_report.md")
    rep = Report()
    tmp = tempfile.mkdtemp(prefix="grade_", dir=os.environ.get("GRADE_TMPDIR"))
    t0 = time.time()
    try:
        overlay = build_overlay(package_root, tmp)
        smoke = {}
        try:
            if os.path.exists(os.path.join(sub, "smoke_ok.json")):
                smoke = json.load(open(os.path.join(sub, "smoke_ok.json")))
        except Exception as e:  # noqa: BLE001
            rep.errors.append(f"smoke_ok.json unreadable: {e}")
        bands = {}
        try:
            bands = load_bands(reference)
        except Exception as e:  # noqa: BLE001
            rep.errors.append(f"bands.json unreadable: {e}")
        info = rep.section("0. Inputs")
        info += [f"- submission: `{sub}`", f"- package root: `{package_root}`", f"- reference dir: `{reference}` "
                 f"(bands: {len(bands)} keys; tokenizer.csv: {os.path.exists(os.path.join(reference, 'tokenizer.csv'))}; "
                 f"predictions/: {os.path.isdir(os.path.join(reference, 'predictions'))})",
                 f"- student_id (smoke_ok.json): {smoke.get('student_id', '?')}"]

        s1 = step(rep, "1. Validators (instructor copies)")(step1_validators)(rep, sub, overlay, args.n_expected) or {}
        s2 = step(rep, "2. CSV recomputation (instructor analysis.py, B=2000, seed=0)")(step2_recompute)(rep, sub, overlay, tmp) or {}
        A, preds = s2.get("A"), s2.get("preds", {})
        if A is None:
            sys.path.insert(0, overlay)
            A = import_from(os.path.join(overlay, "analysis.py"), "instructor_analysis")
            preds = {}
        lite = bool(smoke.get("lite_mode", False))
        if not smoke and preds:                                   # no smoke_ok.json: fall back to the records
            lite = all(bool(rows[0].get("lite_mode")) for rows in preds.values()
                       if rows[0].get("benchmark", "global_mmlu_lite") == "global_mmlu_lite")
        info.append(f"- lite_mode: {lite} ({'smoke_ok.json' if smoke else 'inferred from the prediction records'})")
        s3 = step(rep, "3. Plausibility bands")(step3_bands)(rep, A, preds, bands, lite) or {}
        s4 = step(rep, "4. Permutations: gold remap and mean vs reference")(step4_permutations)(rep, A, preds, bands, s1.get("remap_flags", []), lite) or {}
        s5 = step(rep, "5. tokenizer.csv vs reference (10 %)")(step5_tokenizer)(rep, sub, reference) or {}
        s6 = step(rep, "6. probe.jsonl and agreement with the instructor run")(step6_probe)(rep, sub, overlay, reference, A, preds, bands, smoke.get("student_id"), lite) or {}
        s7 = step(rep, "7. run_log.txt sanity")(step7_run_log)(rep, sub, A, smoke) or {}
        s8 = step(rep, "8. Required files, LITE consistency, frozen code, tests")(step8_files)(rep, sub, package_root, A, preds, smoke) or {}
        s9 = None
        if args.rerun_en:
            s9 = step(rep, "9. Re-run of the submission's Part 2 en command")(step9_rerun)(rep, sub, tmp, A, preds, smoke, args.rerun_device, args.rerun_n) or {"ran": False}
        else:
            rep.section("9. Re-run of the submission's Part 2 en command").append("- skipped (pass --rerun-en)")
        try:
            fill_rubric(rep, A, preds, s1, s2, s3, s4, s5, s6, s7, s8, s9, bands)
        except Exception as e:  # noqa: BLE001
            rep.errors.append(f"rubric scoring: {type(e).__name__}: {e}")
            for line, mx in RUBRIC_MAX.items():
                rep.scores.setdefault(line, (None, mx, "scoring failed; manual"))
        rep.section("Timing").append(f"- grader wall-clock: {time.time() - t0:.0f}s")
        text = rep.render(sub)
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:  # noqa: BLE001
            print(f"[grade] could not write {out_path}: {e}", file=sys.stderr)
        print(text)
        print(f"\n[grade] report written to {out_path}", file=sys.stderr)
        if args.keep_tmp:
            print(f"[grade] temp dir kept: {tmp}", file=sys.stderr)
    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
