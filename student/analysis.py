"""analysis.py — regenerate EVERY CSV and figure from predictions/ with NO model loaded (Parts 1-5).

Usage (from this directory):
    python analysis.py [--predictions predictions] [--out .] [--figures figures] [--B 2000] [--seed 0]

The argument parsing, the output file names and the CSV column lists below are FIXED (the grader recomputes
every table from your predictions/ with the same harness.stats functions and diffs it against yours, so the
schema must match exactly). Fill in the function bodies. Rules:
  * use harness.stats for every CI / count (B=2000, seed=0 by default) — no scipy/statsmodels;
  * align paired comparisons on sample_id (never on row position);
  * skip a table gracefully (print a note) when its inputs are missing — never crash on partial data;
  * figures use the matplotlib "Agg" backend (no display needed).

Outputs
    results.csv              one row per (part, model, lang, scorer, prompt_variant, subset in all/CS/CA)
                             columns = harness.schema.RESULTS_COLUMNS; parse_fail_rate only for GEN;
                             wall_clock_sec from run_log.txt (matched on model + lang + LETTER/CONT/GEN + variant)
    paired_vs_en.csv         Part 2 (qwen2.5 LETTER v1_en): every lang != en vs en, paired
    permutations.csv         Part 3: acc per permutation, mean, std (ddof=0), consistency rate, pooled letter fractions
                             (the Part 2 v1_en LETTER run stands in for perm_ABCD when that file is absent)
    native_instruction.csv   Part 3: acc(v2_native) - acc(v1_en), paired, for zh / hi / sw
    comparison.csv           Part 5: qwen2.5 vs qwen3, diff = acc(qwen3) - acc(qwen2.5) (second minus first, like every
                             other diff column), paired CI, naive unpaired CI overlap, discordant counts, MDD
    part1_scorers.csv        Part 1: scorer x lang (en/zh/sw) accuracy, parse-fail rate, kappa(LETTER, GEN)
    part1_disagreements.jsonl  Part 1: en items where LETTER and CONT disagree (raw material for your 8 cases)
    cs_ca_gap.csv            Part 2: CS - CA gap per language with the UNPAIRED bootstrap CI
    figures/accuracy_by_language.png   points + CI bars, CORE_LANGS order, dashed line at 0.25, labelled axes
    figures/position_bias.png          grouped bars: fraction of predictions per letter, pooled over permutations
    figures/fertility_vs_accuracy.png  x = mean_tokens (qwen2.5 tokenizer, from tokenizer.csv), y = accuracy,
                                       8 labelled points, NO correlation coefficient
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # import harness from this directory

from harness import stats as ST                                    # noqa: E402
from harness.data import CORE_LANGS, LETTERS                       # noqa: E402
from harness.model import MODELS, PRIMARY_MODEL, CONTRAST_MODEL    # noqa: E402
from harness.prompts import PERM_ORDERS                            # noqa: E402
from harness.schema import RESULTS_COLUMNS, read_jsonl             # noqa: E402

# ---- FIXED column lists — do not change ---------------------------------------------------------------------------
PAIRED_VS_EN_COLUMNS = ["lang", "n", "acc_en", "acc_lang", "diff", "paired_ci_low", "paired_ci_high",
                        "n_en_only_correct", "n_lang_only_correct", "agreement_rate"]
PERMUTATIONS_COLUMNS = ["lang", "acc_ABCD", "acc_BCDA", "acc_CDAB", "acc_DABC", "mean", "std", "consistency_rate",
                        "frac_pred_A", "frac_pred_B", "frac_pred_C", "frac_pred_D"]
NATIVE_COLUMNS = ["lang", "acc_v1_en", "acc_v2_native", "diff", "paired_ci_low", "paired_ci_high",
                  "n_v1_only_correct", "n_v2_only_correct"]
COMPARISON_COLUMNS = ["lang", "n", "acc_qwen25", "acc_qwen3", "diff", "paired_ci_low", "paired_ci_high",
                      "unpaired_overlap", "n_qwen25_only_correct", "n_qwen3_only_correct", "mdd_95"]   # diff = acc_qwen3 - acc_qwen25
PART1_COLUMNS = ["scorer", "lang", "n", "accuracy", "ci_low", "ci_high", "parse_fail_rate", "kappa_letter_gen"]
CS_CA_COLUMNS = ["lang", "n_cs", "n_ca", "acc_cs", "acc_ca", "gap", "unpaired_ci_low", "unpaired_ci_high", "detectable"]
PART1_LANGS = ["en", "zh", "sw"]
PART1_SCORERS = ["LETTER", "CONT", "CONT_TOKNORM", "CONT_CHARNORM", "GEN"]
NATIVE_LANGS = ["zh", "hi", "sw"]
PERM_LANGS = ["en", "zh", "hi", "sw"]

FIG_ACCURACY = "accuracy_by_language.png"
FIG_POSITION = "position_bias.png"
FIG_FERTILITY = "fertility_vs_accuracy.png"

HUB_TO_ALIAS = {hub: alias for alias, (hub, _) in MODELS.items()}


# ---- loading and selection (given) --------------------------------------------------------------------------------
def load_predictions(pred_dir: str) -> dict[str, list[dict]]:
    """{basename: rows} for every predictions/*.jsonl, rows sorted by sample_id."""
    out = {}
    for path in sorted(glob.glob(os.path.join(pred_dir, "*.jsonl"))):
        rows = read_jsonl(path)
        if rows:
            rows.sort(key=lambda r: r["sample_id"])
            out[os.path.basename(path)] = rows
    return out


def run_key(rows: list[dict]) -> tuple[str, str, str, str, str]:
    """(part, model_alias, lang, scorer, prompt_variant) of a prediction file, read from its records."""
    r = rows[0]
    return str(r["part"]), HUB_TO_ALIAS.get(r["model"], r["model"]), r["lang"], r["scorer"], r["prompt_variant"]


def select(preds: dict[str, list[dict]], model: str, lang: str, scorer: str, variant: str,
           part: Optional[str] = None) -> Optional[list[dict]]:
    """Rows of the run (model alias, lang, scorer, variant), or None. `part` is preferred, not required:
    a Part 1 `en` LETTER v1_en run IS the Part 2 run, so reuse it rather than re-running."""
    cands = [(run_key(rows), rows) for rows in preds.values()]
    cands = [(k, rows) for k, rows in cands if k[1:] == (model, lang, scorer, variant)]
    if not cands:
        return None
    if part is not None:
        for k, rows in cands:
            if k[0] == str(part):
                return rows
    return cands[0][1]


def align(*runs: list[dict]) -> list[list[dict]]:
    """Restrict several runs to their common sample_ids, returned in the same sorted order (for paired stats)."""
    common = {r["sample_id"] for r in runs[0]}
    for rows in runs[1:]:
        common &= {r["sample_id"] for r in rows}
    ids = sorted(common)
    return [[{r["sample_id"]: r for r in rows}[i] for i in ids] for rows in runs]


def correct(rows: list[dict]) -> np.ndarray:
    return np.array([r["pred"] == r["gold"] for r in rows], dtype=bool)


def load_run_log(path: str) -> dict[tuple[str, str, str, str], float]:
    """{(model_hub, lang, scorer in LETTER/CONT/GEN, variant): wall_sec} parsed from run_log.txt (last line wins)."""
    raise NotImplementedError("TODO")


# ---- tables -------------------------------------------------------------------------------------------------------
def build_results(preds: dict[str, list[dict]], B: int = 2000, seed: int = 0,
                  run_log: Optional[dict] = None) -> pd.DataFrame:
    """results.csv (columns RESULTS_COLUMNS). One row per file x subset in {all, CS, CA}; CI = ST.bootstrap_ci."""
    raise NotImplementedError("TODO (Part 2)")


def build_paired_vs_en(preds: dict[str, list[dict]], B: int = 2000, seed: int = 0, model: str = PRIMARY_MODEL) -> pd.DataFrame:
    """paired_vs_en.csv: diff = acc(lang) - acc(en), ST.paired_bootstrap_ci on sample_id-aligned items,
    ST.discordant_counts, ST.agreement_rate."""
    raise NotImplementedError("TODO (Part 2)")


def predicted_text_id(pred: Optional[str], order: str) -> Optional[int]:
    """Index in the ORIGINAL option order of the option the model chose under permutation `order`
    (needed for the consistency rate: same predicted option TEXT under all four permutations)."""
    raise NotImplementedError("TODO (Part 3)")


def build_permutations(preds: dict[str, list[dict]], model: str = PRIMARY_MODEL, langs=PERM_LANGS) -> pd.DataFrame:
    """permutations.csv: std uses ddof=0; frac_pred_* are pooled over the four permutation files."""
    raise NotImplementedError("TODO (Part 3)")


def build_native_instruction(preds: dict[str, list[dict]], B: int = 2000, seed: int = 0,
                             model: str = PRIMARY_MODEL, langs=NATIVE_LANGS) -> pd.DataFrame:
    """native_instruction.csv: diff = acc(v2_native) - acc(v1_en), paired."""
    raise NotImplementedError("TODO (Part 3)")


def build_comparison(preds: dict[str, list[dict]], B: int = 2000, seed: int = 0,
                     model_a: str = PRIMARY_MODEL, model_b: str = CONTRAST_MODEL) -> pd.DataFrame:
    """comparison.csv: diff = acc(qwen3) - acc(qwen2.5) (model_b minus model_a — second minus first, like every other
    diff column); n_qwen25_only_correct / n_qwen3_only_correct = discordant counts; unpaired_overlap = whether the two independent 95% CIs
    (ST.bootstrap_ci on each model) overlap; mdd_95 = ST.mdd_95(b, c, n)."""
    raise NotImplementedError("TODO (Part 5)")


def build_part1_scorers(preds: dict[str, list[dict]], B: int = 2000, seed: int = 0,
                        model: str = PRIMARY_MODEL, langs=PART1_LANGS) -> pd.DataFrame:
    """part1_scorers.csv: rows scorer x lang; parse_fail_rate only for GEN; kappa_letter_gen = ST.cohens_kappa
    between LETTER and GEN predictions of the language (GEN None is its own class)."""
    raise NotImplementedError("TODO (Part 1)")


def build_part1_disagreements(preds: dict[str, list[dict]], model: str = PRIMARY_MODEL, lang: str = "en",
                              items: Optional[list] = None) -> list[dict]:
    """en items where LETTER and CONT predictions differ: sample_id, gold, pred_letter, pred_cont, both score
    vectors and (if `items` from harness.data.load_gmmlu_lite is given) the option texts and their char lengths."""
    raise NotImplementedError("TODO (Part 1)")


def build_cs_ca_gap(preds: dict[str, list[dict]], B: int = 2000, seed: int = 0, model: str = PRIMARY_MODEL) -> pd.DataFrame:
    """cs_ca_gap.csv: gap = acc(CS) - acc(CA) with ST.unpaired_bootstrap_ci (disjoint item sets!);
    detectable = CI excludes 0."""
    raise NotImplementedError("TODO (Part 2)")


# ---- figures ------------------------------------------------------------------------------------------------------
def fig_accuracy_by_language(results: pd.DataFrame, path: str) -> bool:
    """Points with 95% CI error bars in CORE_LANGS order, dashed line at 0.25, labelled axes, for qwen2.5 LETTER
    v1_en; add qwen3 as a second series when present. Return True if written."""
    raise NotImplementedError("TODO (Part 2)")


def fig_position_bias(perms: pd.DataFrame, path: str) -> bool:
    """Grouped bars per language of frac_pred_A..D (pooled over permutations) with a line at 0.25."""
    raise NotImplementedError("TODO (Part 3)")


def fig_fertility_vs_accuracy(tokenizer_df: pd.DataFrame, results: pd.DataFrame, path: str) -> bool:
    """x = mean_tokens of the qwen2.5 tokenizer, y = LETTER v1_en accuracy; eight labelled points; no r."""
    raise NotImplementedError("TODO (Part 4)")


# ---- driver (given) -----------------------------------------------------------------------------------------------
def _write_csv(df: pd.DataFrame, path: str, name: str) -> None:
    if df is None or not len(df):
        print(f"[analysis] note: {name} not written (no rows)")
        return
    df.to_csv(path, index=False)
    print(f"[analysis] wrote {path} ({len(df)} rows)")


def _load_items_safe(lang: str = "en"):
    try:
        from harness.data import load_gmmlu_lite
        return load_gmmlu_lite(lang)
    except Exception as e:  # noqa: BLE001
        print(f"[analysis] note: could not load the {lang} dataset ({type(e).__name__}); option texts omitted")
        return None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Regenerate every CSV and figure from predictions/ (no model loaded).")
    ap.add_argument("--predictions", default="predictions")
    ap.add_argument("--out", default=".", help="directory for the CSV/JSONL outputs")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-log", default="run_log.txt")
    ap.add_argument("--tokenizer-csv", default=None, help="default: <out>/tokenizer.csv")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figures, exist_ok=True)
    preds = load_predictions(args.predictions)
    print(f"[analysis] {len(preds)} prediction files in {args.predictions}/")
    if not preds:
        return
    run_log = load_run_log(args.run_log) if os.path.exists(args.run_log) else {}

    results = build_results(preds, args.B, args.seed, run_log)
    _write_csv(results, os.path.join(args.out, "results.csv"), "results.csv")
    _write_csv(build_paired_vs_en(preds, args.B, args.seed), os.path.join(args.out, "paired_vs_en.csv"), "paired_vs_en.csv")
    perms = build_permutations(preds)
    _write_csv(perms, os.path.join(args.out, "permutations.csv"), "permutations.csv")
    _write_csv(build_native_instruction(preds, args.B, args.seed), os.path.join(args.out, "native_instruction.csv"), "native_instruction.csv")
    _write_csv(build_comparison(preds, args.B, args.seed), os.path.join(args.out, "comparison.csv"), "comparison.csv")
    _write_csv(build_part1_scorers(preds, args.B, args.seed), os.path.join(args.out, "part1_scorers.csv"), "part1_scorers.csv")
    _write_csv(build_cs_ca_gap(preds, args.B, args.seed), os.path.join(args.out, "cs_ca_gap.csv"), "cs_ca_gap.csv")

    dis = build_part1_disagreements(preds, items=_load_items_safe("en"))
    if dis:
        p = os.path.join(args.out, "part1_disagreements.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in dis:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[analysis] wrote {p} ({len(dis)} LETTER/CONT disagreements)")

    fig_accuracy_by_language(results, os.path.join(args.figures, FIG_ACCURACY))
    fig_position_bias(perms, os.path.join(args.figures, FIG_POSITION))
    tk_path = args.tokenizer_csv or os.path.join(args.out, "tokenizer.csv")
    tk = pd.read_csv(tk_path) if os.path.exists(tk_path) else None
    fig_fertility_vs_accuracy(tk, results, os.path.join(args.figures, FIG_FERTILITY))
    print("[analysis] done")


if __name__ == "__main__":
    main()
