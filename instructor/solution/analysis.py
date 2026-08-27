"""REFERENCE SOLUTION — analysis.py. Regenerates every CSV and figure from predictions/ with NO model loaded.

Usage (from the student/ directory):
    python analysis.py [--predictions predictions] [--out .] [--figures figures] [--B 2000] [--seed 0]

Outputs (all optional — a table is skipped with a note when its inputs are missing):
    results.csv              one row per (part, model, lang, scorer, prompt_variant, subset in all/CS/CA)
    paired_vs_en.csv         Part 2: every lang != en vs en, paired on sample_id (qwen2.5 LETTER v1_en)
    permutations.csv         Part 3: accuracy per permutation, mean/std, consistency rate, pooled letter fractions
    native_instruction.csv   Part 3: v2_native vs v1_en, paired
    comparison.csv           Part 5: qwen2.5 vs qwen3, diff = acc(qwen3) - acc(qwen2.5) (second minus first, like every
                             other diff column), paired CI + naive unpaired overlap + MDD
    part1_scorers.csv        Part 1: scorer x lang (en/zh/sw) accuracy, parse-fail rate, kappa(LETTER, GEN)
    part1_disagreements.jsonl  Part 1: en items where LETTER and CONT disagree (raw material for the 8 cases)
    cs_ca_gap.csv            Part 2: CS - CA gap per language, UNPAIRED bootstrap CI
    figures/accuracy_by_language.png, figures/position_bias.png, figures/fertility_vs_accuracy.png

Every function is importable so that instructor/grade.py can recompute the same tables from a submission's
predictions/ directory with the same harness.stats functions (B=2000, seed=0).
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # import harness from the script's directory

from harness import stats as ST                                    # noqa: E402
from harness.data import CORE_LANGS, LETTERS                       # noqa: E402
from harness.model import MODELS, PRIMARY_MODEL, CONTRAST_MODEL    # noqa: E402
from harness.prompts import PERM_ORDERS                            # noqa: E402
from harness.schema import RESULTS_COLUMNS, read_jsonl             # noqa: E402

# ---- fixed column lists (must match design section 4 / the grader) ------------------------------------------------
PAIRED_VS_EN_COLUMNS = ["lang", "n", "acc_en", "acc_lang", "diff", "paired_ci_low", "paired_ci_high",
                        "n_en_only_correct", "n_lang_only_correct", "agreement_rate"]
PERMUTATIONS_COLUMNS = ["lang", "acc_ABCD", "acc_BCDA", "acc_CDAB", "acc_DABC", "mean", "std", "consistency_rate",
                        "frac_pred_A", "frac_pred_B", "frac_pred_C", "frac_pred_D"]
NATIVE_COLUMNS = ["lang", "acc_v1_en", "acc_v2_native", "diff", "paired_ci_low", "paired_ci_high",
                  "n_v1_only_correct", "n_v2_only_correct"]
COMPARISON_COLUMNS = ["lang", "n", "acc_qwen25", "acc_qwen3", "diff", "paired_ci_low", "paired_ci_high",
                      "unpaired_overlap", "n_qwen25_only_correct", "n_qwen3_only_correct", "mdd_95"]
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


# ---- loading and selection ----------------------------------------------------------------------------------------
def load_predictions(pred_dir: str) -> dict[str, list[dict]]:
    """{basename: rows} for every predictions/*.jsonl (sorted by sample_id). Unreadable files are skipped with a note."""
    out = {}
    for path in sorted(glob.glob(os.path.join(pred_dir, "*.jsonl"))):
        try:
            rows = read_jsonl(path)
        except Exception as e:  # noqa: BLE001
            print(f"[analysis] note: cannot read {path}: {e}")
            continue
        if not rows:
            print(f"[analysis] note: {path} is empty; skipped")
            continue
        rows.sort(key=lambda r: r["sample_id"])
        out[os.path.basename(path)] = rows
    return out


def run_key(rows: list[dict]) -> tuple[str, str, str, str, str]:
    """(part, model_alias, lang, scorer, prompt_variant) of a prediction file, read from its records."""
    r = rows[0]
    alias = HUB_TO_ALIAS.get(r["model"], r["model"])
    return str(r["part"]), alias, r["lang"], r["scorer"], r["prompt_variant"]


def select(preds: dict[str, list[dict]], model: str, lang: str, scorer: str, variant: str,
           part: Optional[str] = None) -> Optional[list[dict]]:
    """Rows of the run (model alias, lang, scorer, variant). If `part` is given, that part is preferred but any
    part with the same protocol is accepted (a Part 1 `en` LETTER v1_en run IS the Part 2 run)."""
    cands = [(run_key(rows), rows) for rows in preds.values()
             if rows[0].get("benchmark", "global_mmlu_lite") == "global_mmlu_lite"]
    cands = [(k, rows) for k, rows in cands if k[1:] == (model, lang, scorer, variant)]
    if not cands:
        return None
    if part is not None:
        for k, rows in cands:
            if k[0] == str(part):
                return rows
    return cands[0][1]


def align(*runs: list[dict]) -> list[list[dict]]:
    """Restrict several runs to their common sample_ids and return them in the same (sorted) order."""
    common = {r["sample_id"] for r in runs[0]}
    for rows in runs[1:]:
        common &= {r["sample_id"] for r in rows}
    ids = sorted(common)
    out = []
    for rows in runs:
        by_id = {r["sample_id"]: r for r in rows}
        out.append([by_id[i] for i in ids])
    return out


def correct(rows: list[dict]) -> np.ndarray:
    return np.array([r["pred"] == r["gold"] for r in rows], dtype=bool)


def _acc_ci(rows: list[dict], B: int, seed: int) -> tuple[float, float, float]:
    return ST.bootstrap_ci(correct(rows), B=B, seed=seed)


# ---- run_log.txt -------------------------------------------------------------------------------------------------
def load_run_log(path: str) -> dict[tuple[str, str, str, str], float]:
    """{(model_hub, lang, scorer(LETTER/CONT/GEN), variant): wall_sec} — last line wins."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    if not lines:
        return out
    cols = [c.strip() for c in lines[0].split("|")]
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) != len(cols):
            continue
        d = dict(zip(cols, parts))
        try:
            out[(d["model"], d["lang"], d["scorer"].upper(), d["variant"])] = float(d["wall_sec"])
        except (KeyError, ValueError):
            continue
    return out


def _base_scorer(scorer: str) -> str:
    return "CONT" if scorer.startswith("CONT") else scorer


# ---- results.csv -------------------------------------------------------------------------------------------------
def build_results(preds: dict[str, list[dict]], B: int = 2000, seed: int = 0,
                  run_log: Optional[dict] = None) -> pd.DataFrame:
    run_log = run_log or {}
    recs = []
    for fname, rows in preds.items():
        r0 = rows[0]
        wall = run_log.get((r0["model"], r0["lang"], _base_scorer(r0["scorer"]), r0["prompt_variant"]), "")
        for subset in ["all", "CS", "CA"]:
            sub = rows if subset == "all" else [r for r in rows if r["cs_label"] == subset]
            if not sub:
                continue
            acc, lo, hi = _acc_ci(sub, B, seed)
            pf = float(np.mean([r["pred"] is None for r in sub])) if r0["scorer"] == "GEN" else ""
            recs.append({"part": str(r0["part"]), "model": r0["model"], "benchmark": r0.get("benchmark", "global_mmlu_lite"),
                         "lang": r0["lang"], "scorer": r0["scorer"], "prompt_variant": r0["prompt_variant"],
                         "subset": subset, "n": len(sub), "accuracy": round(acc, 6), "ci_low": round(lo, 6),
                         "ci_high": round(hi, 6), "parse_fail_rate": pf, "wall_clock_sec": wall})
    df = pd.DataFrame(recs, columns=RESULTS_COLUMNS)
    if len(df):
        lang_rank = {l: i for i, l in enumerate(CORE_LANGS)}
        df["_l"] = df["lang"].map(lambda l: lang_rank.get(l, 99))
        df["_s"] = df["subset"].map({"all": 0, "CS": 1, "CA": 2})
        df = df.sort_values(["part", "model", "scorer", "prompt_variant", "_l", "_s"]).drop(columns=["_l", "_s"])
    return df.reset_index(drop=True)


# ---- paired_vs_en.csv ---------------------------------------------------------------------------------------------
def build_paired_vs_en(preds: dict[str, list[dict]], B: int = 2000, seed: int = 0, model: str = PRIMARY_MODEL) -> pd.DataFrame:
    en = select(preds, model, "en", "LETTER", "v1_en", part="2")
    recs = []
    if en is None:
        print("[analysis] note: paired_vs_en.csv skipped — no en LETTER v1_en run for", model)
        return pd.DataFrame(columns=PAIRED_VS_EN_COLUMNS)
    for lang in CORE_LANGS:
        if lang == "en":
            continue
        rows = select(preds, model, lang, "LETTER", "v1_en", part="2")
        if rows is None:
            print(f"[analysis] note: paired_vs_en.csv: no {lang} run; row skipped")
            continue
        a, b = align(en, rows)
        if not a:
            continue
        ca, cb = correct(a), correct(b)
        diff, lo, hi = ST.paired_bootstrap_ci(cb, ca, B=B, seed=seed)     # acc(lang) - acc(en)
        n_en_only, n_lang_only = ST.discordant_counts(ca, cb)
        recs.append({"lang": lang, "n": len(a), "acc_en": float(ca.mean()), "acc_lang": float(cb.mean()), "diff": diff,
                     "paired_ci_low": lo, "paired_ci_high": hi, "n_en_only_correct": n_en_only,
                     "n_lang_only_correct": n_lang_only,
                     "agreement_rate": ST.agreement_rate([r["pred"] for r in a], [r["pred"] for r in b])})
    return pd.DataFrame(recs, columns=PAIRED_VS_EN_COLUMNS)


# ---- permutations.csv ---------------------------------------------------------------------------------------------
def perm_runs(preds: dict[str, list[dict]], lang: str, model: str = PRIMARY_MODEL) -> dict[str, list[dict]]:
    """{order: rows} for the permutation runs that exist; the Part 2 v1_en LETTER run stands in for perm_ABCD."""
    out = {}
    for order in PERM_ORDERS:
        rows = select(preds, model, lang, "LETTER", f"perm_{order}", part="3")
        if rows is None and order == "ABCD":
            rows = select(preds, model, lang, "LETTER", "v1_en", part="2")
        if rows is not None:
            out[order] = rows
    return out


def predicted_text_id(pred: Optional[str], order: str) -> Optional[int]:
    """Index in the ORIGINAL option order of the option the model chose under permutation `order`."""
    if pred is None:
        return None
    return LETTERS.index(order[LETTERS.index(pred)])


def build_permutations(preds: dict[str, list[dict]], model: str = PRIMARY_MODEL, langs=PERM_LANGS) -> pd.DataFrame:
    recs = []
    for lang in langs:
        runs = perm_runs(preds, lang, model)
        if not runs:
            print(f"[analysis] note: permutations.csv: no permutation runs for {lang}; row skipped")
            continue
        rec = {"lang": lang}
        accs = []
        for order in PERM_ORDERS:
            if order in runs:
                a = float(correct(runs[order]).mean())
                rec[f"acc_{order}"] = a
                accs.append(a)
            else:
                rec[f"acc_{order}"] = float("nan")
        rec["mean"] = float(np.mean(accs))
        rec["std"] = float(np.std(accs, ddof=0))
        # consistency: same predicted option TEXT under all four permutations (needs all four)
        if len(runs) == len(PERM_ORDERS):
            aligned = align(*[runs[o] for o in PERM_ORDERS])
            same = [len({predicted_text_id(rows[i]["pred"], o) for o, rows in zip(PERM_ORDERS, aligned)}) == 1
                    for i in range(len(aligned[0]))]
            rec["consistency_rate"] = float(np.mean(same)) if same else float("nan")
        else:
            print(f"[analysis] note: permutations.csv: {lang} has {len(runs)}/4 permutations; consistency_rate = NaN")
            rec["consistency_rate"] = float("nan")
        pooled = [r["pred"] for rows in runs.values() for r in rows]
        for L in LETTERS:
            rec[f"frac_pred_{L}"] = float(np.mean([p == L for p in pooled])) if pooled else float("nan")
        recs.append(rec)
    return pd.DataFrame(recs, columns=PERMUTATIONS_COLUMNS)


# ---- native_instruction.csv ---------------------------------------------------------------------------------------
def build_native_instruction(preds: dict[str, list[dict]], B: int = 2000, seed: int = 0,
                             model: str = PRIMARY_MODEL, langs=NATIVE_LANGS) -> pd.DataFrame:
    recs = []
    for lang in langs:
        v1 = select(preds, model, lang, "LETTER", "v1_en", part="2")
        v2 = select(preds, model, lang, "LETTER", "v2_native", part="3")
        if v1 is None or v2 is None:
            print(f"[analysis] note: native_instruction.csv: missing v1_en or v2_native run for {lang}; row skipped")
            continue
        a, b = align(v1, v2)
        ca, cb = correct(a), correct(b)
        diff, lo, hi = ST.paired_bootstrap_ci(cb, ca, B=B, seed=seed)     # acc(v2) - acc(v1)
        n1, n2 = ST.discordant_counts(ca, cb)
        recs.append({"lang": lang, "acc_v1_en": float(ca.mean()), "acc_v2_native": float(cb.mean()), "diff": diff,
                     "paired_ci_low": lo, "paired_ci_high": hi, "n_v1_only_correct": n1, "n_v2_only_correct": n2})
    return pd.DataFrame(recs, columns=NATIVE_COLUMNS)


# ---- comparison.csv -----------------------------------------------------------------------------------------------
def build_comparison(preds: dict[str, list[dict]], B: int = 2000, seed: int = 0,
                     model_a: str = PRIMARY_MODEL, model_b: str = CONTRAST_MODEL) -> pd.DataFrame:
    """comparison.csv: diff = acc(model_b = qwen3) - acc(model_a = qwen2.5), i.e. second minus first like every other
    diff column; n_qwen25_only_correct / n_qwen3_only_correct are the discordant counts (a-only, b-only)."""
    recs = []
    for lang in CORE_LANGS:
        ra = select(preds, model_a, lang, "LETTER", "v1_en", part="2")
        rb = select(preds, model_b, lang, "LETTER", "v1_en", part="5")
        if ra is None or rb is None:
            print(f"[analysis] note: comparison.csv: missing {model_a} or {model_b} run for {lang}; row skipped")
            continue
        a, b = align(ra, rb)
        if not a:
            continue
        ca, cb = correct(a), correct(b)
        diff, lo, hi = ST.paired_bootstrap_ci(cb, ca, B=B, seed=seed)     # acc(qwen3) - acc(qwen2.5) (D2: second - first)
        _, alo, ahi = ST.bootstrap_ci(ca, B=B, seed=seed)
        _, blo, bhi = ST.bootstrap_ci(cb, B=B, seed=seed)
        overlap = bool(alo <= bhi and blo <= ahi)
        n_a, n_b = ST.discordant_counts(ca, cb)
        recs.append({"lang": lang, "n": len(a), "acc_qwen25": float(ca.mean()), "acc_qwen3": float(cb.mean()),
                     "diff": diff, "paired_ci_low": lo, "paired_ci_high": hi, "unpaired_overlap": overlap,
                     "n_qwen25_only_correct": n_a, "n_qwen3_only_correct": n_b, "mdd_95": ST.mdd_95(n_a, n_b, len(a))})
    return pd.DataFrame(recs, columns=COMPARISON_COLUMNS)


# ---- part1_scorers.csv --------------------------------------------------------------------------------------------
def build_part1_scorers(preds: dict[str, list[dict]], B: int = 2000, seed: int = 0,
                        model: str = PRIMARY_MODEL, langs=PART1_LANGS) -> pd.DataFrame:
    recs = []
    for lang in langs:
        letter = select(preds, model, lang, "LETTER", "v1_en", part="1")
        gen = select(preds, model, lang, "GEN", "v1_en", part="1")
        kappa = float("nan")
        if letter is not None and gen is not None:
            a, b = align(letter, gen)
            if a:
                kappa = ST.cohens_kappa([r["pred"] for r in a], [r["pred"] for r in b])
        for scorer in PART1_SCORERS:
            rows = select(preds, model, lang, scorer, "v1_en", part="1")
            if rows is None:
                print(f"[analysis] note: part1_scorers.csv: no {scorer} run for {lang}; row skipped")
                continue
            acc, lo, hi = _acc_ci(rows, B, seed)
            pf = float(np.mean([r["pred"] is None for r in rows])) if scorer == "GEN" else float("nan")
            recs.append({"scorer": scorer, "lang": lang, "n": len(rows), "accuracy": acc, "ci_low": lo, "ci_high": hi,
                         "parse_fail_rate": pf, "kappa_letter_gen": kappa})
    return pd.DataFrame(recs, columns=PART1_COLUMNS)


def build_part1_disagreements(preds: dict[str, list[dict]], model: str = PRIMARY_MODEL, lang: str = "en",
                              items: Optional[list] = None) -> list[dict]:
    """en items where LETTER and CONT (raw) predictions differ. `items` (harness.data.Item list) adds option texts."""
    letter = select(preds, model, lang, "LETTER", "v1_en", part="1")
    cont = select(preds, model, lang, "CONT", "v1_en", part="1")
    if letter is None or cont is None:
        print("[analysis] note: part1_disagreements.jsonl skipped — need en LETTER and CONT runs")
        return []
    tok = select(preds, model, lang, "CONT_TOKNORM", "v1_en", part="1")
    chr_ = select(preds, model, lang, "CONT_CHARNORM", "v1_en", part="1")
    by_tok = {r["sample_id"]: r for r in (tok or [])}
    by_chr = {r["sample_id"]: r for r in (chr_ or [])}
    by_item = {it.sample_id: it for it in (items or [])}
    a, b = align(letter, cont)
    out = []
    for rl, rc in zip(a, b):
        if rl["pred"] == rc["pred"]:
            continue
        sid = rl["sample_id"]
        rec = {"sample_id": sid, "subject": sid.split("/")[0], "cs_label": rl["cs_label"], "gold": rl["gold"],
               "pred_letter": rl["pred"], "pred_cont": rc["pred"],
               "pred_cont_toknorm": by_tok.get(sid, {}).get("pred"), "pred_cont_charnorm": by_chr.get(sid, {}).get("pred"),
               "scores_letter": rl["scores"], "scores_cont": rc["scores"],
               "scores_cont_toknorm": by_tok.get(sid, {}).get("scores"), "scores_cont_charnorm": by_chr.get(sid, {}).get("scores")}
        it = by_item.get(sid)
        if it is not None:
            rec["question"] = it.question
            rec["options"] = it.options
            rec["option_char_lengths"] = [len(o) for o in it.options]
        else:
            rec["question"], rec["options"], rec["option_char_lengths"] = None, None, None
        out.append(rec)
    return out


# ---- cs_ca_gap.csv ------------------------------------------------------------------------------------------------
def build_cs_ca_gap(preds: dict[str, list[dict]], B: int = 2000, seed: int = 0, model: str = PRIMARY_MODEL) -> pd.DataFrame:
    recs = []
    for lang in CORE_LANGS:
        rows = select(preds, model, lang, "LETTER", "v1_en", part="2")
        if rows is None:
            print(f"[analysis] note: cs_ca_gap.csv: no {lang} run; row skipped")
            continue
        cs = correct([r for r in rows if r["cs_label"] == "CS"])
        ca = correct([r for r in rows if r["cs_label"] == "CA"])
        if len(cs) == 0 or len(ca) == 0:
            continue
        gap, lo, hi = ST.unpaired_bootstrap_ci(cs, ca, B=B, seed=seed)
        recs.append({"lang": lang, "n_cs": len(cs), "n_ca": len(ca), "acc_cs": float(cs.mean()), "acc_ca": float(ca.mean()),
                     "gap": gap, "unpaired_ci_low": lo, "unpaired_ci_high": hi, "detectable": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(recs, columns=CS_CA_COLUMNS)


# ---- figures ------------------------------------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_accuracy_by_language(results: pd.DataFrame, path: str) -> bool:
    """Points + 95% CI error bars, CORE_LANGS order, dashed line at 0.25; qwen2.5 and (if present) qwen3."""
    plt = _plt()
    series = []
    for alias, label, marker in [(PRIMARY_MODEL, "Qwen2.5-0.5B-Instruct", "o"), (CONTRAST_MODEL, "Qwen3-0.6B", "s")]:
        hub = MODELS[alias][0]
        df = results[(results.model == hub) & (results.scorer == "LETTER") & (results.prompt_variant == "v1_en")
                     & (results.subset == "all") & (results.lang.isin(CORE_LANGS))]
        if len(df):
            df = df.drop_duplicates("lang").set_index("lang").reindex(CORE_LANGS)
            series.append((label, marker, df))
    if not series:
        print("[analysis] note: accuracy_by_language.png skipped — no LETTER v1_en rows")
        return False
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(CORE_LANGS))
    for k, (label, marker, df) in enumerate(series):
        off = (k - (len(series) - 1) / 2) * 0.18
        acc = df["accuracy"].to_numpy(dtype=float)
        lo, hi = df["ci_low"].to_numpy(dtype=float), df["ci_high"].to_numpy(dtype=float)
        ax.errorbar(x + off, acc, yerr=[acc - lo, hi - acc], fmt=marker, capsize=3, label=label)
    ax.axhline(0.25, ls="--", color="gray", label="chance (0.25)")
    ax.set_xticks(x)
    ax.set_xticklabels(CORE_LANGS)
    ax.set_xlabel("language (Global-MMLU-Lite config)")
    ax.set_ylabel("accuracy (LETTER, v1_en) with 95% bootstrap CI")
    ax.set_ylim(0, max(0.7, float(np.nanmax([s[2]["ci_high"].max() for s in series])) + 0.05))
    ax.set_xlim(-0.6, len(CORE_LANGS) - 0.4)
    ax.set_title("Accuracy by language (LETTER, v1_en; item-level 95% bootstrap CI)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def fig_position_bias(perms: pd.DataFrame, path: str) -> bool:
    """Grouped bars: fraction of predictions on each letter per language, pooled over permutations; line at 0.25."""
    plt = _plt()
    if perms is None or not len(perms):
        print("[analysis] note: position_bias.png skipped — permutations.csv empty")
        return False
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(perms))
    w = 0.2
    for k, L in enumerate(LETTERS):
        ax.bar(x + (k - 1.5) * w, perms[f"frac_pred_{L}"].to_numpy(dtype=float), width=w, label=f"pred = {L}")
    ax.axhline(0.25, ls="--", color="gray", label="uniform (0.25)")
    ax.set_xticks(x)
    ax.set_xticklabels(perms["lang"].tolist())
    ax.set_xlabel("language")
    ax.set_ylabel("fraction of predictions (pooled over 4 permutations)")
    ax.set_title("Position bias: predicted-letter distribution (gold is uniform by construction)")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def fig_fertility_vs_accuracy(tokenizer_df: pd.DataFrame, results: pd.DataFrame, path: str,
                              tokenizer_name: str = PRIMARY_MODEL) -> bool:
    """x = mean_tokens (qwen2.5 tokenizer), y = accuracy (qwen2.5 LETTER v1_en), eight labelled points. No r."""
    plt = _plt()
    if tokenizer_df is None or not len(tokenizer_df):
        print("[analysis] note: fertility_vs_accuracy.png skipped — tokenizer.csv missing")
        return False
    hub = MODELS[PRIMARY_MODEL][0]
    acc = results[(results.model == hub) & (results.scorer == "LETTER") & (results.prompt_variant == "v1_en")
                  & (results.subset == "all")].drop_duplicates("lang").set_index("lang")
    tk = tokenizer_df[tokenizer_df.tokenizer.astype(str) == tokenizer_name].set_index("lang")
    if not len(tk):
        tk = tokenizer_df[tokenizer_df.tokenizer.astype(str) == hub].set_index("lang")
    langs = [l for l in CORE_LANGS if l in acc.index and l in tk.index]
    if not langs:
        print("[analysis] note: fertility_vs_accuracy.png skipped — no language with both tokenizer and accuracy rows")
        return False
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for l in langs:
        xv, yv = float(tk.loc[l, "mean_tokens"]), float(acc.loc[l, "accuracy"])
        ax.scatter(xv, yv, s=40)
        ax.annotate(l, (xv, yv), textcoords="offset points", xytext=(5, 4))
    ax.axhline(0.25, ls="--", color="gray", label="chance (0.25)")
    ax.set_xlabel("mean tokens per item (Qwen2.5 tokenizer; question + 4 options)")
    ax.set_ylabel("accuracy (Qwen2.5, LETTER, v1_en)")
    ax.set_title(f"Tokenizer fertility vs accuracy (n = {len(langs)} languages, descriptive only)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


# ---- driver -------------------------------------------------------------------------------------------------------
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
        print(f"[analysis] note: could not load the {lang} dataset for option texts ({type(e).__name__}); "
              "part1_disagreements.jsonl will not carry option lengths")
        return None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Regenerate every CSV and figure from predictions/ (no model loaded).")
    ap.add_argument("--predictions", default="predictions")
    ap.add_argument("--out", default=".", help="directory for the CSV/JSONL outputs")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-log", default=None, help="run_log.txt path (default: <out>/run_log.txt, then ./run_log.txt)")
    ap.add_argument("--tokenizer-csv", default=None, help="tokenizer.csv path (default: <out>/tokenizer.csv)")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figures, exist_ok=True)
    preds = load_predictions(args.predictions)
    print(f"[analysis] {len(preds)} prediction files in {args.predictions}/")
    if not preds:
        print("[analysis] nothing to do")
        return
    run_log_path = args.run_log or (os.path.join(args.out, "run_log.txt") if os.path.exists(os.path.join(args.out, "run_log.txt"))
                                    else "run_log.txt")
    run_log = load_run_log(run_log_path)

    results = build_results(preds, args.B, args.seed, run_log)
    _write_csv(results, os.path.join(args.out, "results.csv"), "results.csv")
    _write_csv(build_paired_vs_en(preds, args.B, args.seed), os.path.join(args.out, "paired_vs_en.csv"), "paired_vs_en.csv")
    perms = build_permutations(preds)
    _write_csv(perms, os.path.join(args.out, "permutations.csv"), "permutations.csv")
    _write_csv(build_native_instruction(preds, args.B, args.seed), os.path.join(args.out, "native_instruction.csv"), "native_instruction.csv")
    _write_csv(build_comparison(preds, args.B, args.seed), os.path.join(args.out, "comparison.csv"), "comparison.csv")
    _write_csv(build_part1_scorers(preds, args.B, args.seed), os.path.join(args.out, "part1_scorers.csv"), "part1_scorers.csv")
    _write_csv(build_cs_ca_gap(preds, args.B, args.seed), os.path.join(args.out, "cs_ca_gap.csv"), "cs_ca_gap.csv")

    have_p1 = select(preds, PRIMARY_MODEL, "en", "LETTER", "v1_en") and select(preds, PRIMARY_MODEL, "en", "CONT", "v1_en")
    dis = build_part1_disagreements(preds, items=_load_items_safe("en") if have_p1 else None)
    if dis:
        p = os.path.join(args.out, "part1_disagreements.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in dis:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[analysis] wrote {p} ({len(dis)} LETTER/CONT disagreements)")

    fig_accuracy_by_language(results, os.path.join(args.figures, FIG_ACCURACY))
    fig_position_bias(perms, os.path.join(args.figures, FIG_POSITION))
    tk_path = args.tokenizer_csv or os.path.join(args.out, "tokenizer.csv")
    if not os.path.exists(tk_path) and os.path.exists("tokenizer.csv"):
        tk_path = "tokenizer.csv"
    tk = pd.read_csv(tk_path) if os.path.exists(tk_path) else None
    fig_fertility_vs_accuracy(tk, results, os.path.join(args.figures, FIG_FERTILITY))
    print("[analysis] done")


if __name__ == "__main__":
    main()
