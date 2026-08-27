"""Prediction-record schema, JSONL I/O and the run log. Frozen."""
from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import time
from typing import Iterable

SCORERS = ["LETTER", "CONT", "CONT_TOKNORM", "CONT_CHARNORM", "GEN"]
PRED_FIELDS = {
    "part": str, "model": str, "model_sha": str, "benchmark": str, "lang": str, "sample_id": str,
    "cs_label": str, "scorer": str, "prompt_variant": str, "gold": str, "gold_text_id": int,
    "pred": (str, type(None)), "scores": (list, type(None)), "raw_generation": (str, type(None)),
    "n_prompt_tokens": int, "lite_mode": bool, "student_id": str,
}
RESULTS_COLUMNS = ["part", "model", "benchmark", "lang", "scorer", "prompt_variant", "subset", "n",
                   "accuracy", "ci_low", "ci_high", "parse_fail_rate", "wall_clock_sec"]


def prediction_filename(part: str, model_alias: str, lang: str, scorer: str, variant: str) -> str:
    return f"p{part}_{model_alias}_{lang}_{scorer}_{variant}.jsonl"


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_record(rec: dict) -> list[str]:
    """Return a list of schema problems for one prediction record (empty list = OK)."""
    errs = []
    for k, typ in PRED_FIELDS.items():
        if k not in rec:
            errs.append(f"missing field {k}")
        elif not isinstance(rec[k], typ):
            errs.append(f"field {k} has type {type(rec[k]).__name__}")
    extra = set(rec) - set(PRED_FIELDS)
    if extra:
        errs.append(f"unexpected fields {sorted(extra)}")
    if errs:
        return errs
    if rec["scorer"] not in SCORERS:
        errs.append(f"unknown scorer {rec['scorer']}")
    if rec["gold"] not in "ABCD" or len(rec["gold"]) != 1:
        errs.append(f"gold must be A-D, got {rec['gold']!r}")
    if rec["pred"] is not None and (rec["pred"] not in "ABCD" or len(rec["pred"]) != 1):
        errs.append(f"pred must be A-D or null, got {rec['pred']!r}")
    if not 0 <= rec["gold_text_id"] <= 3:
        errs.append("gold_text_id out of range")
    if rec["scorer"] == "GEN":
        if rec["scores"] is not None:
            errs.append("GEN records must have scores = null")
        if rec["raw_generation"] is None:
            errs.append("GEN records must carry raw_generation")
    else:
        s = rec["scores"]
        if s is None or len(s) != 4:
            errs.append("non-GEN records need exactly 4 scores")
        else:
            if any((not isinstance(x, (int, float))) or math.isnan(x) or math.isinf(x) for x in s):
                errs.append("scores must be finite numbers")
            elif any(x > 1e-9 for x in s):
                errs.append("scores are log-probabilities and must be <= 0")
            else:
                argmax = "ABCD"[max(range(4), key=lambda i: s[i])]
                if rec["pred"] != argmax:
                    errs.append(f"pred {rec['pred']} != argmax(scores) {argmax}")
        if rec["pred"] is None:
            errs.append("pred may be null only for GEN")
        if rec["raw_generation"] is not None:
            errs.append("raw_generation must be null unless scorer == GEN")
    if rec["n_prompt_tokens"] <= 0:
        errs.append("n_prompt_tokens must be positive")
    return errs


def machine_info() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version()}
    try:
        import torch
        info["torch"] = torch.__version__
    except Exception:
        info["torch"] = "?"
    if platform.system() == "Darwin":
        try:
            info["chip"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            info["macos"] = platform.mac_ver()[0]
        except Exception:
            info["chip"], info["macos"] = "?", platform.mac_ver()[0]
    else:
        info["chip"] = platform.processor() or "?"
        info["macos"] = "-"
    return info


def append_run_log(path: str, fields: dict) -> None:
    """Append one pipe-separated line. Never rewrite this file by hand."""
    cols = ["timestamp", "command", "chip", "macos", "torch", "device", "dtype", "batch_size_final", "model",
            "model_sha", "lang", "scorer", "variant", "n_items", "wall_sec", "items_per_sec"]
    line = " | ".join(str(fields.get(c, "")) for c in cols)
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write(" | ".join(cols) + "\n")
        f.write(line + "\n")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
