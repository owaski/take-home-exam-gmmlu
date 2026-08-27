"""Command-line entry point: `python -m harness run ...`. Frozen; add flags only if you must."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from . import data as D
from . import prompts as PR
from . import schema as S
from .model import MODELS, _Batcher, apply_chat, load, n_tokens, resolve_model

PRED_DIR, PROBE_FILE, RUN_LOG, SMOKE_FILE = "predictions", "probe.jsonl", "run_log.txt", "smoke_ok.json"


def _lite_default() -> bool:
    if os.path.exists(SMOKE_FILE):
        try:
            return bool(json.load(open(SMOKE_FILE)).get("lite_mode", False))
        except Exception:
            return False
    return False


def _load_items(args) -> list[D.Item]:
    if args.benchmark == "global_mmlu_lite":
        return D.load_gmmlu_lite(args.lang, lite=args.lite, n=args.n)
    if args.benchmark == "belebele":
        return D.load_belebele(args.lang, n=args.n)
    raise ValueError(args.benchmark)


def cmd_run(args) -> None:
    from . import scorers as SC   # imported late so a NotImplementedError surfaces with a clear message

    hub_id, _rev = resolve_model(args.model)          # unknown model -> ValueError("model ... is not in the allowed list")
    alias = args.model if args.model in MODELS else [k for k, v in MODELS.items() if v[0] == hub_id][0]
    if args.lite is None:
        args.lite = _lite_default()
    variant = args.variant
    scorer = args.scorer.upper()
    if scorer not in ("LETTER", "CONT", "GEN"):
        sys.exit("--scorer must be LETTER, CONT or GEN (CONT writes the three CONT* files)")
    items = _load_items(args)
    items = [PR.apply_permutation(it, variant) for it in items]
    if args.benchmark == "belebele" and variant != "v1_en":
        sys.exit("Belebele runs use v1_en only")

    outs = {}
    scorers_written = [scorer] if scorer != "CONT" else ["CONT", "CONT_TOKNORM", "CONT_CHARNORM"]
    for sname in scorers_written:
        outs[sname] = args.out if (args.out and len(scorers_written) == 1) else \
            os.path.join(PRED_DIR, S.prediction_filename(args.part, alias, args.lang, sname, variant))

    model, tok, info = load(args.model, args.device, args.dtype)
    print(f"[harness] loaded {info['model']}@{info['model_sha'][:8]} on {info['device']} ({info['dtype']}) in {info['load_seconds']}s; "
          f"{len(items)} items; lite={args.lite}; batch={args.batch_size}", flush=True)
    prompts = [apply_chat(tok, PR.build_messages(it, variant), PR.ASSISTANT_PREFIX) for it in items]
    ntok = [n_tokens(tok, p) for p in prompts]
    batcher = _Batcher(args.batch_size)

    # ---- resumable checkpointing: process in chunks, append to .partial files ----------------------
    partial = {s: outs[s] + ".partial" for s in scorers_written}
    done = {}
    for s in scorers_written:
        done[s] = {r["sample_id"]: r for r in S.read_jsonl(partial[s])} if os.path.exists(partial[s]) else {}
    todo_idx = [i for i, it in enumerate(items) if any(it.sample_id not in done[s] for s in scorers_written)]
    if len(todo_idx) < len(items):
        print(f"[harness] resuming: {len(items) - len(todo_idx)} items already scored", flush=True)

    t0 = time.time()
    CH = args.checkpoint_every
    for c in range(0, len(todo_idx), CH):
        idx = todo_idx[c:c + CH]
        chunk_items = [items[i] for i in idx]
        chunk_prompts = [prompts[i] for i in idx]
        if scorer == "LETTER":
            scores = SC.score_letter(model, tok, chunk_prompts, batcher)
            per_scorer = {"LETTER": [(list(map(float, s)), None) for s in scores]}
        elif scorer == "CONT":
            res = SC.score_cont(model, tok, chunk_prompts, [it.options for it in chunk_items], batcher)
            per_scorer = {s: [(list(map(float, r[s])), None) for r in res] for s in scorers_written}
        else:
            raw = SC.score_gen(model, tok, chunk_prompts, batcher, max_new_tokens=args.max_new_tokens)
            per_scorer = {"GEN": [(None, r) for r in raw]}
        for sname, rows in per_scorer.items():
            recs = []
            for it, i, (sc, raw) in zip(chunk_items, idx, rows):
                if sc is not None:
                    pred = D.LETTERS[int(np.argmax(sc))]
                else:
                    pred = SC.parse_gen(raw)
                rec = {"part": str(args.part), "model": info["model"], "model_sha": info["model_sha"],
                       "benchmark": it.benchmark, "lang": args.lang, "sample_id": it.sample_id, "cs_label": it.cs_label,
                       "scorer": sname, "prompt_variant": variant, "gold": it.answer, "gold_text_id": it.gold_text_id,
                       "pred": pred, "scores": sc, "raw_generation": raw, "n_prompt_tokens": ntok[i],
                       "lite_mode": bool(args.lite), "student_id": args.student_id}
                probs = S.check_record(rec)
                if probs:
                    sys.exit(f"[harness] internal schema error for {it.sample_id}: {probs}")
                done[sname][it.sample_id] = rec
                recs.append(rec)
            with open(partial[sname], "a", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_done = min(c + CH, len(todo_idx))
        el = time.time() - t0
        print(f"[harness] {n_done}/{len(todo_idx)} items  {el:.0f}s  {n_done / max(el, 1e-6):.2f} items/s  batch={batcher.batch_size}", flush=True)

    wall = time.time() - t0
    for sname in scorers_written:
        rows = [done[sname][it.sample_id] for it in items]
        S.write_jsonl(outs[sname], rows)
        os.remove(partial[sname])
        acc = float(np.mean([r["pred"] == r["gold"] for r in rows]))
        print(f"[harness] wrote {outs[sname]}  n={len(rows)}  accuracy={acc:.3f}", flush=True)
    mi = S.machine_info()
    S.append_run_log(RUN_LOG, {"timestamp": S.now_iso(), "command": " ".join(sys.argv), "chip": mi["chip"], "macos": mi["macos"],
                               "torch": mi["torch"], "device": info["device"], "dtype": info["dtype"],
                               "batch_size_final": batcher.batch_size, "model": info["model"], "model_sha": info["model_sha"],
                               "lang": args.lang, "scorer": scorer, "variant": variant, "n_items": len(items),
                               "wall_sec": round(wall, 1), "items_per_sec": round(len(todo_idx) / max(wall, 1e-6), 2)})
    if scorer == "LETTER" and variant == "v1_en" and args.benchmark == "global_mmlu_lite":
        _update_probe(args.student_id, info["model"], args.lang, done["LETTER"], items)


def _update_probe(student_id: str, model: str, lang: str, recs: dict, items: list[D.Item]) -> None:
    """Keep probe.jsonl up to date: the per-student probe items x every (model, lang) LETTER v1_en run.

    The probe ids are always drawn from the full 400-item `en` list (D.probe_ids); here they are intersected
    with the items actually scored, so a LITE run contributes only the probe ids inside the LITE subset.
    """
    scored = {it.sample_id for it in items}
    ids = [sid for sid in D.probe_ids(student_id) if sid in scored]
    old = S.read_jsonl(PROBE_FILE) if os.path.exists(PROBE_FILE) else []
    keep = [r for r in old if not (r["model"] == model and r["lang"] == lang)]
    for sid in ids:
        if sid in recs:
            r = recs[sid]
            keep.append({"student_id": student_id, "model": model, "lang": lang, "sample_id": sid, "scores": r["scores"],
                         "pred": r["pred"], "gold": r["gold"], "n_prompt_tokens": r["n_prompt_tokens"]})
    keep.sort(key=lambda r: (r["model"], r["lang"], r["sample_id"]))
    S.write_jsonl(PROBE_FILE, keep)
    print(f"[harness] probe.jsonl updated ({len(ids)} probe items present in this run for {model} / {lang})", flush=True)


def cmd_probe_ids(args) -> None:
    """Print the 20 probe sample_ids (always drawn from the full 400 `en` items, regardless of LITE mode)."""
    print("\n".join(D.probe_ids(args.student_id)))


FIRST20_FILE = os.path.join("reference", "en_first20.jsonl")


def cmd_check_first20(args) -> None:
    """Score the first 20 `en` items with YOUR score_letter and compare with reference/en_first20.jsonl.

    Writes nothing into predictions/. PASS needs >= 19/20 matching predictions; exit code 1 otherwise.
    """
    from . import scorers as SC

    if not os.path.exists(FIRST20_FILE):
        sys.exit(f"[check-first20] {FIRST20_FILE} not found: run from the student/ directory")
    ref = S.read_jsonl(FIRST20_FILE)
    items = D.load_gmmlu_lite("en", n=len(ref))
    if [it.sample_id for it in items] != [r["sample_id"] for r in ref]:
        sys.exit("[check-first20] sample_id order differs from the reference file (dataset revision changed?)")
    model, tok, info = load("qwen2.5", args.device, args.dtype)
    key = "scores_fp16" if info["dtype"] in ("float16", "bfloat16") else "scores_fp32"
    tol = 0.05 if key == "scores_fp16" else 0.01
    print(f"[check-first20] loaded {info['model']}@{info['model_sha'][:8]} on {info['device']} ({info['dtype']}); "
          f"comparing against {key}", flush=True)
    prompts = [apply_chat(tok, PR.build_messages(it, "v1_en"), PR.ASSISTANT_PREFIX) for it in items]
    scores = SC.score_letter(model, tok, prompts, _Batcher(args.batch_size))
    n_match, max_dev = 0, 0.0
    for it, sc, r in zip(items, scores, ref):
        sc = [float(x) for x in sc]
        if len(sc) != 4 or not all(np.isfinite(sc)):
            sys.exit(f"[check-first20] score_letter returned {sc!r} for {it.sample_id} (expected 4 finite log-probs)")
        pred = D.LETTERS[int(np.argmax(sc))]
        exp = r.get(key) or r["scores"]
        dev = float(np.abs(np.array(sc) - np.array(exp)).max())
        max_dev = max(max_dev, dev)
        ok = pred == r["pred"]
        n_match += ok
        print(f"  {'ok  ' if ok else 'DIFF'} {it.sample_id:<40} pred={pred} ref={r['pred']} gold={r['gold']} "
              f"max|dscore|={dev:.4f}", flush=True)
    verdict = "PASS" if n_match >= 19 else "FAIL"
    print(f"[check-first20] {verdict}: {n_match}/20 predictions match the reference; "
          f"max abs score deviation {max_dev:.4f} vs {key} (expect <= {tol})", flush=True)
    if max_dev > tol and verdict == "PASS":
        print("[check-first20] WARN: predictions match but the scores deviate more than expected; check that you "
              "log-softmax over the FULL vocabulary (not just the four letters)", flush=True)
    if verdict == "FAIL":
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m harness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="score one (model, lang, scorer, variant) and write predictions/*.jsonl")
    r.add_argument("--part", required=True, help='e.g. "1", "2", "3", "5", "S1"')
    r.add_argument("--model", required=True, help=f"alias {sorted(MODELS)} or full hub id")
    r.add_argument("--lang", required=True)
    r.add_argument("--scorer", required=True, help="LETTER | CONT | GEN")
    r.add_argument("--variant", default="v1_en", help="v1_en | v2_native | perm_ABCD | perm_BCDA | perm_CDAB | perm_DABC")
    r.add_argument("--student-id", required=True)
    r.add_argument("--benchmark", default="global_mmlu_lite", choices=["global_mmlu_lite", "belebele"])
    r.add_argument("--lite", action=argparse.BooleanOptionalAction, default=None, help="LITE subset (default: from smoke_ok.json)")
    r.add_argument("--n", type=int, default=None,
                   help="debug only for Global-MMLU-Lite (delete such files before make check); required for Belebele S1 (--n 300)")
    r.add_argument("--batch-size", type=int, default=8)
    r.add_argument("--device", default="auto", help="auto | mps | cpu | cuda")
    r.add_argument("--dtype", default="auto", help="auto | fp16 | fp32 | bf16")
    r.add_argument("--max-new-tokens", type=int, default=8)
    r.add_argument("--checkpoint-every", type=int, default=50)
    r.add_argument("--out", default=None, help="override output path (single-scorer runs only)")
    r.set_defaults(func=cmd_run)
    p = sub.add_parser("probe-ids", help="print the 20 probe sample_ids for a student id")
    p.add_argument("--student-id", required=True)
    p.set_defaults(func=cmd_probe_ids)
    c = sub.add_parser("check-first20", help="score the first 20 en items with your score_letter and compare with reference/en_first20.jsonl")
    c.add_argument("--student-id", required=True)
    c.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    c.add_argument("--dtype", default="auto", help="auto | fp16 | fp32")
    c.add_argument("--batch-size", type=int, default=8)
    c.set_defaults(func=cmd_check_first20)
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
