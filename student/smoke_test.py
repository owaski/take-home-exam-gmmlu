"""Part 0 smoke test. Run from the student/ directory:

    python smoke_test.py --student-id S123 [--device auto|mps|cpu|cuda] [--dtype auto|fp16|fp32]
                         [--force-lite] [--no-lite-auto]

Checks your environment, loads the model and the `en` data, verifies ONE reference item against the
instructor's numbers (reference/smoke_reference.json), times a few forward passes, and writes smoke_ok.json.
Upload smoke_ok.json to the LMS by Day 2. Exit code 0 = everything OK.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

# Weight budgets for the core runs, in single-forward equivalents (one LETTER forward on one item = 1):
#   Part 1: 1200 items (en, zh, sw) x (1 LETTER + 4 CONT passes + ~2.5 for an 8-token greedy GEN)
#   Part 2: 8 langs x 400 items LETTER                                   = 3200
#   Part 3: 3 new permutations x 4 langs x 400 + native on 3 langs x 400 = 6000
#   Part 5: contrast model, 8 langs x 400 LETTER                         = 3200
CORE_FORWARD_EQUIVALENTS = 1200 * (1 + 4 + 2.5) + 3200 + 6000 + 3200
LITE_IF_PROJECTED_OVER_MIN = 150
REFERENCE_PATH = os.path.join("reference", "smoke_reference.json")
TIMING_N, TIMING_BATCH = 16, 8

OK, FAIL, WARN = "[ok]  ", "[FAIL]", "[warn]"


def step(msg: str, ok: bool = True) -> None:
    print(f"{OK if ok else FAIL} {msg}", flush=True)


def die(msg: str) -> None:
    step(msg, ok=False)
    print("\nSmoke test FAILED. See TROUBLESHOOTING.md (and bring this output to office hours).")
    sys.exit(1)


def free_mem_gb() -> float | None:
    try:
        import psutil
        return round(psutil.virtual_memory().available / 1e9, 2)
    except Exception:
        pass
    if platform.system() == "Darwin":
        try:
            return round(int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)) / 1e9, 2)
        except Exception:
            return None
    try:  # Linux fallback
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return round(int(line.split()[1]) * 1024 / 1e9, 2)
    except Exception:
        pass
    return None


def rss_gb() -> float | None:
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(rss / (1e9 if platform.system() == "Darwin" else 1e6), 2)  # bytes on mac, kB on Linux
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student-id", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--force-lite", action="store_true", help="write lite_mode=true regardless of speed")
    ap.add_argument("--no-lite-auto", action="store_true", help="never auto-enable LITE mode (you accept the compute)")
    args = ap.parse_args()
    t_start = time.time()

    # ---- step 1: environment --------------------------------------------------------------------
    try:
        import numpy as np
        import torch
        import transformers
        import datasets
        from harness import data, model as hm, prompts, schema
    except Exception as e:
        die(f"import failed: {e!r}. Is the venv active and are you running from the student/ directory?")
    mi = schema.machine_info()
    device = hm.select_device(args.device)
    dtype = str(hm.select_dtype(device, args.dtype)).replace("torch.", "")
    free_gb = free_mem_gb()
    step(f"python {mi['python']}, torch {torch.__version__}, transformers {transformers.__version__}, "
         f"datasets {datasets.__version__}")
    step(f"platform {mi['platform']} | chip {mi['chip']} | macOS {mi['macos']} | free memory {free_gb} GB")
    step(f"device {device}, dtype {dtype}")
    if device == "cuda" and not torch.cuda.is_available():
        die("--device cuda requested but CUDA is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        die("--device mps requested but MPS is not available")

    # ---- step 2: dataset --------------------------------------------------------------------------
    try:
        items = data.load_gmmlu_lite("en")
    except Exception as e:
        die(f"could not load Global-MMLU-Lite en @ {data.GMMLU_LITE_REV[:8]}: {e!r}")
    n_cs = sum(it.cs_label == "CS" for it in items)
    n_ca = sum(it.cs_label == "CA" for it in items)
    if not (len(items) == 400 and n_cs == 200 and n_ca == 200):
        die(f"expected 400 items (200 CS / 200 CA), got {len(items)} ({n_cs} CS / {n_ca} CA)")
    step(f"dataset en: 400 items, {n_cs} CS / {n_ca} CA, revision {data.GMMLU_LITE_REV[:8]}")

    # ---- step 3: model ----------------------------------------------------------------------------
    try:
        model, tok, info = hm.load("qwen2.5", device=device, dtype=args.dtype)
    except Exception as e:
        die(f"model load failed: {e!r}")
    rss_after = rss_gb()
    step(f"loaded {info['model']} @ {info['model_sha'][:8]} in {info['load_seconds']} s; process RSS {rss_after} GB")

    # ---- step 4: reference check on ONE item (this is NOT the scorer; write your own in harness/scorers.py) -
    if not os.path.exists(REFERENCE_PATH):
        die(f"{REFERENCE_PATH} missing — re-download the handout")
    with open(REFERENCE_PATH) as f:
        ref = json.load(f)
    by_id = {it.sample_id: it for it in items}
    if ref["sample_id"] not in by_id:
        die(f"reference sample_id {ref['sample_id']} not in the en data")
    prompt = hm.apply_chat(tok, prompts.build_messages(by_id[ref["sample_id"]], "v1_en"))
    sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if sha != ref["prompt_sha256"]:
        die("prompt hash mismatch: the frozen template in harness/prompts.py or harness/model.apply_chat was changed")
    n_tok = hm.n_tokens(tok, prompt)
    letter_ids = hm.letter_token_ids(tok)
    if letter_ids != ref["letter_ids"]:
        die(f"letter token ids {letter_ids} != reference {ref['letter_ids']} (wrong tokenizer revision?)")
    if n_tok != ref["n_prompt_tokens"]:
        step(f"prompt has {n_tok} tokens, reference says {ref['n_prompt_tokens']} (tokenizer changed?)", ok=False)
    # reference check, not the scorer: log-softmax over the full vocab, then pick the four letter tokens
    logits = hm.forward_last_logits(model, tok, [prompt])[0]
    logp = logits - np.logaddexp.reduce(logits)
    letter_lp = [float(logp[i]) for i in letter_ids]
    key = "fp16" if dtype in ("float16", "bfloat16") else "fp32"
    tol = 0.05 if key == "fp16" else 0.01
    max_dev = max(abs(a - b) for a, b in zip(letter_lp, ref["letter_logprobs"][key]))
    if not np.isfinite(letter_lp).all():
        die("non-finite log-probs: on mps try --dtype fp32 (see TROUBLESHOOTING.md)")
    if max_dev > tol:
        die(f"letter log-probs deviate from reference by {max_dev:.4f} nats (tolerance {tol}); "
            f"got {[round(x, 3) for x in letter_lp]}, expected {ref['letter_logprobs'][key]}")
    step(f"reference item {ref['sample_id']}: {n_tok} tokens, letter logprobs {[round(x, 3) for x in letter_lp]}, "
         f"max deviation {max_dev:.4f} <= {tol}")
    top1 = int(np.argmax(logits))
    if top1 != ref["top1_token_id"]:
        print(f"{WARN} argmax token id {top1} != reference {ref['top1_token_id']} (numerics differ; not fatal)")
    probe_hash = hashlib.sha256(json.dumps([round(x, 3) for x in letter_lp]).encode()).hexdigest()

    # ---- step 5: throughput and compute projection -------------------------------------------------
    timing_prompts = [hm.apply_chat(tok, prompts.build_messages(it, "v1_en")) for it in items[:TIMING_N]]
    batcher = hm._Batcher(TIMING_BATCH)
    hm.forward_last_logits(model, tok, timing_prompts[:TIMING_BATCH], batcher)      # warm-up (not timed)
    t0 = time.time()
    hm.forward_last_logits(model, tok, timing_prompts, batcher)
    ips = TIMING_N / (time.time() - t0)
    projected_min = CORE_FORWARD_EQUIVALENTS / ips / 60
    step(f"throughput {ips:.2f} items/s (batch {batcher.batch_size}); projected core compute "
         f"{projected_min:.0f} min for {CORE_FORWARD_EQUIVALENTS:.0f} forward-equivalents")
    # LITE decision: --force-lite wins; otherwise auto-enable when the projection exceeds the threshold,
    # unless --no-lite-auto was given. (The device itself does not matter, only the measured speed.)
    lite = args.force_lite or (not args.no_lite_auto and projected_min > LITE_IF_PROJECTED_OVER_MIN)
    if args.force_lite:
        reason = "--force-lite given"
    elif args.no_lite_auto:
        reason = f"--no-lite-auto given (projected {projected_min:.0f} min, you accept the compute)"
    elif lite:
        reason = f"projected {projected_min:.0f} min > {LITE_IF_PROJECTED_OVER_MIN} min threshold"
    else:
        reason = f"projected {projected_min:.0f} min <= {LITE_IF_PROJECTED_OVER_MIN} min threshold"
    print("=" * 78, flush=True)
    print(f"LITE MODE = {'ON' if lite else 'OFF'}  ({reason}).", flush=True)
    print(f"  `python -m harness run` reads lite_mode from smoke_ok.json. Override: re-run smoke_test.py with "
          f"--force-lite (ON) or --no-lite-auto (OFF), or pass --lite / --no-lite to `python -m harness run`.", flush=True)
    print("=" * 78, flush=True)
    step(f"lite_mode = {lite} ({reason})")

    # ---- step 6: smoke_ok.json --------------------------------------------------------------------
    out = {
        "student_id": args.student_id, "timestamp": schema.now_iso(), "python": mi["python"],
        "torch": torch.__version__, "transformers": transformers.__version__, "platform": mi["platform"],
        "macos": mi["macos"], "chip": mi["chip"], "device": device, "dtype": dtype, "free_mem_gb": free_gb,
        "rss_gb_after_load": rss_after, "model_sha": info["model_sha"], "dataset_sha": data.GMMLU_LITE_REV,
        "lite_mode": bool(lite), "items_per_sec": round(ips, 3), "projected_core_minutes": round(projected_min, 1),
        "probe_hash": probe_hash, "wall_seconds": round(time.time() - t_start, 1),
    }
    with open("smoke_ok.json", "w") as f:
        json.dump(out, f, indent=2)
    step(f"wrote smoke_ok.json ({out['wall_seconds']} s total). Upload it to the LMS.")


if __name__ == "__main__":
    main()
