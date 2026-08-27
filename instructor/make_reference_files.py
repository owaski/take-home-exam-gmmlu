"""Generate/merge student/reference/en_first20.jsonl and student/reference/smoke_reference.json.

Run once per backend, from a directory containing the SOLUTION overlay (student/ + instructor/solution/harness/):
    python instructor/make_reference_files.py --out student/reference --device cpu  --dtype fp32
    python instructor/make_reference_files.py --out student/reference --device cuda --dtype fp16   # or mps
Results for each dtype are merged into the same files; `scores` in en_first20.jsonl is the fp16 vector when
available (what MPS students will see), `scores_fp32` / `scores_fp16` are always kept separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from harness import data as D, prompts as PR                      # noqa: E402
from harness.model import apply_chat, forward_last_logits, letter_token_ids, load, n_tokens   # noqa: E402
from harness.scorers import score_letter                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="student/reference")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--k", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model, tok, info = load("qwen2.5", a.device, a.dtype)
    key = "fp16" if info["dtype"] in ("float16", "bfloat16") else "fp32"
    print("backend", info)
    items = D.load_gmmlu_lite("en")[: a.k]
    prompts = [apply_chat(tok, PR.build_messages(it, "v1_en"), PR.ASSISTANT_PREFIX) for it in items]
    scores = score_letter(model, tok, prompts)

    # ---- en_first20.jsonl (merge) ----
    path = os.path.join(a.out, "en_first20.jsonl")
    old = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line); old[r["sample_id"]] = r
    with open(path, "w", encoding="utf-8") as f:
        for it, sc in zip(items, scores):
            r = old.get(it.sample_id, {"sample_id": it.sample_id, "gold": it.answer})
            r[f"scores_{key}"] = [round(float(x), 6) for x in sc]
            canon = r.get("scores_fp16") or r.get("scores_fp32")
            r["scores"] = canon
            r["pred"] = D.LETTERS[int(np.argmax(canon))]
            f.write(json.dumps(r) + "\n")
    print("wrote", path)

    # ---- smoke_reference.json (merge) ----
    it, p, sc = items[0], prompts[0], scores[0]
    spath = os.path.join(a.out, "smoke_reference.json")
    ref = json.load(open(spath)) if os.path.exists(spath) else {}
    logits = forward_last_logits(model, tok, [p])[0]
    ref.update({"sample_id": it.sample_id, "prompt_sha256": hashlib.sha256(p.encode("utf-8")).hexdigest(),
                "n_prompt_tokens": n_tokens(tok, p), "letter_ids": letter_token_ids(tok),
                "top1_token_id": int(np.argmax(logits)), "model_sha": info["model_sha"]})
    ref.setdefault("letter_logprobs", {})[key] = [round(float(x), 6) for x in sc]
    json.dump(ref, open(spath, "w"), indent=1)
    print("wrote", spath, ref["letter_logprobs"])


if __name__ == "__main__":
    main()
