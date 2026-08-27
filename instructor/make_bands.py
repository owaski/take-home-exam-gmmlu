"""Regenerate instructor/reference_results/bands.json and probe.jsonl.

Usage: python instructor/make_bands.py <dir with full results.csv> <dir with LITE results.csv> <reference_results dir>
Bands: accuracy +- 0.045 (full, n=400) / +- 0.065 (LITE, n=200), keyed "alias|lang|scorer|variant" and "lite:...".
probe.jsonl: the INSTRUCTOR probe ids x every LETTER v1_en (model, lang) run found in <reference_results>/predictions.
"""
import json, os, sys
import pandas as pd
full, lite, ref = sys.argv[1:4]           # dirs holding results.csv (full), results.csv (LITE), and the reference_results dir to write
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "student"))   # harness package (data/model/schema only; no scorers needed)
from harness.model import MODELS
from harness.data import probe_ids
from harness.schema import read_jsonl
HUB_TO_ALIAS = {hub: a for a, (hub, _) in MODELS.items()}
TOL = {"": 0.045, "lite:": 0.065}
bands = {}
for prefix, d in [("", full), ("lite:", lite)]:
    df = pd.read_csv(os.path.join(d, "results.csv"), dtype=str, keep_default_na=False)
    for _, r in df[df.subset == "all"].iterrows():
        key = prefix + f"{HUB_TO_ALIAS.get(r.model, r.model)}|{r.lang}|{r.scorer}|{r.prompt_variant}"
        if key in bands:
            continue
        acc = float(r.accuracy)
        bands[key] = {"acc": acc, "lo": round(acc - TOL[prefix], 3), "hi": round(acc + TOL[prefix], 3), "n": int(r.n)}
json.dump(bands, open(os.path.join(ref, "bands.json"), "w"), indent=1)
print("bands.json:", len(bands), "keys;", sum(k.startswith("lite:") for k in bands), "lite")
# probe.jsonl from the reference predictions: 20 probe ids x every (model, lang) LETTER v1_en global_mmlu_lite run
ids = probe_ids("INSTRUCTOR")
out, seen = [], set()
import glob
for p in sorted(glob.glob(os.path.join(ref, "predictions", "*.jsonl"))):
    rows = read_jsonl(p); r0 = rows[0]
    if r0.get("benchmark", "global_mmlu_lite") != "global_mmlu_lite" or r0["scorer"] != "LETTER" or r0["prompt_variant"] != "v1_en":
        continue
    if (r0["model"], r0["lang"]) in seen:
        continue
    seen.add((r0["model"], r0["lang"]))
    by = {r["sample_id"]: r for r in rows}
    for sid in ids:
        r = by[sid]
        out.append({"student_id": "INSTRUCTOR", "model": r0["model"], "lang": r0["lang"], "sample_id": sid, "scores": r["scores"],
                    "pred": r["pred"], "gold": r["gold"], "n_prompt_tokens": r["n_prompt_tokens"]})
out.sort(key=lambda r: (r["model"], r["lang"], r["sample_id"]))
with open(os.path.join(ref, "probe.jsonl"), "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("probe.jsonl:", len(out), "records,", len(seen), "groups")
