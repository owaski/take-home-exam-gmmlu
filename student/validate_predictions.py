#!/usr/bin/env python
"""validate_predictions.py -- schema and dataset-consistency checks for predictions/*.jsonl.

This is the SAME script the grader runs. Run it from the student/ directory:

    python validate_predictions.py                       # every predictions/*.jsonl
    python validate_predictions.py --dir predictions     # same
    python validate_predictions.py predictions           # same (a bare directory argument is scanned)
    python validate_predictions.py predictions/p2_qwen2.5_en_LETTER_v1_en.jsonl ...

Per file it checks that
  * every record passes harness.schema.check_record (types, scores <= 0, pred == argmax, GEN rules, ...);
  * all records share part / model / model_sha / benchmark / lang / scorer / prompt_variant / lite_mode / student_id;
  * the file name is p{part}_{alias}_{lang}_{scorer}_{variant}.jsonl with alias from harness.model.MODELS;
  * `model` is one of harness.model.ALLOWED_MODEL_IDS;
  * global_mmlu_lite files have exactly 400 lines (200 in LITE mode) with unique sample_ids that equal the
    dataset's (belebele: unique ids only, no count check);
  * cs_label and gold_text_id match the dataset, and `gold` is the letter expected under the permutation
    (computed here from gold_text_id + the order, independently of harness.prompts.permute);
  * n_prompt_tokens > 0.
Across files it checks that every global_mmlu_lite file carries the SAME lite_mode, and -- when smoke_ok.json exists
in the current directory -- that it equals the lite_mode recorded there (FAIL otherwise; a missing smoke_ok.json is
only a WARN).

--n-expected N is a DEBUG-ONLY override of the expected line count (for files produced with `--n N`);
graded submissions are always validated with the default (400 / 200).
Exit code 1 if any file fails.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from harness import data as D
from harness import schema as S
from harness.model import ALLOWED_MODEL_IDS, MODELS
from harness.prompts import PERM_ORDERS

SHARED_FIELDS = ["part", "model", "model_sha", "benchmark", "lang", "scorer", "prompt_variant", "lite_mode", "student_id"]
ALIAS_OF = {hub: alias for alias, (hub, _rev) in MODELS.items()}
_DATASET_CACHE: dict = {}


def expected_gold(gold_text_id: int, variant: str) -> str:
    """Letter of the correct option as shown under `variant`.

    perm_XXXX: new slot k shows the ORIGINAL option order[k] (see harness.prompts.permute docstring), so the
    gold letter is the slot k for which order[k] == original gold letter. Any other variant keeps the order.
    """
    orig_letter = D.LETTERS[gold_text_id]
    order = variant[len("perm_"):] if variant.startswith("perm_") else "ABCD"
    if order not in PERM_ORDERS:
        raise ValueError(f"unknown permutation order {order!r}")
    return D.LETTERS[order.index(orig_letter)]


def load_reference(benchmark: str, lang: str, lite: bool):
    """{sample_id: Item} for the dataset behind a file (cached). Returns None if the benchmark cannot be loaded."""
    key = (benchmark, lang, lite)
    if key not in _DATASET_CACHE:
        try:
            if benchmark == "global_mmlu_lite":
                items = D.load_gmmlu_lite(lang, lite=lite)
            elif benchmark == "belebele":
                items = D.load_belebele(lang)
            else:
                items = None
        except Exception as e:  # pragma: no cover - network / cache problems
            print(f"  WARN: could not load {benchmark}/{lang}: {e}", file=sys.stderr)
            items = None
        _DATASET_CACHE[key] = None if items is None else {it.sample_id: it for it in items}
    return _DATASET_CACHE[key]


def validate_file(path: str, n_expected: int | None = None) -> list[str]:
    errs: list[str] = []
    try:
        recs = S.read_jsonl(path)
    except Exception as e:
        return [f"cannot parse JSONL: {e}"]
    if not recs:
        return ["file is empty"]

    # 1. per-record schema
    for i, r in enumerate(recs):
        for e in S.check_record(r):
            errs.append(f"line {i + 1}: {e}")
    if any(not isinstance(r, dict) or set(S.PRED_FIELDS) - set(r) for r in recs):
        return errs  # cannot go further without the fields

    # 2. shared fields
    for k in SHARED_FIELDS:
        vals = {repr(r[k]) for r in recs}
        if len(vals) > 1:
            errs.append(f"field {k} is not constant across the file: {sorted(vals)[:5]}")
    first = recs[0]
    part, model, bench = first["part"], first["model"], first["benchmark"]
    lang, scorer, variant, lite = first["lang"], first["scorer"], first["prompt_variant"], first["lite_mode"]

    # 3. model + filename
    if model not in ALLOWED_MODEL_IDS:
        errs.append(f"model {model!r} not in the allowed list {sorted(ALLOWED_MODEL_IDS)}")
    alias = ALIAS_OF.get(model)
    if alias is not None:
        want = S.prediction_filename(str(part), alias, lang, scorer, variant)
        if os.path.basename(path) != want:
            errs.append(f"filename {os.path.basename(path)!r} does not match record fields (expected {want!r})")
    if variant.startswith("perm_") and variant[len("perm_"):] not in PERM_ORDERS:
        errs.append(f"unknown permutation variant {variant!r}")

    # 4. counts and ids
    ids = [r["sample_id"] for r in recs]
    if len(set(ids)) != len(ids):
        dup = sorted({x for x in ids if ids.count(x) > 1})
        errs.append(f"duplicate sample_ids ({len(ids) - len(set(ids))}): {dup[:5]}")
    ref = load_reference(bench, lang, bool(lite))
    if bench == "global_mmlu_lite":
        want_n = n_expected if n_expected is not None else (2 * D.LITE_PER_LABEL if lite else 400)
        if len(recs) != want_n:
            if n_expected is not None:
                errs.append(f"DEBUG override: expected {want_n} lines (--n-expected), found {len(recs)}; a file made "
                            f"with a different --n is truncated -- delete truncated debug files before re-running")
            else:
                errs.append(f"expected {want_n} lines, found {len(recs)}")
        if ref is not None:
            ref_ids = set(ref)
            if n_expected is not None:
                missing = []  # debug mode: only require the given ids to exist in the dataset
            else:
                missing = sorted(ref_ids - set(ids))
            extra = sorted(set(ids) - ref_ids)
            if missing:
                errs.append(f"{len(missing)} dataset sample_ids missing from file, e.g. {missing[:5]}")
            if extra:
                errs.append(f"{len(extra)} sample_ids not in the dataset ({'LITE' if lite else 'full'}), e.g. {extra[:5]}")
    elif ref is None:
        errs.append(f"unknown or unloadable benchmark {bench!r}; dataset checks skipped")

    # 5. gold / cs_label against the dataset
    if ref is not None:
        for i, r in enumerate(recs):
            it = ref.get(r["sample_id"])
            if it is None:
                continue
            if it.cs_label != "-" and r["cs_label"] != it.cs_label:
                errs.append(f"line {i + 1}: cs_label {r['cs_label']!r} != dataset {it.cs_label!r}")
            if r["gold_text_id"] != it.gold_text_id:
                errs.append(f"line {i + 1}: gold_text_id {r['gold_text_id']} != dataset {it.gold_text_id}")
                continue
            try:
                want = expected_gold(r["gold_text_id"], variant)
            except ValueError:
                break
            if r["gold"] != want:
                errs.append(f"line {i + 1}: gold {r['gold']!r} != expected {want!r} under {variant} "
                            f"(gold_text_id={r['gold_text_id']})")
    return errs


def file_lite_mode(path: str):
    """(benchmark, lite_mode) of a file's first record, or None when unreadable."""
    try:
        recs = S.read_jsonl(path)
        r = recs[0]
        return r.get("benchmark", "global_mmlu_lite"), bool(r.get("lite_mode"))
    except Exception:  # noqa: BLE001
        return None


def check_lite_consistency(paths: list[str], smoke_path: str = "smoke_ok.json") -> tuple[list[str], list[str]]:
    """Cross-file check: all global_mmlu_lite files share one lite_mode, equal to smoke_ok.json's when present.
    Returns (failures, warnings)."""
    fails, warns = [], []
    modes: dict[bool, list[str]] = {}
    for p in paths:
        got = file_lite_mode(p)
        if got is None or got[0] != "global_mmlu_lite":
            continue
        modes.setdefault(got[1], []).append(os.path.basename(p))
    if len(modes) > 1:
        fails.append("lite_mode is not the same in every prediction file: "
                     + "; ".join(f"lite_mode={m}: {len(fs)} file(s) e.g. {fs[:3]}" for m, fs in sorted(modes.items()))
                     + " -- a submission is either entirely LITE or entirely full")
    if not os.path.exists(smoke_path):
        warns.append(f"{smoke_path} not found (run smoke_test.py); lite_mode cannot be checked against it")
        return fails, warns
    try:
        with open(smoke_path, encoding="utf-8") as f:
            smoke_lite = bool(json.load(f).get("lite_mode", False))
    except Exception as e:  # noqa: BLE001
        warns.append(f"{smoke_path} unreadable ({e}); lite_mode not checked against it")
        return fails, warns
    for m, fs in sorted(modes.items()):
        if m != smoke_lite:
            fails.append(f"{len(fs)} file(s) have lite_mode={m} but {smoke_path} says lite_mode={smoke_lite} "
                         f"(e.g. {fs[:3]}); re-run smoke_test.py or the affected runs so that they agree")
    return fails, warns


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="prediction files or a directory (default: every *.jsonl in --dir)")
    ap.add_argument("--dir", default="predictions", help="directory scanned when no paths are given")
    ap.add_argument("--n-expected", type=int, default=None,
                    help="DEBUG ONLY: expected line count override for files made with `--n N`; never used in grading")
    ap.add_argument("--show", type=int, default=5, help="errors shown per file")
    args = ap.parse_args(argv)

    paths = []
    for p in args.paths:
        if os.path.isdir(p):                       # a bare directory argument works like --dir
            paths += sorted(glob.glob(os.path.join(p, "*.jsonl")))
        else:
            paths.append(p)
    if not args.paths:
        paths = sorted(glob.glob(os.path.join(args.dir, "*.jsonl")))
    if not paths:
        print(f"no prediction files found ({'in ' + repr(args.paths) if args.paths else 'dir=' + repr(args.dir)})")
        return 1
    n_bad = 0
    for p in paths:
        errs = validate_file(p, args.n_expected)
        if errs:
            n_bad += 1
            shown = "; ".join(errs[: args.show])
            more = f" (+{len(errs) - args.show} more)" if len(errs) > args.show else ""
            print(f"FAIL  {p}: {len(errs)} error(s): {shown}{more}")
        else:
            print(f"OK    {p}")
    lite_fails, lite_warns = check_lite_consistency(paths)
    for w in lite_warns:
        print(f"WARN  {w}")
    for e in lite_fails:
        print(f"FAIL  lite_mode: {e}")
    print(f"{len(paths) - n_bad}/{len(paths)} files OK" + ("" if n_bad == 0 else f"; {n_bad} FAILED")
          + ("" if not lite_fails else f"; lite_mode consistency FAILED"))
    return 1 if (n_bad or lite_fails) else 0


if __name__ == "__main__":
    sys.exit(main())
