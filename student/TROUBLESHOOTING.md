# TROUBLESHOOTING

Everything below assumes you are in the `student/` directory with the virtual environment
activated (`source .venv/bin/activate`). **`python -m harness ...` only works from `student/`**
because `harness` is imported from the current working directory; running it from elsewhere gives
`No module named harness` (or, worse, silently imports a different copy).

Quick reference of the flags that fix most problems:

```
python -m harness run --part 2 --model qwen2.5 --lang en --scorer LETTER --variant v1_en \
    --student-id S123 [--lite/--no-lite] [--batch-size 8] [--device auto|mps|cpu|cuda] [--dtype auto|fp16|fp32]
```

---

## 1. Apple Silicon (M1/M2/M3/M4) — MPS backend

### "My log-probs are NaN / -inf / all identical / accuracy is at chance" on MPS
fp16 on MPS occasionally produces NaN or garbage logits, in particular on older macOS or torch
versions, and for the long prompts of Part 3. Symptoms: the smoke test fails its 0.05-nat check,
`scores` contain `nan` (the harness refuses to write them), or every `pred` is the same letter.
**Fix:** rerun with `--dtype fp32`. It is about 1.5-2x slower but exact. Record the dtype you used
(it is in `run_log.txt` automatically) and keep it constant across a part.

### "MPS backend out of memory" / "RuntimeError: MPS allocator ..."
The harness catches this and automatically halves the batch size (8 -> 4 -> 2 -> 1); the final batch
size is logged in the `batch_size_final` column of `run_log.txt`. If the run still dies at batch size
1, or you want to avoid the halving pauses, start with `--batch-size 4` (or `--batch-size 2` on 8 GB
machines), close memory-hungry apps (browsers with many tabs), and keep the Part 3 permutation runs
on their own. Setting `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` disables the MPS memory cap entirely,
which can let the machine swap instead of crashing; use it only as a last resort.

### macOS < 13.3
MPS support for the operations the harness uses (left-padded attention masks, fp16 log_softmax,
`generate`) is incomplete before macOS 13.3. Symptoms: "The operator ... is not currently implemented
for the MPS device", wrong results, or hangs. Either set `PYTORCH_ENABLE_MPS_FALLBACK=1` (slow ops fall
back to CPU, results still correct) or use `--device cpu --dtype fp32` (works everywhere, 3-5x slower,
consider `--lite`). Upgrading macOS is the real fix.

### "If run_log.txt shows < 3 items/s on an M-series chip you are probably on CPU"
Check the `device` column. `torch.backends.mps.is_available()` returns False if you installed an x86
(Rosetta) Python; reinstall an arm64 Python (`python -c "import platform; print(platform.machine())"`
must print `arm64`).

---

## 2. Intel Macs (and any CPU-only machine)

**Install first:** PyTorch's last macOS x86_64 wheel is 2.2.2 (needs Python 3.9-3.11), so run
`pip install torch==2.2.2` *before* `pip install -r requirements.txt`; if that fails, use
`colab/fallback.ipynb`. Then use `--device cpu --dtype fp32` (also what `auto` picks when no GPU is
present). The smoke test measures your throughput and writes `lite_mode: true` into `smoke_ok.json`
when the projected core compute exceeds 150 minutes (`--force-lite` / `--no-lite-auto` override it);
every subsequent `python -m harness run` reads that default, so every run uses the fixed 200-item
LITE subset (first 100 CS + first 100 CA items in `sample_id` order). You can force either mode per
run with `--lite` / `--no-lite`, but **do not mix modes within a submission**: `validate_predictions.py`
checks that every file has the same `lite_mode` as `smoke_ok.json`. Batch size 2 is a
good default; bigger batches do not help on CPU. Expect P1 60-90 min, P2 ~30 min, P3 45-75 min, P5
~25 min; start long parts in the evening. There is no penalty for LITE mode; just say so in the
report (the CIs are wider).

---

## 3. Hugging Face cache, offline mode, pre-downloading

Models and datasets are cached under `~/.cache/huggingface/` (`hub/` for models and datasets).
Override the location with `export HF_HOME=/path/with/space` before running anything if your home
directory is small. Sizes: Qwen2.5-0.5B-Instruct ~1.0 GB, Qwen3-0.6B ~1.5 GB, SmolLM2-360M ~0.7 GB,
Global-MMLU-Lite < 10 MB.

After the first successful download you can work without network:

```
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

If you see `OSError: We couldn't connect to 'https://huggingface.co'` with offline mode on, the file
was never downloaded (or a different revision is being requested). Unset the variables and rerun.

Pre-download everything (models at the pinned revisions from `harness/model.py`, datasets at the
revisions from `harness/data.py`) in one go, e.g. on campus Wi-Fi before going home:

```
python -c "from huggingface_hub import snapshot_download as s; from harness.model import MODELS; from harness import data as D; [s(h, revision=r) for h, r in MODELS.values()]; [s(i, revision=r, repo_type='dataset') for i, r in [(D.GMMLU_LITE_ID, D.GMMLU_LITE_REV)]]"
```

(Add `(D.BELEBELE_ID, D.BELEBELE_REV), (D.MGSM_ID, D.MGSM_REV), (D.XCOPA_ID, D.XCOPA_REV)` to the
dataset list only if you attempt the stretch tasks; Belebele is large.) The smoke test also downloads
Qwen2.5 and the `en` config.

A `401`/`403` on a dataset means you are logged into a Hugging Face account that has not accepted
that dataset's terms; the core datasets are ungated, so try `huggingface-cli logout` or unset
`HF_TOKEN`.

---

## 4. Classic scorer bugs (accuracy at chance on `en`)

Qwen2.5-0.5B scores roughly 0.45-0.55 on `en` with LETTER. If your `en` accuracy is near 0.25,
one of these is almost certainly the cause:

* **`"A"` vs `" A"`.** After the assistant prefix `Answer:` the model predicts the token for
  `" A"` (with a leading space), not `"A"`. The tokenizer has different ids for the two, and
  `"A"` without the space is close to never predicted at that position. Use
  `harness.model.letter_token_ids(tok)` (which asserts each `" X"` is a single token) and read the
  logits at those ids. The same applies to the CONT scorer: the continuation is `" " + option_text`.
* **Reading the wrong position.** With left padding the last real token is always at index -1;
  with right padding it is not. `forward_last_logits` handles this for you; if you tokenise
  yourself, set `tok.padding_side = "left"` and pass `attention_mask`. Symptom: batch-of-8 results
  differ from single-prompt results (`tests/test_scorers.py::test_batch_equals_single` fails) and
  accuracy drops as batch size grows.
* **`add_special_tokens=True`.** The chat template already contains the special tokens; adding a
  BOS again shifts every position by one. Symptom: tiny but systematic score differences vs
  `reference/en_first20.jsonl`.
* **Softmax over the four letters only, or raw logits instead of log-probs.** `scores` must be
  log-probabilities over the *full* vocabulary (`log_softmax` over all logits, then index A-D), so
  every value is <= 0. The validator rejects positive scores.
* **CONT: scoring the prompt tokens too.** Only the continuation tokens count. If the continuation
  starts with a token that merged with the prompt's `:` (e.g. `":A"`), your token boundary is off:
  compare `len(tok.encode(prompt))` with the length of the encoded prompt+continuation.
* **Part 3: gold not remapped.** After `permute`, `gold` must be the letter that is correct in the
  *displayed* order. A permutation run scoring ~0.25 while `perm_ABCD` scores ~0.5 is flagged
  automatically by `check_results.py` as "gold not remapped".

---

## 5. Interrupted runs, `.partial` files, resuming

The harness appends finished records every `--checkpoint-every` (default 50) items to
`predictions/<name>.jsonl.partial`. If a run dies (OOM, sleep, Ctrl-C), rerun the *same* command:
it reads the `.partial` file, prints `resuming: N items already scored`, and continues. When the run
completes, the final `.jsonl` is written in dataset order and the `.partial` file removed. Delete
`.partial` files only if you changed the scorer code in between (`make clean-partial`), otherwise
you would mix old and new scores in one file. Never submit `.partial` files.

---

## 6. Reading `run_log.txt`

One pipe-separated line per completed run, appended by the harness (never edit it by hand; the
grader cross-checks it against `smoke_ok.json`). Columns:

```
timestamp | command | chip | macos | torch | device | dtype | batch_size_final | model | model_sha | lang | scorer | variant | n_items | wall_sec | items_per_sec
```

* `device`/`dtype`: what was actually used (`mps float16`, `cpu float32`, ...). If it says `cpu` on an
  M-series Mac, see section 1.
* `batch_size_final` smaller than what you passed: OOM halvings happened.
* `items_per_sec`: M1 fp16 LETTER ~5-10 items/s, CONT ~1-2, GEN ~2-4; CPU is 5-10x slower.
* `wall_sec` is what goes into `results.csv` `wall_clock_sec` for that run.

---

## 7. Where does `probe.jsonl` come from?

You never write it. Every `LETTER` + `v1_en` run on Global-MMLU-Lite updates `probe.jsonl` with the
20 items chosen deterministically for your student id from the full 400-item list (`python -m harness
probe-ids --student-id S123` prints them; in LITE mode only the ones inside the 200-item subset get
rows) for that (model, lang): `student_id, model, lang, sample_id, scores, pred, gold,
n_prompt_tokens`. Rerunning the same (model, lang) replaces those rows. Your reflection must cite
three of these items by `sample_id` with numbers that match this file, so run Part 2 with the
`--student-id` you will submit under.

---

## 8. Other common errors

| Message | Cause / fix |
|---|---|
| `NotImplementedError: TODO ...` | That function is yours to write (`harness/scorers.py`, `harness/stats.py`, `prompts.permute`). |
| `[harness] internal schema error ...` | Your scorer returned something invalid (NaN, positive score, wrong length). See section 4. |
| `model ... is not in the allowed list` | Use an alias: `qwen2.5`, `qwen3`, `smollm2`. |
| `no native template for language` | `v2_native` exists only for `zh`, `hi`, `sw`. |
| `expected 400 test items` | Wrong/partial dataset download; delete the dataset from the cache and redo. |
| `pytest: unrecognized arguments: --no-model` | Run `pytest tests --no-model` from `student/` so `tests/conftest.py` is found. |
| `FAIL ... expected 8 lines, found 16` from the validator with `--n-expected` | Debug files of different `--n` sizes in `predictions/`; delete every `--n`-truncated file before `make check`. |
| very slow first run | Model compilation/warm-up on MPS; the second batch is fast. |
