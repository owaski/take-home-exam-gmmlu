# Small Models, Many Languages
### Take-home exam · one week · individual work · MacBook only

You will build a small but rigorous evaluation harness and use it to measure how a sub-1B-parameter
multilingual language model — **Qwen2.5-0.5B-Instruct** — behaves on a parallel eight-language
multiple-choice benchmark, **Global-MMLU-Lite**. The exam is deliberately *not* about producing one
accuracy number. Each part isolates one lesson about multilingual evaluation:

| Part | Lesson | Points |
|---|---|---|
| 0 | Your machine can run the model (smoke test, required gate) | 0 |
| 1 | The *scoring method* changes the number (LETTER vs CONT vs GEN) | 15 |
| 2 | Accuracy collapses toward chance in unsupported languages; CIs; paired comparisons on parallel items | 20 |
| 3 | How much of a "result" is option-position and prompt-language noise | 15 |
| 4 | What the tokenizer does to non-Latin scripts, and to cost | 10 |
| 5 | Comparing two models *correctly* on the same items | 15 |
| 6 | Report, reflection, reproducibility | 25 |
| S1–S5 | Stretch tasks (optional, bonus capped at 10) | ≤ 10 |

Total model compute is roughly **30–60 minutes on an Apple-Silicon MacBook** (the smoke test measures your
machine and projects the exact figure) and 3–4 hours in LITE mode on an Intel Mac. Budget **15–20 hours of
your own time** including the report. Everything you need is in this folder; `TROUBLESHOOTING.md` covers
the known failure modes.

---

## 1. Learning objectives

By the end you will be able to

1. implement and compare three multiple-choice scoring protocols for a causal LM — answer-letter
   log-probability (**LETTER**), full-option continuation log-likelihood with and without length
   normalisation (**CONT**), and greedy generation plus parsing (**GEN**) — and explain, with evidence,
   why they disagree and which to trust for a tiny model;
2. run a controlled multilingual sweep across resource levels, scripts and official-support status, and
   separate culturally-agnostic from culturally-sensitive items;
3. quantify prompt sensitivity (option permutations, instruction language) and compare that spread to
   sampling uncertainty, so you can say whether a single reported number means anything;
4. relate tokenizer fertility to accuracy and inference cost, and use a controlled language pair (id vs sw)
   to argue what fertility does and does not explain;
5. apply correct small-*n* statistics: an item-level bootstrap you wrote yourself, **paired** bootstraps and
   discordant counts on parallel items, the contrast with naive CI overlap, and a minimum-detectable-difference
   calculation — and know which comparisons are paired and which are not;
6. ship a reproducible harness: pinned revisions, frozen prompts, a machine-checkable prediction schema, a run
   log and a README with one command per part.

---

## 2. Fixed choices (frozen — do not change for any graded run)

### 2.1 Models

| Role | Model | Params | Licence | Alias in the CLI |
|---|---|---|---|---|
| Primary (Parts 1–4) | `Qwen/Qwen2.5-0.5B-Instruct` | 494 M | Apache-2.0 | `qwen2.5` |
| Contrast (Part 5) | `Qwen/Qwen3-0.6B` (thinking disabled) | 752 M | Apache-2.0 | `qwen3` |
| Stretch S3 only | `HuggingFaceTB/SmolLM2-360M-Instruct` | 362 M | Apache-2.0 | `smollm2` |

Revisions are pinned by commit hash in `harness/model.py`; `run_log.txt` records the hash actually loaded.
Models over 1 B parameters (e.g. Llama-3.2-1B) are not allowed anywhere in this exam.

### 2.2 Benchmark

`CohereLabs/Global-MMLU-Lite`, `test` split, revision pinned in `harness/data.py`. Every language config has
**400 items**: 200 labelled **CS** (culturally sensitive) and 200 **CA** (culturally agnostic). The 400
questions are *the same questions, translated*: item `sample_id` X in `en` is the same question as X in `yo`.
Items are always processed in `sample_id` order.

### 2.3 Languages (exact config names, fixed presentation order)

| Config | Language | Resource level | Script | Qwen2.5 official support* | Role |
|---|---|---|---|---|---|
| `en` | English | high | Latin | yes | anchor |
| `de` | German | high | Latin | yes | high-resource, Latin |
| `zh` | Chinese | high | Han | yes | high-resource, non-Latin, no whitespace |
| `ar` | Arabic | high/mid | Arabic (RTL) | yes | non-Latin, high fertility |
| `hi` | Hindi | mid | Devanagari | yes | Indic script, very high fertility |
| `id` | Indonesian | mid | Latin | yes | **controlled pair with `sw`**: same script, comparable fertility (1.5× vs 1.9× English), supported |
| `sw` | Swahili | low | Latin | **no** | unsupported — expect near chance |
| `yo` | Yoruba | low | Latin + tone diacritics | **no** | unsupported; diacritics inflate fertility — expect near chance |

\* From the Qwen2/Qwen2.5 release notes (27 languages beyond English and Chinese; Swahili and Yoruba are
absent). Two chance-level languages are included on purpose; analyses that are only meaningful above chance
(permutation consistency, native-instruction delta) are run on clearly-above-chance languages plus `sw` as the
single unsupported probe — say so when you interpret them.

Per-part language subsets: **Part 1** en, zh, sw · **Part 2** all 8 · **Part 3** permutations on en, zh, hi,
sw; native instruction on zh, hi, sw · **Part 4** all 8 · **Part 5** all 8.

### 2.4 Prompt templates

Every prompt is rendered with the model's own chat template
(`tokenizer.apply_chat_template(..., add_generation_prompt=True)`, Qwen3 with `enable_thinking=False`) and
then the assistant prefix **`Answer:`** is appended as plain text. The prefix stays English in *every* variant
so that the LETTER candidates are scored in an identical local context. `harness/model.py::apply_chat` does
this; never hand-roll `<|im_start|>` tokens.

**`v1_en`** (default; used for LETTER, CONT, GEN and all permutation runs):

```
[system] You are a helpful assistant.
[user]   The following is a multiple-choice question about {subject}. Choose the single best answer.

         {question}

         A. {option_a}
         B. {option_b}
         C. {option_c}
         D. {option_d}

         Reply with only the letter (A, B, C, or D).
[assistant prefix] Answer:
```

`{subject}` is the dataset subject with underscores replaced by spaces. No few-shot exemplars in the core.

**`v2_native`** (zh, hi, sw only): identical structure; the system line and the two instruction sentences are
translated (`harness/prompts.py::NATIVE_TEMPLATES`, frozen). The question and options are already in the
target language; the subject name stays in English (the dataset does not translate it) — a limitation you may
discuss.

**Option permutations** `perm_ABCD`, `perm_BCDA`, `perm_CDAB`, `perm_DABC`: the option *texts* are
cyclically rotated into slots A–D (slot k of the new prompt shows the option that was in slot `order[k]`);
the letters in the prompt stay A–D; the gold letter must be remapped (your job, Part 3). `perm_ABCD` is the
identity and equals your Part 2 run.

### 2.5 Scoring protocols

* **LETTER** — one forward pass; log-softmax over the *full* vocabulary at the last prompt position; the four
  scores are the log-probs of the tokens `" A"`, `" B"`, `" C"`, `" D"` (each is a single token for all three
  tokenizers — `harness.model.letter_token_ids` asserts it). `pred = argmax`.
* **CONT** — for each option, the log-likelihood of the continuation `" " + option_text` after the prompt:
  the sum of the continuation tokens' log-probs *only*. From the same four forward passes report
  `CONT` (raw sum), `CONT_TOKNORM` (sum / number of continuation tokens) and `CONT_CHARNORM`
  (sum / number of characters in the option text). `pred = argmax` under each.
* **GEN** — greedy decoding (`do_sample=False`, `max_new_tokens=8`), then *your* parser maps the string to
  A–D or `null`. Parse-failure rate is a required metric (even when it turns out to be zero).

### 2.6 Numerics and seeds

* dtype `float16` on `mps`/`cuda`, `float32` on `cpu` (auto; override with `--dtype fp32` if MPS misbehaves;
  `bf16` is accepted for graders' GPUs only).
* Batch size 8 by default, halved automatically on out-of-memory (minimum 1). Left padding with an attention
  mask; `harness/model.py` handles it.
* All scoring is deterministic (argmax / greedy). Seeds only matter for the bootstrap:
  `numpy.random.default_rng(0)`, **B = 2000** resamples over items, percentile 95 % CI, resampling scheme
  fixed in `harness/stats.py`. Paired bootstraps draw *one* index set and apply it to both arrays.
* **LITE mode**: every Global-MMLU-Lite run uses the fixed 200-item subset (first 100 CS + first 100 CA in
  `sample_id` order), identical for all LITE students. The smoke test switches it on when your machine's
  *projected* core compute exceeds 150 minutes (in practice: Intel/CPU-only Macs), and writes the decision to
  `smoke_ok.json`, which every `python -m harness run` reads. Override with `python smoke_test.py --force-lite`
  or `--no-lite-auto`; never mix modes within one submission. Bands and rubric apply unchanged; your CIs are
  wider — say so.

---

## 3. Tasks

### Part 0 — Setup and smoke test (0 points, required gate; `smoke_ok.json` due Tuesday 23:59)

```
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python smoke_test.py --student-id <your id>
```

The smoke test prints your torch version, chip, device and memory; downloads the model (~1 GB) and the `en`
config; checks one reference item against instructor log-probs (this catches a tampered template or a broken
backend); measures your throughput and projects the total core compute; and writes `smoke_ok.json`, which
also decides LITE mode. Upload `smoke_ok.json` to the LMS by **Tuesday 23:59** (Day 2; release day is Day 1).
It is not graded, but it is
how the TA learns who is on CPU, and missing it forfeits the 24-hour schema-fix grace window at the end.

### Part 1 — Three ways to score a multiple-choice question (15 points)

Implement `score_letter`, `score_cont` (three variants from one run), `score_gen` and `parse_gen` in
`harness/scorers.py` so that `python -m harness run … --scorer LETTER|CONT|GEN` works. Pass `pytest tests/`
(including the model tests; run `pytest tests --no-model` while iterating). **Before running anything larger**,
run `python -m harness check-first20 --student-id <id>`: it scores the first 20 `en` items with your
`score_letter` and compares them with `reference/en_first20.jsonl` (≥ 19/20 predictions must match; if not,
you have the classic `" A"`-vs-`"A"` or wrong-position bug). Cohen's κ for (b) below is `cohens_kappa` in
`harness/stats.py`, so implement that function now. A `--scorer CONT` run writes **three** prediction files
(`CONT`, `CONT_TOKNORM`, `CONT_CHARNORM`); raw `CONT` accuracy is expected to be far below LETTER.

Run all three scorers on `en`, `zh`, `sw` with `v1_en`:

```
python -m harness run --part 1 --model qwen2.5 --lang en --scorer LETTER --variant v1_en --student-id <id>
python -m harness run --part 1 --model qwen2.5 --lang en --scorer CONT   --variant v1_en --student-id <id>
python -m harness run --part 1 --model qwen2.5 --lang en --scorer GEN    --variant v1_en --student-id <id>
… same for zh and sw
```

Deliver in the report: **(a)** a table, rows = {LETTER, CONT, CONT_TOKNORM, CONT_CHARNORM, GEN}, columns =
{en, zh, sw}, accuracy and, for GEN, parse-failure rate; **(b)** Cohen's κ between LETTER and GEN predictions
per language (GEN `null` is its own class); **(c)** **8 `en` items** where LETTER and CONT disagree, each
labelled with one category from the fixed taxonomy {`length bias`, `option-text prior`,
`formatting/tokenisation`, `near-tie`, `other`} and one sentence of justification (English only — you are not
expected to read zh or sw); **(d)** ≤ ¾ page: which scorer would you recommend for a 0.5 B model, and why does
the choice matter more for sw than for en? Use your own numbers (normalisation flips, the GEN parse-failure rate — which may well be zero; say what that does and does not tell you — and κ).

Compute: 1,200 items × (1 LETTER + 4 CONT + ~2.5 GEN forward-equivalents) ≈ 9,000 forwards.

### Part 2 — Multilingual sweep, CS/CA split, confidence intervals, paired comparison with `en` (20 points)

Run LETTER, `v1_en`, on all eight languages with `--part 2` — a separate `p2_…` file for every language,
`en` included (the Part 1 `en` file is used only for the Part 1 tables). Implement **all of
`harness/stats.py` in numpy only** (scipy/statsmodels are not accepted) and check yourself against `reference/stats_toy.json`
(`pytest tests/test_stats.py`).

Deliver

* `results.csv` — rows for every (part, model, lang, scorer, prompt_variant, subset ∈ {all, CS, CA}) with
  n, accuracy, 95 % CI (all your runs, not just Part 2);
* `figures/accuracy_by_language.png` — accuracy with CI bars, languages in the §2.3 order, dashed line at 0.25;
* `paired_vs_en.csv` — for each language L ≠ en: acc(L) − acc(en) with a *paired* bootstrap CI, the discordant
  counts `n_en_only_correct`, `n_lang_only_correct`, and the per-item `agreement_rate` (fraction of items with
  the same predicted letter in en and L);
* `cs_ca_gap.csv` — per language the CS − CA gap with an *unpaired* CI (CS and CA are disjoint item sets;
  columns below);
* written analysis (≤ 1 page): (i) which languages are statistically distinguishable from `en` under the
  paired CI and which are not; explain why an n = 400 CI is about ±5 points and an n = 200 CI about ±7
  (√(p(1−p)/n)); (ii) the CS − CA gap per language — for how many languages is it detectable at all at
  200 vs 200? (expect: almost none — that is the lesson); (iii) interpret the en-vs-L agreement rate and
  discordant counts as a translation-consistency measure, contrasting a supported (`de`) and an unsupported
  (`yo`) language.

Compute: 3,200 forwards.

### Part 3 — Option-position and prompt-language sensitivity (15 points)

Complete `harness/prompts.py::permute` (the gold-letter remapping is deliberately left as a TODO; the
semantics are in its docstring and `tests/test_prompts.py`). Run LETTER under the four permutations on
`en`, `zh`, `hi`, `sw` (`--part 3 --variant perm_BCDA` etc.; `perm_ABCD` may be produced by re-running or by
letting `analysis.py` reuse your Part 2 file) and the frozen `v2_native` template on `zh`, `hi`, `sw`.

Deliver

* `permutations.csv` — per language: accuracy under each permutation, mean, std across the four, and the
  **consistency rate** (fraction of items whose predicted option *text* is the same under all four
  permutations), plus the pooled letter frequencies `frac_pred_A..D`;
* `figures/position_bias.png` — per language the fraction of predictions on each letter pooled over the four
  permutations (so the gold distribution is uniform by construction), against the 0.25 line;
* `native_instruction.csv` — per language acc(`v2_native`) − acc(`v1_en`) with paired CI and discordant counts;
* written analysis (≤ ¾ page): is the permutation std smaller or larger than the half-width of your Part 2
  bootstrap CI, and what does that imply about reporting one number from one option order? Which languages
  show position bias (a letter chosen > 35 % of the time)? Does the native instruction help, hurt, or is it
  undetectable — noting that the supported-vs-unsupported contrast rests on `sw` alone, which is near chance?
  Interpret `sw`'s consistency rate: a model at chance that is nonetheless *consistent* across permutations is
  answering from an option-text prior; one that is *inconsistent* is answering from position.

`check_results.py` flags any permutation run near 0.25 while the unpermuted run is above 0.35
("gold not remapped").

Compute: 4 languages × 3 new permutations × 400 + 3 × 400 native = 6,000 forwards.

### Part 4 — Tokenizer fertility and its cost (10 points; no model inference)

Implement `tokenizer_stats.py` (skeleton given). For the **qwen2.5, qwen3 and smollm2 tokenizers** and each of
the 8 languages, over the 400 items (text = question + the four options, newline-joined): `mean_tokens`,
`tokens_per_char` (characters = code points after NFC normalisation), `tokens_per_word` (whitespace-split;
`NaN` for zh and say why) and `frac_partial_char_tokens` — the fraction of tokens whose decoded string is
empty or contains U+FFFD, i.e. tokens that do not by themselves form a complete Unicode character (all three
tokenizers are byte-level BPE; this measures how often a character is split across byte tokens).

Deliver `tokenizer.csv` and `figures/fertility_vs_accuracy.png` (qwen2.5 `mean_tokens` on x, Part 2 accuracy
on y, eight labelled points, **no correlation coefficient** — n = 8 is descriptive only). Written analysis
(≤ ½ page): rank languages by fertility; explain the mechanism (byte-level BPE on non-Latin scripts; combining
diacritics in yo); estimate the relative wall-clock cost of a yo vs en evaluation from token counts (linear is
an acceptable approximation; note that attention is super-linear); and use the **id vs sw** pair — same
script, comparable fertility, opposite official support — to argue whether fertility alone explains the accuracy
ordering or whether pre-training exposure must be invoked. (You will notice that qwen2.5 and qwen3 share a
vocabulary — one sentence on what that implies for Part 5 is welcome but not graded.)

### Part 5 — Paired comparison with a second model (15 points)

Run the identical Part 2 protocol with **Qwen3-0.6B** (`--part 5 --model qwen3`) on all eight languages.
Qwen3 is newer, claims 119 languages, and has 50 % more parameters; does that move `sw`/`yo` off the floor,
and does it change position bias? Load one model at a time; reuse your Part 2 predictions.

Deliver `comparison.csv` — per language: `acc_qwen25`, `acc_qwen3`, `diff` **= acc_qwen3 − acc_qwen25**
(positive means Qwen3 is better; every `diff` column in this exam is *second − first*), the **paired** bootstrap CI,
whether the two *independent* 95 % CIs overlap (`unpaired_overlap`), the discordant counts, and `mdd_95`, the
minimum detectable paired difference: under the null, var(diff) ≈ (b + c)/n², so
`mdd_95 ≈ 1.96·√(b + c)/n` with b, c the discordant counts (show the derivation in the report). Written
analysis (≤ ¾ page): where is the difference real, where is either model at chance (CI includes 0.25), and —
required — where would the naive CI-overlap test have led you to a different conclusion than the paired test
(name a language, or show none differ and explain why using the discordant counts). Name **one** confound
(parameter count, chat template, training data) and a concrete, executable control for it.

Compute: 3,200 forwards with a 0.6 B model.

### Part 6 — Report, reflection, reproducibility (25 points)

* **Report** (`report.pdf`, ≤ 4 pages at 11 pt, references and an appendix of extra tables excluded), with
  these section headings: **Setup** (hardware, backend, dtype, LITE yes/no, wall-clock per part, model and
  dataset revisions, the `v1_en` prompt verbatim); **Results** (the Part 1 table, the Part 2 figure,
  `comparison.csv` rendered; every table carries CIs); **Analyses** (the written answers for Parts 1–5 under
  sub-headings, referencing tables/figures by ID); **Threats to validity** (at least three, each with a
  mitigation you could actually run); **Three rules for practitioners**, each traceable to one of your own
  results by table/figure ID; and one sentence on what you did *not* conclude. (15 points)
* **Reflection** (`reflection.pdf`, ≤ 1 page). The harness writes `probe.jsonl`: 20 items chosen
  deterministically from your student id (`python -m harness probe-ids --student-id <id>` lists them; in LITE
  mode only those that fall in the LITE subset appear), with the LETTER score vectors from every `v1_en`
  (model, language) run. Discuss **three** of them by `sample_id` with their actual numbers (e.g. "item X was
  correct in de and en, wrong in yo and sw; the yo prompt was 2.6× as many tokens; under `perm_CDAB` it
  flipped to …" — look up non-LETTER or permutation facts in your `predictions/` files), and
  describe one bug or surprise you hit and how you found it. (5 points)
* **Reproducibility** (5 points). README with exactly one command per part; `requirements.txt`;
  harness-generated `run_log.txt`; `validate_predictions.py` and `check_results.py` pass on every file
  (`make check`); `analysis.py` regenerates every CSV and figure from `predictions/` without loading a model.
  The grader re-runs your Part 2 `en` command: ≥ 99 % per-item agreement with your submitted file and the
  same accuracy to three decimals (backend differences can flip near-tie items; byte identity is not required).

### Stretch tasks (optional; bonus capped at 10; attempt only after Parts 0–6 validate)

* **S1 Belebele (5)** — `--benchmark belebele --n 300` on `eng_Latn`, `zho_Hans`, `hin_Deva`, `swh_Latn`
  (LETTER; prompts ≈ 250–700 tokens). Does the cross-language ordering match Global-MMLU-Lite? Do
  passage-grounded questions narrow the gap for Swahili?
* **S2 MGSM (5)** — `juletxara/mgsm`, `en` and `sw`, 100 items each, 4-shot from `train`, greedy,
  `max_new_tokens=256` (loader in `harness/data.py`; write your own runner). Robust number parser with three
  unit tests; accuracy and parse-failure rate; why are generative benchmarks a poor fit for sub-1B models in
  low-resource languages? Skip on Intel.
* **S3 SmolLM2-360M (4)** — repeat Part 2 with `--model smollm2` (English-centric pre-training). Compare with
  a paired bootstrap. What does its position bias look like, and what does that say about LETTER scoring for
  models that were never trained on the A/B/C/D format?
* **S4 mlx-lm throughput (3, Apple Silicon only)** — port LETTER to `mlx-lm`; report items/s for both stacks
  and per-item agreement on `en` and `zh`; explain disagreements. mlx is never allowed for graded core runs.
* **S5 XCOPA (3)** — `it`, `zh`, `sw` (500 items): continuation scoring of the two choices with and without
  per-token normalisation, plus a *prompt-free* baseline (score the choices without the premise) and a
  majority-class baseline. How much context does the model actually use?

---

## 4. Deliverables

One zip (or private git URL) with the layout below. File names and schemas are checked by scripts.

```
README.md                 one command per part; your letter-token verification note; LITE statement
requirements.txt          as shipped, plus anything you added
run_log.txt               auto-generated by the harness (never edit)
smoke_ok.json             from Part 0 (also uploaded to the LMS by Day 2)
integrity.txt             signed declaration (template: integrity_template.txt)
harness/                  starter code + your implementations
tests/                    instructor tests + at least one test of your own
predictions/*.jsonl       one file per run:  p{part}_{model}_{lang}_{scorer}_{variant}.jsonl
probe.jsonl               auto-generated; do not edit
results.csv  paired_vs_en.csv  cs_ca_gap.csv  permutations.csv  native_instruction.csv  tokenizer.csv  comparison.csv
figures/accuracy_by_language.png  figures/position_bias.png  figures/fertility_vs_accuracy.png
analysis.py  tokenizer_stats.py
report.pdf  reflection.pdf
```

**Prediction records** (`harness/schema.py`, one JSON object per line): `part, model, model_sha, benchmark,
lang, sample_id, cs_label, scorer, prompt_variant, gold` (the correct letter *as shown in the prompt*, i.e.
after permutation), `gold_text_id` (0–3, index of the correct option in the original dataset order), `pred`
(A–D, or `null` only for a GEN parse failure), `scores` (4 log-probs in prompt letter order; `null` for GEN),
`raw_generation` (GEN only), `n_prompt_tokens, lite_mode, student_id`. The CLI writes these for you;
`validate_predictions.py` checks them (400 lines per file, or 200 in LITE mode; unique ids; gold consistent
with the dataset and the permutation; `pred == argmax(scores)`; allowed model).

**CSV columns**

* `results.csv`: `part, model, benchmark, lang, scorer, prompt_variant, subset, n, accuracy, ci_low, ci_high, parse_fail_rate, wall_clock_sec`
  (`model` is the full Hub id exactly as in the prediction records; `diff` columns everywhere are *second − first*)
* `paired_vs_en.csv`: `lang, n, acc_en, acc_lang, diff, paired_ci_low, paired_ci_high, n_en_only_correct, n_lang_only_correct, agreement_rate`
* `cs_ca_gap.csv`: `lang, n_cs, n_ca, acc_cs, acc_ca, gap, unpaired_ci_low, unpaired_ci_high, detectable`
* `permutations.csv`: `lang, acc_ABCD, acc_BCDA, acc_CDAB, acc_DABC, mean, std, consistency_rate, frac_pred_A, frac_pred_B, frac_pred_C, frac_pred_D`
* `native_instruction.csv`: `lang, acc_v1_en, acc_v2_native, diff, paired_ci_low, paired_ci_high, n_v1_only_correct, n_v2_only_correct`
* `tokenizer.csv`: `tokenizer, lang, n_items, mean_tokens, tokens_per_char, tokens_per_word, frac_partial_char_tokens`
* `comparison.csv`: `lang, n, acc_qwen25, acc_qwen3, diff, paired_ci_low, paired_ci_high, unpaired_overlap, n_qwen25_only_correct, n_qwen3_only_correct, mdd_95`

`check_results.py` recomputes every accuracy and count in these CSVs from `predictions/` and requires a match to
0.001 (delete any `--n`-truncated debug files from `predictions/` first); the grader recomputes the CIs with
the fixed scheme and requires agreement within 0.01.

---

## 5. Suggested timeline (release Monday 09:00; deadline the following Monday 23:59)

| Day | Do | Compute (Apple Silicon) | Your time |
|---|---|---|---|
| Mon (Day 1) | Read this; install; `python smoke_test.py`; upload `smoke_ok.json` | 3 min | 1 h |
| Tue (Day 2) | Part 1: LETTER; tests; `check-first20`; run LETTER on en/zh/sw; start CONT. **`smoke_ok.json` due 23:59** | 3 min | 4 h |
| Wed | Part 1: finish CONT + GEN + parser; run; κ; disagreement cases | 15 min | 4 h |
| Thu | Part 2: sweep; `stats.py`; `results.csv`, figure, `paired_vs_en.csv`. Part 4 tokenizer script | 10 min | 4 h |
| Fri | Part 3: `permute`; permutation + native runs; tables and figure (Intel: start in the evening) | 12 min | 3 h |
| Sat | Part 5: Qwen3 run; `comparison.csv`, MDD. `analysis.py` end to end; `make check` | 6 min | 3 h |
| Sun | Report and reflection. Stretch only if everything validates | 0 | 4 h |
| Mon | Final `make check`; submit by 23:59 | | ½ h |

The 24-hour grace period after the deadline applies **only** to fixing validator-flagged schema errors, and
only for students who uploaded `smoke_ok.json` on time. A random 20 % of students will be asked to a 5-minute
oral check on their own code and one probe item during the following week.

---

## 6. Starter code: provided vs yours

**Provided (do not modify unless the file says so):** `harness/data.py` (pinned loaders, LITE subset,
probe ids), `harness/prompts.py` (templates; only `permute` is yours), `harness/model.py` (device/dtype
selection, loading, `apply_chat`, batched `forward_last_logits`, `forward_token_logprobs`,
`generate_greedy`, `letter_token_ids`), `harness/schema.py`, `harness/cli.py`, `smoke_test.py`,
`validate_predictions.py`, `check_results.py`, `tests/`, `reference/` (`en_first20.jsonl`,
`smoke_reference.json`, `stats_toy.json`), `PLAUSIBILITY_BANDS.md`, `TROUBLESHOOTING.md`,
`colab/fallback.ipynb`, `Makefile`.

**Yours:** `harness/scorers.py` (all four functions), `harness/prompts.py::permute`, all of
`harness/stats.py`, `tokenizer_stats.py`, `analysis.py` (skeletons give you the CLI and column names), one
unit test of your own, README, report, reflection. You may add CLI flags and helper files; you may not change
the schema, the frozen templates, the revisions or the resampling scheme.

---

## 7. Hardware and setup

**Apple Silicon (M1–M4, 8–16 GB).** Default path: `float16` on `mps`, batch 8. Downloads: Qwen2.5-0.5B
≈ 1.0 GB, Qwen3-0.6B ≈ 1.5 GB, SmolLM2 ≈ 0.7 GB (stretch only), datasets < 10 MB. Peak memory ≈ 2.5 GB —
close memory-hungry apps on an 8 GB machine. If `run_log.txt` shows fewer than ~3 items/s, check the `device`
column: you are probably on CPU.

**Intel Macs (CPU only).** PyTorch's last macOS x86_64 release is **2.2.2** (Python ≤ 3.11):
`pip install torch==2.2.2` before `pip install -r requirements.txt`. `float32`, ~5–10× slower; the smoke
test puts you in LITE mode automatically (no penalty). Expected total 3–4 h; start Part 3 in the evening and
let it run. Runs checkpoint every 50 items and resume where they stopped (`.partial` files), so a crash costs
at most a minute. If the install fails, use the Colab fallback (item 4 below) — same commands, same outputs.

**If you are stuck** — this path still lets you complete the whole exam:

1. NaN/garbage scores on MPS → `--dtype fp32` (about 2× slower).
2. Out of memory → `--batch-size 4` (the harness also halves automatically).
3. Hub unreachable → pre-download once (`TROUBLESHOOTING.md`) and set `HF_HUB_OFFLINE=1`.
4. Machine genuinely too slow or broken → `colab/fallback.ipynb` runs the identical CLI on the Colab free tier;
   copy `predictions/`, `run_log.txt` and `probe.jsonl` back and continue locally. Declare it in *Setup*;
   no penalty.
5. Still stuck by Day 2 → TA office hours (twice daily on Days 1–2). Bring `smoke_ok.json` and `run_log.txt`.

Whatever path you take, the grader needs only your `predictions/` files, CSVs and one reproducible `en` command.

---

## 8. Academic integrity and AI-tool use

> This is individual work. You may use AI coding assistants (ChatGPT, Claude, Copilot, …), search engines and
> public documentation for environment setup, boilerplate, debugging and language help. You must write the
> scorers, the gold remapping, `stats.py`, `analysis.py`, the report and the reflection yourself. Every AI
> tool you used, and what for, must be listed in `integrity.txt`, which you sign. You must be able to explain
> any line of your code or any sentence of your report in a 5-minute oral check. Sharing code, prompts,
> prediction files or results with classmates, submitting AI-generated analysis as your own, or reporting any
> number that cannot be regenerated from your own `predictions/` files by your own `analysis.py` is academic
> misconduct: zero for the exam and referral under university policy. Using the starter code and the public
> model/dataset documentation is of course allowed.

Grading criteria, point by point, are in `RUBRIC.md`. Expected result ranges are published in
`PLAUSIBILITY_BANDS.md`: being outside a band with a documented cause costs at most one point; being outside a
band with no diagnosis is what costs code points.
