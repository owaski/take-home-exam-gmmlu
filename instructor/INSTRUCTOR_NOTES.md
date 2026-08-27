# Instructor notes — "Small Models, Many Languages"

Never distribute this directory. Student-facing material is `student/` only.

## 1. What the reference run shows (and which lesson each number carries)

All numbers: reference solution, frozen protocol, fp16 on an NVIDIA GPU (CPU fp32 reproduces every
prediction to within one near-tie item). Raw files: `reference_results/predictions/`, CSVs next to them,
figures in `reference_results/figures/`. Student-facing subset: `student/PLAUSIBILITY_BANDS.md`.

**Part 1 — scoring method changes the number.** Qwen2.5-0.5B on `en`: LETTER 0.485, GEN 0.410,
CONT_TOKNORM 0.348, CONT_CHARNORM 0.328, raw CONT 0.302. The raw continuation score is dominated by
option length (longer options accumulate more negative log-prob); normalisation recovers part of it but
never reaches LETTER, because the model has been instruction-tuned to *emit* a letter, not to reproduce
option text. GEN parse failure is 0 % in every language (the model always produces a bare letter), yet GEN is
7–11 points below LETTER on en/zh — the generated letter is the argmax of the *first* token under the chat
template, which is not always the letter with the highest log-prob after `Answer:`. κ(LETTER, GEN) drops
from 0.41 (en) to 0.07 (sw): in an unsupported language the two protocols are essentially uncorrelated.
Good students will say "for a tiny model, report LETTER, and report the others to show fragility".

**Part 2 — chance floors and paired CIs.** LETTER accuracy: en 0.485 · zh 0.470 · de 0.428 · id 0.385 ·
ar 0.330 · hi 0.310 · yo 0.278 · sw 0.260. Every non-English language except zh (−0.015, CI [−0.065, 0.035])
is distinguishable from en under the paired CI; de (−0.057, CI [−0.110, 0.000]) is borderline — a nice case to
discuss. The en-vs-L agreement rate tracks support: de 0.55, zh 0.63, id 0.60 vs sw 0.39, yo 0.43 (0.25 would
be independent guessing). **CS − CA gap:** detectable only for `en` (−0.10, unpaired CI [−0.20, −0.005]);
for every other language the 200-vs-200 CI (± ~0.10) includes zero. This is the intended lesson: the
cultural-sensitivity split cannot be resolved at this n with a model this size.

**Part 3 — position and prompt noise.** Permutation std is 1.3–1.9 points, i.e. *smaller* than the ± 4.8
bootstrap half-width — so option order is not the dominant noise source at n = 400, but it is the same order
of magnitude as many "improvements" reported in papers. Position bias is dramatic where the model is lost:
pooled over permutations, hi picks B 54 % of the time and sw picks B 44 % / D 1.5 %. Consistency rate
(same option text under all four orders): en 0.33, zh 0.29, hi 0.05, **sw 0.02** — the sw predictions are
almost entirely position-driven, which is why sw sits at chance *and* is inconsistent. Native instructions:
zh +0.025, hi +0.010, sw +0.018, every paired CI includes 0 (the instruction language does not matter when
the question itself is already in the target language). Note the hi native value was re-measured after the
template was polished by a Hindi reviewer; regenerate the band if you edit the template again.

**Part 4 — fertility.** Mean tokens per item (qwen2.5 = qwen3 vocabulary): en 86 · zh 83 · ar 129 · id 129
· de 136 · sw 164 · yo 254 · hi 361. `frac_partial_char_tokens` is 0.51 for Hindi (half of all tokens are
byte fragments of a Devanagari character) vs 0.07 for yo and ≤ 0.03 elsewhere. The **id vs sw** pair
(129 vs 164 tokens, both Latin, supported vs unsupported, 0.385 vs 0.260) shows fertility alone does not
explain accuracy; zh (lowest fertility, second-best accuracy) and hi (highest fertility, third-worst) show
the correlation is real but confounded with pre-training exposure. SmolLM2's vocabulary is far worse on
zh (215) and ar (319).

**Part 5 — paired comparison.** Qwen3-0.6B vs Qwen2.5-0.5B (diff = Qwen3 − Qwen2.5): id **+0.085** (paired CI
[+0.030, +0.142]), ar +0.032, hi +0.025, de +0.022, zh +0.020, en +0.017, sw 0.000, yo 0.000. Only `id` is
significant; the *unpaired* CIs overlap for **every** language including id — the exact case where the
naive CI-overlap test and the paired test disagree. mdd_95 ≈ 0.053–0.059 everywhere (discordant rate
≈ 30 %): differences below ~5.5 points are undetectable at n = 400 no matter how they are tested. Qwen3 does
**not** move sw/yo off the floor despite "119 languages", but it removes most of the position bias (its
letter distribution is near-uniform in every language) — students who look at `probe.jsonl` score vectors
will see it.

**Stretch.** Belebele (300 items): eng 0.580, zho 0.550, hin 0.323, swh 0.297 — same ordering, passage
grounding does not rescue Swahili. SmolLM2-360M-Instruct: 0.215–0.270 in every language *including English*
— an English-centric model that was never trained on the A/B/C/D format cannot be letter-scored at all
(strong C/D bias); a good S3 write-up says LETTER is only meaningful for models that answer in that format.

## 2. Expected environment behaviour

* **Measured here:** CUDA fp16 (reference), CPU fp32 (validation of every path, tests, grader). CPU
  throughput on an 8-core server: LETTER ≈ 4–6 items/s, CONT ≈ 1.1 items/s, GEN ≈ 4.7 items/s (fp32).
* **Not measured here — verify on real hardware before release:** Apple-Silicon MPS fp16 (throughput,
  NaN behaviour on the student's macOS/torch combination, `generate` on MPS), and the Intel-Mac install
  path (`torch==2.2.2`, Python ≤ 3.11). Run `bash instructor/run_reference.sh` on an M-series Mac and then
  `python instructor/make_reference_files.py --device mps --dtype fp16` so the fp16 reference vectors in
  `student/reference/` come from the backend students actually use (the tolerance is 0.05 nats; CUDA vs
  MPS fp16 differences are expected to be ~0.01).
* **Compute budget.** Core forward-equivalents ≈ 21,400. At 6 items/s (this server's CPU) that is ~60 min;
  a MacBook Air CPU is slower (expect 1–3 items/s → LITE mode kicks in above 150 projected minutes). MPS
  on M1/M2 should land in the 5–15 items/s range for LETTER — confirm.
* Peak memory: ~2.5 GB with Qwen2.5 in fp16; Qwen3 ~3 GB. One model per process (the CLI enforces it).

## 3. Common pitfalls (and where the package catches them)

| Pitfall | Symptom | Caught by |
|---|---|---|
| `"A"` instead of `" A"` after `Answer:` | en accuracy ≈ 0.25 | `check-first20`, `test_score_letter_reproduces_reference`, band |
| Wrong position with left padding | batch ≠ single, accuracy falls with batch size | `test_batch_equals_single` |
| Extra BOS (`add_special_tokens=True`) | small systematic score offsets | `check-first20` max-deviation |
| CONT sums prompt tokens too | CONT ≈ chance, scores ≈ −200 | tests + band |
| Gold not remapped under permutation | perm accuracy ≈ 0.25 while ABCD ≈ 0.48 | `check_results.py` GOLD-NOT-REMAPPED |
| Bootstrap over languages/subjects; normal approximation; scipy | CI mismatch > 0.01 | grader recomputation; `test_stats` vs `stats_toy.json` |
| CS vs CA treated as paired | wrong CI width; rubric P2-stats | manual |
| Unpaired CI overlap used as a test | wrong Part 5 conclusion | rubric P5-analysis requires the contrast |
| Both models loaded at once on 8 GB | OOM | CLI loads one model per invocation |
| Qwen3 `<think>` block in the prompt | prompt hash mismatch / odd scores | `apply_chat` asserts; smoke test hash |
| `--n` debug files left in `predictions/` | validator line-count FAIL | validator message says so |
| Mixed LITE/full files | wrong n | validator cross-file + `smoke_ok.json` check |

## 4. Anti-copying and fabrication detection

* **Per-student probe.** `probe_ids(student_id)` picks 20 items from the full list; the reflection must cite
  three by id with numbers matching `probe.jsonl`. Copied reflections show different ids. Do **not** treat
  identical score vectors across students as copying — deterministic forward passes on the same chip/dtype
  legitimately agree to many decimals; *different* chips disagree in the 3rd–4th decimal.
* **Recomputation.** `grade.py` regenerates every CSV from `predictions/` with the reference `analysis.py`
  and diffs; recomputes accuracy per file; optionally re-runs the Part 2 `en` command (`--rerun-en`) and
  checks ≥ 99 % item agreement; cross-checks `run_log.txt` throughput against the device in `smoke_ok.json`
  (a 3,200-item run logged at 2 minutes on a CPU is a fabrication signal, as is a log with identical
  wall-clock across runs or no OOM halvings on an 8 GB machine that claims batch 8 for Part 3).
* **Fabricated predictions.** Fabricated files fail `pred == argmax(scores)` or have implausibly smooth
  score vectors; real LETTER vectors for the primary model correlate > 0.95 per item with the reference on
  `en`. Flag: agreement < 0.90 with the reference *while* accuracy is inside the band.
* **Report claims** are checked against the CSVs (−1 each, max −5; systematic → referral).
* MOSS-style similarity on `scorers.py` / `stats.py`; 20 % random 5-minute vivas on own code + one probe item.

## 5. Grading workflow (~30 min per submission)

1. `python instructor/grade.py <submission_dir> --rerun-en --out grade_report.md` (≈ 2–10 min, most of it
   the `en` re-run; drop `--rerun-en` on a first pass). Read **FLAGS** first, then the pre-filled rubric.
2. Report against the checklist (sections, claim–evidence linkage, three threats with executable
   mitigations, three practitioner rules): ≈ 12 min.
3. Reflection: confirm the three probe items match `probe.jsonl` (listed by the grader): ≈ 3 min.
4. Fill the manual analysis lines (P1 taxonomy, P2/P3/P4/P5 analysis): ≈ 5 min.

Bands are soft: outside with a documented cause −1 at most; outside with no diagnosis is what costs code
points. The grader tolerances are 0.045 (full) / 0.065 (LITE) around the reference accuracy, 1.5 points on
the permutation mean, 10 % on tokenizer values, 0.01 on CIs.

## 6. Pre-release checklist

- [ ] Run `bash instructor/run_reference.sh` on an Apple-Silicon Mac (or at least
      `python instructor/make_reference_files.py --device mps --dtype fp16` in a solution overlay) and confirm
      `python smoke_test.py` passes on MPS with the shipped `student/reference/smoke_reference.json`.
- [ ] Install on an Intel Mac (`torch==2.2.2`, Python 3.11) and run the smoke test; confirm LITE triggers.
- [ ] Native templates: zh and sw were reviewed as fine by fluent reviewers; hi was polished — have a native
      Hindi speaker glance at `NATIVE_TEMPLATES["hi"]` once more. Any edit → re-run that language's
      `v2_native` reference and regenerate `bands.json` + `PLAUSIBILITY_BANDS.md`.
- [ ] `bash instructor/make_student_zip.sh` (refuses if a reference-solution marker is inside `student/`).
- [ ] Put the LMS/office-hour specifics (Day 1 date, upload link, TA hours) into `ASSIGNMENT.md` §3 Part 0, §5, §7.
- [ ] Mirror the model/dataset snapshots on the course server for students behind a broken Hub connection
      (`TROUBLESHOOTING.md` §3 explains the offline path).
- [ ] Decide the Colab policy: the notebook works but produces `run_log.txt` lines with a non-Mac chip;
      the grader flags nothing for that, but ask students to declare it in *Setup*.

## 7. Regenerating everything

```
bash instructor/run_reference.sh            # all graded runs + stretch + LITE; then CSVs, figures, bands.json,
                                            # probe.jsonl and student/reference/ are refreshed automatically
python instructor/make_bands.py instructor/reference_results instructor/reference_results/lite instructor/reference_results
                                            # bands.json + probe.jsonl only (after hand-editing CSVs)
python instructor/make_stats_toy.py         # student/reference/stats_toy.json
```
`bands.json` keys are `alias|lang|scorer|variant` (± 0.045) and `lite:alias|lang|scorer|variant` (± 0.065).
Then update the numbers quoted in `student/PLAUSIBILITY_BANDS.md` and §1 above by hand.
