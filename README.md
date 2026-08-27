# Take-home exam: "Small Models, Many Languages"

A one-week, individual take-home exam for an NLP course. Students evaluate a sub-1B language model
(**Qwen2.5-0.5B-Instruct**, contrast **Qwen3-0.6B**) on the parallel eight-language
**Global-MMLU-Lite** benchmark using only a MacBook (Apple Silicon via PyTorch/MPS; Intel Macs via a fixed
CPU "LITE" subset), and learn what multilingual evaluation of tiny models actually involves: scoring
protocols, chance floors, option-position bias, tokenizer fertility, paired statistics on parallel items.

```
student/        everything students receive (zip it with instructor/make_student_zip.sh)
  ASSIGNMENT.md         the handout (tasks, frozen protocol, deliverables, timeline, policy)
  RUBRIC.md             point-by-point rubric (also shown to students)
  PLAUSIBILITY_BANDS.md expected result ranges measured with the reference solution
  TROUBLESHOOTING.md    MPS / CPU / Hub / resume issues
  harness/              Python package: data, prompts, model helpers, schema, CLI  (+ stubs to implement)
  smoke_test.py         Part 0 gate; measures the machine, writes smoke_ok.json, decides LITE mode
  validate_predictions.py, check_results.py   the same checks the grader runs
  analysis.py, tokenizer_stats.py             skeletons students complete
  tests/                instructor pytest suite
  reference/            en_first20.jsonl, smoke_reference.json, stats_toy.json
  colab/fallback.ipynb  identical CLI on Colab, for students whose Mac cannot run it
instructor/     never distributed
  INSTRUCTOR_NOTES.md   measured reference results, pitfalls, anti-copying, grading workflow, pre-release checklist
  solution/             reference implementations (scorers, stats, permute, analysis.py, tokenizer_stats.py)
  grade.py              held-back grader: validators, CSV recomputation, bands, probe/fabrication checks, rubric sheet
  reference_results/    predictions + run log from the reference run, bands.json, tokenizer.csv
  run_reference.sh      regenerates every reference result with the frozen protocol
  make_reference_files.py, make_stats_toy.py
```

## Before release

1. `bash instructor/run_reference.sh` on a GPU box or an Apple-Silicon Mac (≈ 15 min on a GPU) to refresh
   `instructor/reference_results/` and `student/reference/`; then run `instructor/make_reference_files.py`
   once on an Apple-Silicon Mac (`--device mps --dtype fp16`) so the fp16 reference vectors come from the
   backend students will use.
2. Have the zh/hi/sw native templates in `student/harness/prompts.py` checked by fluent speakers.
3. `bash instructor/make_student_zip.sh` — refuses to run if any reference-solution file is inside `student/`.
4. Work through `instructor/INSTRUCTOR_NOTES.md` § "Pre-release checklist".

## Grading a submission

```
python instructor/grade.py <unzipped_submission_dir> --rerun-en --out grade_report.md
```
