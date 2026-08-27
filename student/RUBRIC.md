# Rubric — Small Models, Many Languages (100 points + ≤ 10 bonus)

Automatic lines are scored by the grader's script from your `predictions/`, CSVs and `run_log.txt`; manual
lines are scored from the report against the checklist below. Two graders should agree within ~5 points.
"Band" refers to `PLAUSIBILITY_BANDS.md`.

| # | Criterion | Pts | Full credit | Partial | Zero |
|---|---|---|---|---|---|
| P1-code | Three scorers | 8 | All instructor tests pass plus ≥ 1 test of your own; `en_first20` self-check ≥ 19/20; LETTER and CONT `en` inside band; README documents the single-token letter check (auto: LETTER 3, CONT 3, GEN parser tests 2) | −1 per failed instructor test; a scorer or a CONT variant missing (−1 each); out of band with a documented cause (−1) | No scorer runs; or LETTER at chance on `en` with no diagnosis |
| P1-analysis | Scorer comparison | 7 | Table 5 scorers × 3 languages with parse-failure rates (2); κ for 3 languages (1); 8 disagreement cases labelled with the fixed taxonomy + one-line justification each (2); recommendation cites ≥ 2 numbers from the table (2) | < 8 cases or labels outside the taxonomy (0–1 of 2); recommendation without numbers (0–1 of 2) | No analysis |
| P2-runs | Sweep, `results.csv` | 8 | 8 languages × {all, CS, CA}; validator passes; accuracies recompute to 0.001; CIs match the grader's recomputation within 0.01; bootstrap hand-written in numpy (auto) | −1 per missing language (max −4); CI from scipy / normal approximation (−3); CIs off by > 0.01 (−2) | Predictions missing |
| P2-figure | `accuracy_by_language.png` | 2 | CI bars, chance line, fixed language order, labelled axes | Missing chance line or CIs (1) | Missing |
| P2-stats | Statistics reasoning | 5 | Correct list of languages distinguishable from `en` under the **paired** CI (2); √(p(1−p)/n) explanation with the ±5/±7 numbers (1); CS − CA treated as **unpaired**, detectability stated per language (2) | Significance claimed from point gaps smaller than the CI (0 of 2); CS − CA treated as paired (0 of 2); vague explanation (0 of 1) | No statistics discussion |
| P2-parallel | Parallel-item analysis | 5 | `paired_vs_en.csv` and `cs_ca_gap.csv` complete and recompute (3); agreement/discordance interpreted, `de` vs `yo` contrasted (2) | CSVs present, interpretation missing (3); interpretation without CSVs (1) | Neither |
| P3-runs | Permutation + native runs | 7 | 12 new permutation runs + 3 native runs present, gold correctly remapped (auto: no "gold not remapped" flag; mean over the four permutations within 1.5 points of the instructor's); `permutations.csv` (incl. consistency rate) and `native_instruction.csv` recompute | Unremapped-gold flag on any run (2–3 total); a language missing (−1.5 each); native ablation missing (−2) | No permutation runs |
| P3-figure | Position bias | 3 | Letter frequencies pooled over the four permutations, 0.25 line, labelled axes, fixed language order | Raw counts instead of fractions, or no baseline (1) | Missing |
| P3-analysis | Sensitivity reasoning | 5 | Numeric comparison of permutation std vs Part 2 CI half-width with the right conclusion (2); position bias identified per language with the > 35 % criterion (1); native-instruction effect stated with paired CI and the `sw`-only caveat (2) | Descriptive only (0–1 of 2); caveat missing (1 of 2) | None |
| P4-csv | `tokenizer.csv` | 4 | 3 tokenizers × 8 languages, all four metrics, within 10 % of instructor values; zh `tokens_per_word` = NaN with reason | Fewer tokenizers (−1 each); `frac_partial_char_tokens` missing or redefined without note (−1); values off > 10 % (−1) | Missing |
| P4-figure | Fertility plot | 1 | Eight labelled points, no correlation coefficient | Unlabelled (0.5) | Missing |
| P4-analysis | Fertility reasoning | 5 | Byte-level BPE mechanism (1); diacritics / yo explained (1); quantitative cost estimate (1); **id vs sw** argument made explicitly (2) | "More tokens = worse" with no counter-example (2–3) | None |
| P5-runs | Qwen3 sweep | 5 | 8 languages, identical protocol (auto: `en` inside the Qwen3 band; `prompt_variant` and item order identical to Part 2; no `<think>` leakage) | −1 per missing language; protocol differs (−2) | Missing |
| P5-stats | Paired statistics | 5 | `comparison.csv` recomputes; paired CI (2); discordant counts (1); `mdd_95` correct with the derivation in the report (2) | CI-overlap only (1); MDD missing or wrong (0 of 2) | Missing |
| P5-analysis | Model comparison | 5 | Paired-vs-unpaired contrast with a concrete language (2); chance-level cells identified via CI (1); one confound with an executable control (2) | Confound named, no control (1 of 2); contrast asserted without a language (1 of 2) | None |
| P6-report | Report | 15 | Sections present incl. the "not concluded" sentence, ≤ 4 pages (3); every claim in *Analyses* tied to a table/figure ID and consistent with the CSVs (5; −1 per contradicted claim); three threats each with an executable mitigation (4); three practitioner rules each traceable to a result (3) | Over length −2 per page; generic threats ("small model") 1–2 of 4; rules not tied to evidence 1 of 3; results restated without interpretation 2–3 of 5 | No report |
| P6-reflection | Reflection | 5 | Three probe items cited by `sample_id` with numbers matching `probe.jsonl` (3); one concrete bug/surprise and how it was found (2) | < 3 items or numbers mismatch (1–2); generic (1) | Missing, or no reference to own outputs → integrity review |
| P6-repro | Reproducibility | 5 | Grader re-run of the Part 2 `en` command: ≥ 99 % item agreement and accuracy to 3 decimals (3); validators pass on every file, `run_log.txt` complete, `requirements.txt` present and installable (2) | Accuracy reproduces but < 99 % agreement (2 of 3); validator fails on some files (1 of 2) | Command does not run without edits |
| Bonus | S1–S5 | ≤ 10 | As stated per task; graded on analysis quality with the same claims-tied-to-numbers rule | | |

**Penalties.** Late: −10 per day. A prediction file still failing the validator after the 24-hour schema-fix
window: −5 per file. Disallowed model or altered frozen template in a graded run: that part scored 0.
Missing `integrity.txt` or `smoke_ok.json`: report not graded until supplied, and no grace window. Claims in
the report contradicted by the submitted CSVs: −1 each (max −5), on top of the P6 line.

**Point check.** P1 15 · P2 20 · P3 15 · P4 10 · P5 15 · P6 25 = 100.
