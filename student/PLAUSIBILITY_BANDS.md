# Plausibility bands (published up front)

Measured by the instructor with the reference implementation and the exact frozen protocol
(`v1_en`, pinned revisions, fp16 on a GPU; fp32 on CPU reproduces every prediction below to within one
item). Your numbers should land inside **accuracy ± 4.5 points** for a full 400-item run and
**± 6.5 points** for a 200-item LITE run. Outside a band *with a documented cause* costs at most one
point; outside a band *with no diagnosis* is what costs code points. The grader uses these same values.

## Part 1 — scorers (Qwen2.5-0.5B-Instruct, n = 400)

| scorer | en | zh | sw |
|---|---|---|---|
| LETTER | 0.485 | 0.470 | 0.260 |
| CONT (raw sum) | 0.302 | 0.285 | 0.245 |
| CONT_TOKNORM | 0.348 | 0.315 | 0.300 |
| CONT_CHARNORM | 0.328 | 0.310 | 0.292 |
| GEN (greedy, 8 tokens) | 0.410 | 0.362 | 0.240 |
| GEN parse-failure rate | 0.00 | 0.00 | 0.00 |
| κ(LETTER, GEN) | 0.41 | 0.39 | 0.07 |

Yes, the parse-failure rate really is zero: the instruction "Reply with only the letter" is followed by
this model in every language, which is itself a finding (the generated letter is nevertheless *less*
accurate than the log-prob argmax). Your parser must still handle the cases in `tests/test_scorers.py`.

## Part 2 — LETTER sweep (Qwen2.5-0.5B-Instruct, n = 400 each; 95 % CI half-width ≈ ±0.048)

| | en | de | zh | ar | hi | id | sw | yo |
|---|---|---|---|---|---|---|---|---|
| all | 0.485 | 0.428 | 0.470 | 0.330 | 0.310 | 0.385 | 0.260 | 0.278 |
| CS (n = 200) | 0.435 | 0.410 | 0.460 | 0.310 | 0.280 | 0.365 | 0.245 | 0.260 |
| CA (n = 200) | 0.535 | 0.445 | 0.480 | 0.350 | 0.340 | 0.405 | 0.275 | 0.295 |
| paired diff vs en | — | −0.057 | −0.015 | −0.155 | −0.175 | −0.100 | −0.225 | −0.207 |
| paired 95 % CI half-width | — | ≈ 0.055 | ≈ 0.050 | ≈ 0.061 | ≈ 0.056 | ≈ 0.054 | ≈ 0.058 | ≈ 0.058 |
| agreement rate with en | — | 0.548 | 0.632 | 0.492 | 0.460 | 0.595 | 0.392 | 0.432 |

LITE subset (n = 200): en 0.440 · de 0.410 · zh 0.465 · ar 0.320 · hi 0.280 · id 0.355 · sw 0.265 · yo 0.240.

CS − CA gap: only `en` (−0.10) has an unpaired CI that excludes zero; all seven other languages are
undetectable at 200 vs 200.

## Part 3 — permutations and native instruction (Qwen2.5, n = 400)

| lang | mean over 4 permutations | std | consistency rate | most-chosen letter (pooled) |
|---|---|---|---|---|
| en | 0.486 | 0.019 | 0.33 | B 35 % |
| zh | 0.451 | 0.013 | 0.29 | A 34 % |
| hi | 0.292 | 0.014 | 0.05 | **B 54 %** |
| sw | 0.277 | 0.019 | 0.02 | **B 44 %**, D 1.5 % |

The grader requires your mean over the four permutations to be within 1.5 points of these means.
Native instruction (`v2_native` − `v1_en`): zh +0.025 · hi +0.010 · sw +0.018 — every paired CI includes 0.

## Part 4 — tokenizer (`mean_tokens` per item, question + 4 options)

| tokenizer | en | de | zh | ar | hi | id | sw | yo |
|---|---|---|---|---|---|---|---|---|
| qwen2.5 (= qwen3) | 85.7 | 135.5 | 82.5 | 128.8 | 360.8 | 129.0 | 164.1 | 253.6 |
| smollm2 | 88.1 | 180.2 | 214.8 | 319.1 | 424.4 | 163.2 | 179.5 | 323.5 |

qwen2.5 `frac_partial_char_tokens`: en 0.001 · de 0.000 · zh 0.028 · ar 0.004 · **hi 0.509** · id 0.000 ·
sw 0.000 · yo 0.070. Tolerance: 10 % of each value.

## Part 5 — Qwen3-0.6B, LETTER, `v1_en`, n = 400

| | en | de | zh | ar | hi | id | sw | yo |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 0.502 | 0.450 | 0.490 | 0.362 | 0.335 | 0.470 | 0.260 | 0.278 |
| paired diff (Qwen2.5 − Qwen3) | −0.017 | −0.022 | −0.020 | −0.032 | −0.025 | **−0.085** | 0.000 | 0.000 |
| mdd_95 | ≈ 0.057 | ≈ 0.056 | ≈ 0.053 | ≈ 0.057 | ≈ 0.057 | ≈ 0.057 | ≈ 0.053 | ≈ 0.059 |

## Stretch

S1 Belebele (Qwen2.5, LETTER, first 300 items): eng_Latn 0.580 · zho_Hans 0.550 · hin_Deva 0.323 · swh_Latn 0.297.
S3 SmolLM2-360M-Instruct, LETTER, all eight languages: 0.215–0.270 (at chance everywhere, including `en`).
