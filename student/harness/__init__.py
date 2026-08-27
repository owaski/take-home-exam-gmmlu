"""Evaluation harness for the take-home exam "Small Models, Many Languages".

Package layout
--------------
data.py     dataset loading (pinned revision), LITE subset, per-student probe ids
prompts.py  frozen prompt templates, option permutation (student TODO)
model.py    device/dtype selection, model loading, batched forward helpers
scorers.py  LETTER / CONT / GEN scorers (student TODO)
stats.py    bootstrap CIs, paired statistics, kappa, MDD (student TODO)
schema.py   prediction-record schema, JSONL I/O, run log
cli.py      `python -m harness run ...`
"""
__version__ = "1.0.0"
