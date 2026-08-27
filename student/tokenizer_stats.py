"""tokenizer_stats.py — Part 4. Tokenizer-only statistics; loads NO model weights (CPU, ~1-2 min).

Usage (from this directory):
    python tokenizer_stats.py [--out tokenizer.csv] [--langs en de zh ar hi id sw yo] [--tokenizers qwen2.5 qwen3 smollm2]

For every tokenizer alias in harness.model.MODELS (load with AutoTokenizer.from_pretrained(hub, revision=rev) —
the pinned revision, tokenizer only) and every core language, over the 400 test items:
    text = question + "\\n" + "\\n".join(options)          (NFC-normalise before counting characters)
    n_items                   400
    mean_tokens               mean tokens per item, tokenizer.encode(text, add_special_tokens=False)
    tokens_per_char           total tokens / total characters (characters = len of the NFC-normalised text)
    tokens_per_word           total tokens / total whitespace-split words; NaN for zh — print why
    frac_partial_char_tokens  fraction of tokens whose tokenizer.convert_tokens_to_string([tok]) is "" or contains
                              U+FFFD, i.e. byte-level tokens that do not by themselves form a whole character
Columns (FIXED): tokenizer, lang, n_items, mean_tokens, tokens_per_char, tokens_per_word, frac_partial_char_tokens
`tokenizer` holds the alias (qwen2.5 / qwen3 / smollm2). Qwen2.5 and Qwen3 share a vocabulary: keep both rows —
identical numbers are themselves a finding worth one sentence in the report.
"""
from __future__ import annotations

import argparse
import os
import sys
import unicodedata

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness.data import CORE_LANGS, load_gmmlu_lite   # noqa: E402
from harness.model import MODELS                        # noqa: E402

TOKENIZER_COLUMNS = ["tokenizer", "lang", "n_items", "mean_tokens", "tokens_per_char", "tokens_per_word",
                     "frac_partial_char_tokens"]
NO_WHITESPACE_LANGS = {"zh"}


def item_text(item) -> str:
    return item.question + "\n" + "\n".join(item.options)


def load_tokenizer(alias: str):
    """AutoTokenizer for the alias at its pinned revision (harness.model.MODELS). No model weights."""
    raise NotImplementedError("TODO (Part 4)")


def is_partial_char_token(tok, token: str) -> bool:
    """True when the token alone does not decode to at least one complete Unicode character."""
    raise NotImplementedError("TODO (Part 4)")


def stats_for(tok, items: list, lang: str) -> dict:
    """One row (without the `tokenizer` column) for one tokenizer x language."""
    raise NotImplementedError("TODO (Part 4)")


def build_tokenizer_table(tokenizers=("qwen2.5", "qwen3", "smollm2"), langs=CORE_LANGS) -> pd.DataFrame:
    """DataFrame with columns TOKENIZER_COLUMNS, one row per tokenizer x language."""
    raise NotImplementedError("TODO (Part 4)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Part 4: tokenizer fertility statistics (no model loaded).")
    ap.add_argument("--out", default="tokenizer.csv")
    ap.add_argument("--langs", nargs="+", default=CORE_LANGS)
    ap.add_argument("--tokenizers", nargs="+", default=list(MODELS), help=f"aliases from {sorted(MODELS)}")
    args = ap.parse_args(argv)
    df = build_tokenizer_table(args.tokenizers, args.langs)
    assert list(df.columns) == TOKENIZER_COLUMNS, "column order is fixed"
    df.to_csv(args.out, index=False)
    print(f"[tokenizer_stats] wrote {args.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
