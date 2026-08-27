"""REFERENCE SOLUTION — tokenizer_stats.py (Part 4). Tokenizer-only, CPU, no model weights.

Usage (from the student/ directory):
    python tokenizer_stats.py [--out tokenizer.csv] [--langs en de zh ar hi id sw yo] [--tokenizers qwen2.5 qwen3 smollm2]

For every tokenizer in harness.model.MODELS (pinned revision) and every core language, over the 400 test items:
    text = question + "\\n" + "\\n".join(options)
    n_items                   400
    mean_tokens               mean number of tokens per item (add_special_tokens=False)
    tokens_per_char           total tokens / total characters, characters = len(NFC-normalised text)
    tokens_per_word           total tokens / total whitespace-split words; NaN for zh (no whitespace word boundaries)
    frac_partial_char_tokens  fraction of tokens whose decoded string (tokenizer.convert_tokens_to_string([tok]))
                              is empty or contains U+FFFD, i.e. byte-level tokens that do not form a whole character
Columns: tokenizer, lang, n_items, mean_tokens, tokens_per_char, tokens_per_word, frac_partial_char_tokens.
Qwen2.5 and Qwen3 share a vocabulary; both rows are kept on purpose (identical numbers are themselves a finding).
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
NO_WHITESPACE_LANGS = {"zh"}   # tokens_per_word is undefined: Chinese does not delimit words with whitespace


def item_text(item) -> str:
    return item.question + "\n" + "\n".join(item.options)


def load_tokenizer(alias: str):
    from transformers import AutoTokenizer
    hub, rev = MODELS[alias]
    return AutoTokenizer.from_pretrained(hub, revision=rev)


def is_partial_char_token(tok, token: str, cache: dict) -> bool:
    """True when the token alone does not decode to at least one complete Unicode character."""
    if token not in cache:
        s = tok.convert_tokens_to_string([token])
        cache[token] = (s == "") or ("�" in s)
    return cache[token]


def stats_for(tok, items: list, lang: str) -> dict:
    n_tok, n_char, n_word, n_partial = 0, 0, 0, 0
    per_item = []
    cache: dict = {}
    for it in items:
        text = unicodedata.normalize("NFC", item_text(it))
        ids = tok.encode(text, add_special_tokens=False)
        toks = tok.convert_ids_to_tokens(ids)
        per_item.append(len(ids))
        n_tok += len(ids)
        n_char += len(text)
        n_word += len(text.split())
        n_partial += sum(is_partial_char_token(tok, t, cache) for t in toks)
    tpw = float("nan") if lang in NO_WHITESPACE_LANGS else n_tok / max(n_word, 1)
    return {"lang": lang, "n_items": len(items), "mean_tokens": float(np.mean(per_item)),
            "tokens_per_char": n_tok / max(n_char, 1), "tokens_per_word": tpw,
            "frac_partial_char_tokens": n_partial / max(n_tok, 1)}


def build_tokenizer_table(tokenizers=("qwen2.5", "qwen3", "smollm2"), langs=CORE_LANGS) -> pd.DataFrame:
    items_by_lang = {l: load_gmmlu_lite(l) for l in langs}
    recs = []
    for alias in tokenizers:
        tok = load_tokenizer(alias)
        for lang in langs:
            row = {"tokenizer": alias, **stats_for(tok, items_by_lang[lang], lang)}
            if lang in NO_WHITESPACE_LANGS:
                print(f"[tokenizer_stats] {alias}/{lang}: tokens_per_word = NaN — {lang} has no whitespace word "
                      "boundaries, so whitespace-split 'words' are sentences, not words")
            recs.append(row)
            print(f"[tokenizer_stats] {alias:8s} {lang}: mean_tokens={row['mean_tokens']:.1f} "
                  f"tok/char={row['tokens_per_char']:.3f} tok/word={row['tokens_per_word']:.3f} "
                  f"partial={row['frac_partial_char_tokens']:.3f}")
    return pd.DataFrame(recs, columns=TOKENIZER_COLUMNS)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Part 4: tokenizer fertility statistics (no model loaded).")
    ap.add_argument("--out", default="tokenizer.csv")
    ap.add_argument("--langs", nargs="+", default=CORE_LANGS)
    ap.add_argument("--tokenizers", nargs="+", default=list(MODELS), help=f"aliases from {sorted(MODELS)}")
    args = ap.parse_args(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # tokenizers only; never touch a GPU
    df = build_tokenizer_table(args.tokenizers, args.langs)
    df.to_csv(args.out, index=False)
    print(f"[tokenizer_stats] wrote {args.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
