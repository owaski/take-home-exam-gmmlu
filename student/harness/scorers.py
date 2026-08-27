"""The three multiple-choice scorers. THIS IS YOUR MAIN IMPLEMENTATION TASK (Part 1).

Signatures are fixed; the CLI calls exactly these. Use the helpers in harness.model:
  forward_last_logits(model, tok, prompts, batcher)      -> np.ndarray [N, vocab]  (raw logits, last position)
  forward_token_logprobs(model, tok, texts, batcher)     -> list of np.ndarray, per-token log p(t_j | t_<j)
  generate_greedy(model, tok, prompts, max_new_tokens, batcher) -> list[str]
  letter_token_ids(tok)                                  -> ids of " A", " B", " C", " D" (asserts single tokens)
  n_tokens(tok, text)                                    -> token count without special tokens
All scores you return are LOG-probabilities (<= 0). The CLI sets pred = argmax(scores).
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

from .model import forward_last_logits, forward_token_logprobs, generate_greedy, letter_token_ids, n_tokens


def score_letter(model, tok, prompts: list[str], batcher=None) -> list[list[float]]:
    """LETTER: log-softmax over the FULL vocabulary at the last prompt position; return the four
    log-probs of the tokens " A", " B", " C", " D" (in that order) for every prompt."""
    raise NotImplementedError("TODO (Part 1)")


def score_cont(model, tok, prompts: list[str], options: list[list[str]], batcher=None) -> list[dict[str, list[float]]]:
    """CONT: for each prompt and each of its 4 option texts, the log-likelihood of the continuation
    " " + option_text after the prompt (sum of the continuation tokens' log-probs ONLY).

    Return, per prompt, {"CONT": [4 sums],
                         "CONT_TOKNORM":  [sum / n_continuation_tokens],
                         "CONT_CHARNORM": [sum / len(option_text)]}
    where n_continuation_tokens = n_tokens(prompt + " " + option) - n_tokens(prompt) (the added leading
    space is part of the continuation tokens) and len(option_text) is the number of characters of the
    ORIGINAL option text, i.e. WITHOUT the added leading space. Options are never empty in Global-MMLU-Lite
    (guard with max(1, ...) anyway if you like).
    Hint: the continuation's tokens are the last n_continuation_tokens positions of
    forward_token_logprobs(prompt + " " + option). Check that the prompt's tokenisation is a prefix of the
    full text's tokenisation (it is for these tokenizers because "Answer:" ends in ":" and the continuation
    starts with a space; tests/test_scorers.py::test_prompt_tokens_are_prefix checks it) — and say so in your README.
    """
    raise NotImplementedError("TODO (Part 1)")


def score_gen(model, tok, prompts: list[str], batcher=None, max_new_tokens: int = 8) -> list[str]:
    """GEN: greedy-decode up to max_new_tokens new tokens; return the raw decoded strings."""
    raise NotImplementedError("TODO (Part 1)")


def parse_gen(text: str) -> Optional[str]:
    """Map a raw generation to "A"/"B"/"C"/"D", or None when no single letter can be identified.

    Rule (case-insensitive for the letter; whitespace is stripped first):
      1. An explicit phrase wins: "answer"/"option"/"choice" optionally followed by "is", ":" or "=",
         then an optionally parenthesised letter -> that letter ("The answer is C", "The answer is (D).",
         "Answer: A").
      2. Otherwise a LEADING letter, optionally preceded by non-word characters such as "(" and followed by
         end-of-text, whitespace or one of . ) : , ; -> that letter (" B", "B.", "(B)", "b) because ...",
         "B. Paris"), UNLESS the text goes on to mention a DIFFERENT standalone capital letter without a
         terminator right after the first one ("A and B" -> None).
      3. Otherwise, if exactly one standalone capital A-D occurs anywhere -> that letter ("I think it's C").
      4. Otherwise None ("", "None of the above", "A and B").
    tests/test_scorers.py::PARSE_CASES lists every case above; all of them must pass:
      " B"->B, "B."->B, "(B)"->B, "b) because"->B, "B. Paris"->B, "The answer is C"->C,
      "The answer is (D)."->D, "Answer: A"->A, "I think it's C"->C,
      "None of the above"->None, "A and B"->None, ""->None.
    """
    raise NotImplementedError("TODO (Part 1)")
