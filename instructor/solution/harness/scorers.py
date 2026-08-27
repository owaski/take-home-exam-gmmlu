"""REFERENCE SOLUTION — Part 1 scorers. Do not distribute to students."""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

from .model import forward_last_logits, forward_token_logprobs, generate_greedy, letter_token_ids, n_tokens


def _log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def score_letter(model, tok, prompts, batcher=None):
    ids = letter_token_ids(tok)
    logits = forward_last_logits(model, tok, prompts, batcher)
    logp = _log_softmax(logits.astype(np.float64))
    return logp[:, ids].tolist()


def score_cont(model, tok, prompts, options, batcher=None):
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
    texts, n_cont = [], []
    for p, opts in zip(prompts, options):
        p_ids = tok.encode(p, add_special_tokens=False)
        for o in opts:
            full = p + " " + o
            f_ids = tok.encode(full, add_special_tokens=False)
            assert f_ids[:len(p_ids)] == p_ids, "prompt tokenisation is not a prefix of prompt+continuation"
            texts.append(full)
            n_cont.append(max(1, len(f_ids) - len(p_ids)))
    lps = forward_token_logprobs(model, tok, texts, batcher)
    out, k = [], 0
    for opts in options:
        raw, tokn, charn = [], [], []
        for o in opts:
            s = float(lps[k][-n_cont[k]:].sum())
            raw.append(s); tokn.append(s / n_cont[k]); charn.append(s / max(1, len(o)))
            k += 1
        out.append({"CONT": raw, "CONT_TOKNORM": tokn, "CONT_CHARNORM": charn})
    return out


def score_gen(model, tok, prompts, batcher=None, max_new_tokens: int = 8):
    return generate_greedy(model, tok, prompts, max_new_tokens=max_new_tokens, batcher=batcher)


_LEAD = re.compile(r"^\W*([ABCDabcd])(?=[\s\.\):,;]|$)")
_PHRASE = re.compile(r"\b(?:answer|option|choice)\b\s*(?:is|:|=)?\s*\(?([ABCD])\)?(?=[\s\.\):,;]|$)", re.I)


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
    t = (text or "").strip()
    if not t:
        return None
    standalone = set(re.findall(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", t))
    m = _PHRASE.search(t)                      # "The answer is (B)." beats everything else
    if m:
        return m.group(1).upper()
    m = _LEAD.match(t)                         # " B", "B.", "(B)", "b) because", "B. Paris", "C\n\nExplanation..."
    if m:
        terminated = t[m.end():m.end() + 1] in ("", "\n", ".", ")", ":")
        if terminated or len(standalone - {m.group(1).upper()}) == 0:
            return m.group(1).upper()          # "A and B" (no terminator, second letter present) stays ambiguous
    return standalone.pop() if len(standalone) == 1 else None
