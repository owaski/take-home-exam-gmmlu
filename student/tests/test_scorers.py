"""Tests for harness.scorers: parse_gen, letter tokens, left-padding safety, LETTER and CONT on real items."""
from __future__ import annotations

import json

import numpy as np
import pytest

from harness import prompts as PR
from harness import scorers as SC
from harness.model import apply_chat, forward_last_logits, letter_token_ids
from tests.conftest import reference_path

PARSE_CASES = [
    (" B", "B"), ("B.", "B"), ("(B)", "B"), ("b) because", "B"), ("B. Paris", "B"),
    ("The answer is C", "C"), ("The answer is (D).", "D"), ("Answer: A", "A"),
    ("None of the above", None), ("A and B", None), ("", None), ("I think it's C", "C"),
]


@pytest.mark.parametrize("text,expected", PARSE_CASES)
def test_parse_gen(text, expected):
    assert SC.parse_gen(text) == expected


def test_letter_token_ids_single_token(tok_qwen25, tok_qwen3):
    for tok in (tok_qwen25, tok_qwen3):
        ids = letter_token_ids(tok)
        assert len(ids) == 4 and len(set(ids)) == 4
        assert [tok.decode([i]) for i in ids] == [" A", " B", " C", " D"]


def _chat_prompts(tok, texts):
    return [apply_chat(tok, [{"role": "system", "content": "You are a helpful assistant."},
                             {"role": "user", "content": t}], PR.ASSISTANT_PREFIX) for t in texts]


@pytest.mark.model
def test_batched_forward_matches_single(qwen):
    model, tok, _ = qwen
    texts = ["Hi", "What is the capital of France? Reply with one word.",
             "Pick A or B.", "Consider the following long question about the history of astronomy and answer it briefly: who first proposed heliocentrism?"]
    ps = _chat_prompts(tok, texts)
    lens = [len(tok.encode(p, add_special_tokens=False)) for p in ps]
    assert len(set(lens)) == 4, "prompts should have distinct lengths for a meaningful padding test"
    batched = forward_last_logits(model, tok, ps)
    singles = np.stack([forward_last_logits(model, tok, [p])[0] for p in ps])
    assert batched.shape == singles.shape
    assert np.abs(batched - singles).max() < 1e-3


@pytest.mark.model
def test_score_letter_reproduces_reference(qwen):
    path = reference_path("en_first20.jsonl")
    with open(path, encoding="utf-8") as f:
        ref = [json.loads(l) for l in f if l.strip()][:8]
    from harness.data import load_gmmlu_lite
    model, tok, _ = qwen
    items = load_gmmlu_lite("en", n=8)
    assert [it.sample_id for it in items] == [r["sample_id"] for r in ref]
    ps = [apply_chat(tok, PR.build_messages(it, "v1_en"), PR.ASSISTANT_PREFIX) for it in items]
    scores = SC.score_letter(model, tok, ps)
    assert len(scores) == 8 and all(len(s) == 4 for s in scores)
    n_match, max_diff = 0, 0.0
    for s, r in zip(scores, ref):
        assert all(np.isfinite(s)) and max(s) <= 1e-9
        pred = "ABCD"[int(np.argmax(s))]
        n_match += pred == r["pred"]
        exp = r.get("scores_fp32") or r["scores"]
        max_diff = max(max_diff, float(np.abs(np.array(s) - np.array(exp)).max()))
    assert n_match >= 7, f"only {n_match}/8 predictions match the reference"
    assert max_diff < 0.15, f"max abs score diff {max_diff:.3f} vs reference"


@pytest.mark.model
def test_score_cont_shapes(qwen):
    from harness.data import load_gmmlu_lite
    model, tok, _ = qwen
    items = load_gmmlu_lite("en", n=2)
    ps = [apply_chat(tok, PR.build_messages(it, "v1_en"), PR.ASSISTANT_PREFIX) for it in items]
    res = SC.score_cont(model, tok, ps, [it.options for it in items])
    assert len(res) == 2
    for r in res:
        assert set(r) == {"CONT", "CONT_TOKNORM", "CONT_CHARNORM"}
        for k in r:
            assert len(r[k]) == 4 and all(np.isfinite(r[k])) and max(r[k]) <= 1e-9
        for raw, tn in zip(r["CONT"], r["CONT_TOKNORM"]):
            assert tn >= raw - 1e-9


def test_prompt_tokens_are_prefix(tok_qwen25):
    """The CONT scorer relies on tok(prompt) being a prefix of tok(prompt + " " + option)."""
    from harness.data import load_gmmlu_lite
    items = load_gmmlu_lite("en", n=3)
    for it in items:
        p = apply_chat(tok_qwen25, PR.build_messages(it, "v1_en"), PR.ASSISTANT_PREFIX)
        p_ids = tok_qwen25.encode(p, add_special_tokens=False)
        assert len(it.options) == 4 and all(o for o in it.options)
        for o in it.options:
            f_ids = tok_qwen25.encode(p + " " + o, add_special_tokens=False)
            assert f_ids[:len(p_ids)] == p_ids, f"prompt tokens are not a prefix for {it.sample_id} / {o!r}"
            assert len(f_ids) > len(p_ids)


@pytest.mark.model
def test_score_gen_and_parse(qwen):
    from harness.data import load_gmmlu_lite
    model, tok, _ = qwen
    items = load_gmmlu_lite("en", n=2)
    ps = [apply_chat(tok, PR.build_messages(it, "v1_en"), PR.ASSISTANT_PREFIX) for it in items]
    raw = SC.score_gen(model, tok, ps, max_new_tokens=8)
    assert len(raw) == 2
    for r in raw:
        assert isinstance(r, str) and r.strip() != ""
        assert SC.parse_gen(r) in ("A", "B", "C", "D", None)
