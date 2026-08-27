"""Tests for harness.prompts: frozen template, permutation semantics, native templates, chat rendering."""
from __future__ import annotations

import pytest

from harness.data import Item, LETTERS
from harness.model import apply_chat
from harness.prompts import ASSISTANT_PREFIX, NATIVE_TEMPLATES, PERM_ORDERS, build_messages, permute


def toy_item(**kw) -> Item:
    base = dict(sample_id="toy/test/1", subject="world_history", question="Which letter comes third?",
                options=["w", "x", "y", "z"], answer="C", cs_label="CA", lang="en")
    base.update(kw)
    return Item(**base)


def test_build_messages_v1_en_is_frozen():
    msgs = build_messages(toy_item(), "v1_en")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == "You are a helpful assistant."
    assert msgs[1]["content"] == (
        "The following is a multiple-choice question about world history. Choose the single best answer.\n\n"
        "Which letter comes third?\n\n"
        "A. w\nB. x\nC. y\nD. z\n\n"
        "Reply with only the letter (A, B, C, or D)."
    )


def test_perm_variants_use_v1_wording():
    assert build_messages(toy_item(), "perm_BCDA") == build_messages(toy_item(), "v1_en")


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        build_messages(toy_item(), "v9_bogus")


@pytest.mark.parametrize("order", PERM_ORDERS)
def test_permute_semantics(order):
    it = toy_item()
    new = permute(it, order)
    # slot k of the new item shows the option that sat in slot order[k] of the original
    assert new.options == [it.options[LETTERS.index(L)] for L in order]
    # the gold letter now points at the slot holding the originally-correct text "y"
    assert new.options[LETTERS.index(new.answer)] == "y"
    assert new.gold_index == new.options.index("y")
    # provenance fields untouched
    assert new.gold_text_id == it.gold_text_id == 2
    assert (new.sample_id, new.subject, new.question, new.lang, new.cs_label) == \
        (it.sample_id, it.subject, it.question, it.lang, it.cs_label)
    # the original item must not be mutated
    assert it.options == ["w", "x", "y", "z"] and it.answer == "C"


def test_permute_ABCD_is_identity():
    it = toy_item()
    new = permute(it, "ABCD")
    assert new.options == it.options and new.answer == it.answer and new.gold_text_id == it.gold_text_id


def test_permute_every_gold_letter():
    for gold in LETTERS:
        it = toy_item(answer=gold)
        text = it.options[LETTERS.index(gold)]
        for order in PERM_ORDERS:
            new = permute(it, order)
            assert new.options[new.gold_index] == text
            assert new.gold_text_id == LETTERS.index(gold)


def test_permute_rejects_bad_order():
    with pytest.raises(ValueError):
        permute(toy_item(), "ACBD")


def test_v2_native_requires_template():
    assert "en" not in NATIVE_TEMPLATES
    with pytest.raises(ValueError):
        build_messages(toy_item(lang="en"), "v2_native")
    msgs = build_messages(toy_item(lang="zh"), "v2_native")
    assert msgs[0]["content"] == NATIVE_TEMPLATES["zh"]["system"]
    assert "A. w\nB. x\nC. y\nD. z" in msgs[1]["content"]


def test_apply_chat_qwen25(tok_qwen25):
    text = apply_chat(tok_qwen25, build_messages(toy_item()), ASSISTANT_PREFIX)
    assert text.endswith("<|im_start|>assistant\n" + ASSISTANT_PREFIX)
    assert "Which letter comes third?" in text


def test_apply_chat_qwen3_no_thinking(tok_qwen3):
    text = apply_chat(tok_qwen3, build_messages(toy_item()), ASSISTANT_PREFIX)
    assert text.endswith(ASSISTANT_PREFIX)
    # Qwen3 with enable_thinking=False emits an empty "<think>\n\n</think>" pair at most; no content inside
    body = text.split("<|im_start|>assistant")[-1]
    if "<think>" in body:
        inner = body.split("<think>", 1)[1].split("</think>", 1)[0]
        assert inner.strip() == "", "thinking content leaked into the prompt"
    assert "Which letter comes third?" in text
