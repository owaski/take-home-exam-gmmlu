"""REFERENCE SOLUTION (permute implemented). Frozen prompt templates and option permutation.

Do NOT edit V1_EN, NATIVE_TEMPLATES, ASSISTANT_PREFIX or PERM_ORDERS for any graded run.
The only thing to implement in this file is `permute` (Part 3).
"""
from __future__ import annotations

from dataclasses import replace

from .data import Item, LETTERS

# The assistant prefix appended (as plain text) after the chat template's generation prompt.
# It is deliberately kept in English for EVERY variant, so that the LETTER candidates " A".." D"
# are scored in an identical local context. (Discuss this as a threat to validity if you like.)
ASSISTANT_PREFIX = "Answer:"

PERM_ORDERS = ["ABCD", "BCDA", "CDAB", "DABC"]
VARIANTS = ["v1_en", "v2_native"] + [f"perm_{o}" for o in PERM_ORDERS]

V1_EN = {
    "system": "You are a helpful assistant.",
    "user": (
        "The following is a multiple-choice question about {subject}. Choose the single best answer.\n\n"
        "{question}\n\n"
        "A. {a}\nB. {b}\nC. {c}\nD. {d}\n\n"
        "Reply with only the letter (A, B, C, or D)."
    ),
}

# Instruction sentences translated by the instructor; the question/options are already in the target language.
# The subject name stays in English (the dataset does not provide translated subject names) — a known limitation.
NATIVE_TEMPLATES = {
    "zh": {
        "system": "你是一个乐于助人的助手。",
        "user": (
            "以下是一道关于“{subject}”的单项选择题。请选出唯一的最佳答案。\n\n"
            "{question}\n\n"
            "A. {a}\nB. {b}\nC. {c}\nD. {d}\n\n"
            "请只回答字母（A、B、C 或 D）。"
        ),
    },
    "hi": {
        "system": "आप एक मददगार सहायक हैं।",
        "user": (
            "नीचे \"{subject}\" विषय पर एक बहुविकल्पीय प्रश्न दिया गया है। केवल एक सर्वोत्तम उत्तर चुनें।\n\n"
            "{question}\n\n"
            "A. {a}\nB. {b}\nC. {c}\nD. {d}\n\n"
            "उत्तर में केवल अक्षर (A, B, C या D) लिखें।"
        ),
    },
    "sw": {
        "system": "Wewe ni msaidizi mwenye manufaa.",
        "user": (
            "Lifuatalo ni swali la chaguo-nyingi kuhusu \"{subject}\". Chagua jibu moja bora zaidi.\n\n"
            "{question}\n\n"
            "A. {a}\nB. {b}\nC. {c}\nD. {d}\n\n"
            "Jibu kwa herufi pekee (A, B, C, au D)."
        ),
    },
}


def subject_pretty(subject: str) -> str:
    return subject.replace("_", " ")


def build_messages(item: Item, variant: str = "v1_en") -> list[dict]:
    """Return the chat messages (system + user) for an item under a prompt variant.

    Permutation variants (perm_XXXX) use the V1_EN wording; the caller is responsible for
    having permuted the item's options first (see `permute`). `v2_native` requires a template
    for item.lang.
    """
    if variant == "v2_native":
        if item.lang not in NATIVE_TEMPLATES:
            raise ValueError(f"no native template for language {item.lang!r} (available: {sorted(NATIVE_TEMPLATES)})")
        tpl = NATIVE_TEMPLATES[item.lang]
    elif variant == "v1_en" or variant.startswith("perm_"):
        tpl = V1_EN
    else:
        raise ValueError(f"unknown prompt variant {variant!r}")
    a, b, c, d = item.options
    user = tpl["user"].format(subject=subject_pretty(item.subject), question=item.question, a=a, b=b, c=c, d=d)
    return [{"role": "system", "content": tpl["system"]}, {"role": "user", "content": user}]


def permute(item: Item, order: str) -> Item:
    """Return a copy of `item` whose options are cyclically re-ordered, with the gold letter remapped.

    `order` is one of PERM_ORDERS. Semantics: slot A of the new item shows the option that was in
    slot order[0] of the original, slot B shows the original order[1], and so on. Example: with
    order "BCDA" the new options are [orig_B, orig_C, orig_D, orig_A]. The new `answer` must be the
    letter of the slot that now contains the correct option. `gold_text_id` (index of the correct
    option in the ORIGINAL dataset order) must be preserved unchanged.

    Use `replace(item, options=..., answer=...)` (already imported from dataclasses) so that all other fields are kept.
    """
    if order not in PERM_ORDERS:
        raise ValueError(f"order must be one of {PERM_ORDERS}, got {order!r}")
    src_idx = [LETTERS.index(ch) for ch in order]            # new slot k shows original option src_idx[k]
    new_options = [item.options[i] for i in src_idx]
    new_gold = LETTERS[src_idx.index(item.gold_index)]        # slot that now holds the correct option
    return replace(item, options=new_options, answer=new_gold, gold_text_id=item.gold_text_id)


def apply_permutation(item: Item, variant: str) -> Item:
    """Helper used by the CLI: perm_XXXX -> permute; anything else -> the item unchanged."""
    if variant.startswith("perm_"):
        return permute(item, variant[len("perm_"):])
    return item
