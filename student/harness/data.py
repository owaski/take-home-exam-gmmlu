"""Dataset loading. Everything here is frozen: do not change revisions, ordering or subset rules."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset

# ---- pinned revisions (commit hashes on the Hugging Face Hub) ----------------
GMMLU_LITE_ID = "CohereLabs/Global-MMLU-Lite"
GMMLU_LITE_REV = "36c2fd756f19ccf13a9a96c8e53ccecc02192b8b"
BELEBELE_ID, BELEBELE_REV = "facebook/belebele", "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"
MGSM_ID, MGSM_REV = "juletxara/mgsm", "b2f13d426afe3be8d69a7e739b36724db8b66bbc"
XCOPA_ID, XCOPA_REV = "cambridgeltl/xcopa", "042f78955ba48e6404616762fa6e05e839c3907a"

# The eight core languages, in the fixed presentation order used by every table and figure.
CORE_LANGS = ["en", "de", "zh", "ar", "hi", "id", "sw", "yo"]
LANG_NAMES = {"en": "English", "de": "German", "zh": "Chinese", "ar": "Arabic",
              "hi": "Hindi", "id": "Indonesian", "sw": "Swahili", "yo": "Yoruba"}
ALL_GMMLU_LANGS = ["ar", "bn", "cs", "cy", "de", "en", "es", "fr", "hi", "hu", "id", "it", "ja",
                   "ko", "my", "or", "pt", "sk", "sq", "sw", "tg", "yo", "zh"]

LETTERS = ["A", "B", "C", "D"]
LITE_PER_LABEL = 100   # LITE mode = first 100 CS + first 100 CA items (sample_id order)
PROBE_SIZE = 20


@dataclass
class Item:
    """One multiple-choice item. `options` are in dataset order; `answer` is the gold letter."""
    sample_id: str
    subject: str
    question: str
    options: list[str]
    answer: str                 # "A".."D" — the letter that is correct *as shown in `options`*
    cs_label: str               # "CS" | "CA" | "-" (dev split has no label)
    lang: str
    benchmark: str = "global_mmlu_lite"
    gold_text_id: int = -1      # index of the correct option in the ORIGINAL dataset order
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.gold_text_id < 0:
            self.gold_text_id = LETTERS.index(self.answer)

    @property
    def gold_index(self) -> int:
        """Index (0-3) of the correct option in the current `options` order."""
        return LETTERS.index(self.answer)


def _row_to_item(row: dict, lang: str) -> Item:
    return Item(
        sample_id=row["sample_id"],
        subject=row["subject"],
        question=row["question"].strip(),
        options=[row["option_a"].strip(), row["option_b"].strip(), row["option_c"].strip(), row["option_d"].strip()],
        answer=row["answer"].strip().upper(),
        cs_label=row.get("cultural_sensitivity_label", "-") or "-",
        lang=lang,
    )


def load_gmmlu_lite(lang: str, split: str = "test", lite: bool = False, n: Optional[int] = None) -> list[Item]:
    """Load one language config of Global-MMLU-Lite, sorted by sample_id.

    lite=True selects the fixed LITE subset (first 100 CS + first 100 CA in sample_id order).
    n (debug only) truncates after subset selection; never use it for a graded run.
    """
    if lang not in ALL_GMMLU_LANGS:
        raise ValueError(f"unknown Global-MMLU-Lite config {lang!r}")
    ds = load_dataset(GMMLU_LITE_ID, lang, split=split, revision=GMMLU_LITE_REV)
    items = sorted((_row_to_item(r, lang) for r in ds), key=lambda it: it.sample_id)
    if split == "test":
        assert len(items) == 400, f"expected 400 test items for {lang}, got {len(items)}"
    if lite:
        items = lite_subset(items)
    if n is not None:
        items = items[:n]
    return items


def lite_subset(items: list[Item]) -> list[Item]:
    """First LITE_PER_LABEL CS items and first LITE_PER_LABEL CA items in sample_id order (order preserved)."""
    keep, seen = [], {"CS": 0, "CA": 0}
    for it in items:
        if it.cs_label in seen and seen[it.cs_label] < LITE_PER_LABEL:
            seen[it.cs_label] += 1
            keep.append(it)
    return keep


def probe_ids(student_id: str, all_sample_ids=None, k: int = PROBE_SIZE) -> list[str]:
    """Deterministically pick k sample_ids for this student (same ids in every language, because the data are parallel).

    The ids are ALWAYS drawn from the full list of 400 `en` sample_ids, independent of LITE mode, so that a
    student's probe set is fixed once and for all. `all_sample_ids` may be the 400 sample_id strings (or Items);
    when None, `load_gmmlu_lite("en")` is loaded and its sorted sample_ids are used. In LITE mode the CLI simply
    intersects these ids with the items it actually scored.

    Uses SHA-256 of the student id as the seed of a simple linear-congruential shuffle so that the choice does not
    depend on numpy's version. Items are addressed by their sorted sample_id order.
    """
    if all_sample_ids is None:
        all_sample_ids = load_gmmlu_lite("en")
    ids = sorted(x.sample_id if isinstance(x, Item) else str(x) for x in all_sample_ids)
    seed = int(hashlib.sha256(student_id.strip().encode("utf-8")).hexdigest(), 16)
    chosen, x, n = [], seed, len(ids)
    while len(chosen) < min(k, n):
        x = (x * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        cand = ids[x % n]
        if cand not in chosen:
            chosen.append(cand)
    return sorted(chosen)


# ---- stretch-task loaders ---------------------------------------------------
def load_belebele(config: str, n: Optional[int] = None) -> list[Item]:
    """Belebele reading comprehension (e.g. config 'eng_Latn'). Passage is folded into the question text."""
    ds = load_dataset(BELEBELE_ID, config, split="test", revision=BELEBELE_REV)
    items = []
    for r in ds:
        gold = int(r["correct_answer_num"]) - 1
        items.append(Item(
            sample_id=f"{r['link']}#{r['question_number']}",
            subject="reading comprehension",
            question=f"Passage: {r['flores_passage'].strip()}\n\nQuestion: {r['question'].strip()}",
            options=[r[f"mc_answer{i}"].strip() for i in range(1, 5)],
            answer=LETTERS[gold], cs_label="-", lang=config, benchmark="belebele"))
    items.sort(key=lambda it: it.sample_id)
    return items[:n] if n else items


def load_xcopa(lang: str, n: Optional[int] = None) -> list[dict]:
    ds = load_dataset(XCOPA_ID, lang, split="test", revision=XCOPA_REV)
    rows = [dict(r) for r in ds]
    return rows[:n] if n else rows


def load_mgsm(lang: str, split: str = "test", n: Optional[int] = None) -> list[dict]:
    ds = load_dataset(MGSM_ID, lang, split=split, revision=MGSM_REV)
    rows = [dict(r) for r in ds]
    return rows[:n] if n else rows
