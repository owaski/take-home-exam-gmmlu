"""Model loading and batched forward helpers. Students do not modify this file; use CLI flags instead.

Supported devices: cuda (graders), mps (Apple Silicon), cpu (Intel Macs / fallback).
"""
from __future__ import annotations

import gc
import math
import time
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# alias -> (hub id, pinned revision). Only these models may be used in graded core runs.
MODELS = {
    "qwen2.5": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    "qwen3":   ("Qwen/Qwen3-0.6B",            "c1899de289a04d12100db370d81485cdf75e47ca"),
    "smollm2": ("HuggingFaceTB/SmolLM2-360M-Instruct", "a10cc1512eabd3dde888204e902eca88bddb4951"),
}
PRIMARY_MODEL, CONTRAST_MODEL = "qwen2.5", "qwen3"
ALLOWED_MODEL_IDS = {hub for hub, _ in MODELS.values()}


def resolve_model(name: str) -> tuple[str, str]:
    """Accept an alias ('qwen2.5') or a full hub id; return (hub_id, revision)."""
    if name in MODELS:
        return MODELS[name]
    for hub, rev in MODELS.values():
        if name == hub:
            return hub, rev
    raise ValueError(f"model {name!r} is not in the allowed list {sorted(MODELS)}")


def select_device(override: Optional[str] = None) -> str:
    if override and override != "auto":
        return override
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def select_dtype(device: str, override: Optional[str] = None) -> torch.dtype:
    if override and override != "auto":
        return {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[override]
    return torch.float32 if device == "cpu" else torch.float16


def load(name: str, device: Optional[str] = None, dtype: Optional[str] = None):
    """Load (model, tokenizer, info). `info` records what was actually loaded, for run_log.txt."""
    hub, rev = resolve_model(name)
    device = select_device(device)
    torch_dtype = select_dtype(device, dtype)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(hub, revision=rev)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    try:
        model = AutoModelForCausalLM.from_pretrained(hub, revision=rev, dtype=torch_dtype)
    except TypeError:   # transformers < 4.56 does not know `dtype=`; the older keyword is `torch_dtype=`
        model = AutoModelForCausalLM.from_pretrained(hub, revision=rev, torch_dtype=torch_dtype)
    model.to(device).eval()
    torch.manual_seed(0)
    info = {"model": hub, "model_sha": rev, "device": device, "dtype": str(torch_dtype).replace("torch.", ""),
            "load_seconds": round(time.time() - t0, 1)}
    return model, tok, info


def apply_chat(tok, messages: list[dict], assistant_prefix: str = "Answer:") -> str:
    """Render chat messages with the model's own template, then append the assistant prefix as plain text."""
    kwargs = {}
    if "enable_thinking" in (tok.chat_template or ""):
        kwargs["enable_thinking"] = False        # Qwen3: never let a <think> block into the prompt
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    assert "<think>" not in text or text.rstrip().endswith("</think>"), "thinking block leaked into prompt"
    return text + assistant_prefix


class _Batcher:
    """Runs a batched function over a list, halving the batch size on out-of-memory errors."""

    def __init__(self, batch_size: int):
        self.batch_size = max(1, batch_size)
        self.halvings = 0

    def run(self, fn, xs: list, *extra):
        out, i = [], 0
        while i < len(xs):
            bs = min(self.batch_size, len(xs) - i)
            chunk = xs[i:i + bs]
            extras = [e[i:i + bs] for e in extra]
            try:
                out.extend(fn(chunk, *extras))
                i += bs
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:  # MPS raises RuntimeError on OOM
                msg = str(e).lower()
                if "out of memory" not in msg and "mps backend" not in msg and "allocat" not in msg:
                    raise
                _free_memory()
                if self.batch_size == 1:
                    raise
                self.batch_size = max(1, self.batch_size // 2)
                self.halvings += 1
        return out


def _free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


@torch.no_grad()
def forward_last_logits(model, tok, prompts: list[str], batcher: Optional[_Batcher] = None) -> np.ndarray:
    """Logits at the LAST (non-pad) position of each prompt. Returns float32 array [N, vocab].

    Left padding + attention mask are handled here, so the last position is always the final
    real token. This is the only helper the LETTER scorer needs.
    """
    batcher = batcher or _Batcher(8)
    device = next(model.parameters()).device

    def _fn(chunk):
        tok.padding_side = "left"
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        logits = model(**enc).logits[:, -1, :].float().cpu().numpy()
        return list(logits)

    return np.stack(batcher.run(_fn, prompts))


@torch.no_grad()
def forward_token_logprobs(model, tok, texts: list[str], batcher: Optional[_Batcher] = None) -> list[np.ndarray]:
    """Per-token log-probabilities for each full text (no padding tokens included).

    For a text tokenised to ids t_0..t_{T-1} the returned array has length T-1 and entry j is
    log p(t_{j+1} | t_0..t_j). Token 0 has no score. Texts are tokenised with add_special_tokens=False
    (chat templates already contain their special tokens). Used by the CONT scorer, which must work
    out which of these positions belong to the continuation.
    """
    batcher = batcher or _Batcher(8)
    device = next(model.parameters()).device

    def _fn(chunk):
        tok.padding_side = "left"
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        logits = model(**enc).logits.float()                    # [B, T, V]
        logp = torch.log_softmax(logits[:, :-1, :], dim=-1)     # predict positions 1..T-1
        target = enc["input_ids"][:, 1:]
        gathered = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)   # [B, T-1]
        mask = enc["attention_mask"][:, 1:].bool()
        out = []
        for row, m in zip(gathered, mask):
            out.append(row[m].cpu().numpy())                    # left padding -> real tokens are the suffix
        return out

    return batcher.run(_fn, texts)


def n_tokens(tok, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))


@torch.no_grad()
def generate_greedy(model, tok, prompts: list[str], max_new_tokens: int = 8, batcher: Optional[_Batcher] = None) -> list[str]:
    """Greedy decoding continuation for each prompt (decoded text of the NEW tokens only)."""
    batcher = batcher or _Batcher(8)
    device = next(model.parameters()).device

    def _fn(chunk):
        tok.padding_side = "left"
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None)
        new = out[:, enc["input_ids"].shape[1]:]
        return tok.batch_decode(new, skip_special_tokens=True)

    return batcher.run(_fn, prompts)


def letter_token_ids(tok, letters=("A", "B", "C", "D"), prefix: str = " ") -> list[int]:
    """Token id of `prefix + letter` for each letter. Asserts each is exactly ONE token."""
    ids = []
    for L in letters:
        enc = tok.encode(prefix + L, add_special_tokens=False)
        assert len(enc) == 1, f"{prefix + L!r} is not a single token for this tokenizer: {enc}"
        ids.append(enc[0])
    return ids
