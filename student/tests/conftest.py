"""Shared pytest configuration for the instructor-provided test suite.

Run from the student/ directory:   python -m pytest tests            (includes small CPU model tests)
                                   python -m pytest tests --no-model (tokenizer/pure-python tests only)
Model tests always load Qwen2.5-0.5B-Instruct on the CPU in fp32 and use only a handful of items.
"""
from __future__ import annotations

import os
import sys

import pytest

# Make `import harness` work regardless of how pytest was launched (rootdir / cwd).
_STUDENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _STUDENT_DIR not in sys.path:
    sys.path.insert(0, _STUDENT_DIR)

REFERENCE_DIR = os.path.join(_STUDENT_DIR, "reference")


def pytest_addoption(parser):
    parser.addoption("--no-model", action="store_true", default=False,
                     help="skip every test that needs to load a model (tokenizer-only tests still run)")


def pytest_configure(config):
    config.addinivalue_line("markers", "model: needs a loaded model (slow-ish, CPU)")


@pytest.fixture(scope="session")
def qwen(request):
    """(model, tok, info) for Qwen2.5-0.5B-Instruct on the CPU, loaded once per session."""
    if request.config.getoption("--no-model"):
        pytest.skip("--no-model given: model tests skipped")
    try:
        from harness.model import load
        model, tok, info = load("qwen2.5", device="cpu", dtype="fp32")
    except Exception as e:  # network / cache / memory problems should not crash the whole suite
        pytest.skip(f"could not load qwen2.5 on cpu: {type(e).__name__}: {e}")
    return model, tok, info


def _tokenizer(alias: str):
    from harness.model import resolve_model
    from transformers import AutoTokenizer
    hub, rev = resolve_model(alias)
    return AutoTokenizer.from_pretrained(hub, revision=rev)


@pytest.fixture(scope="session")
def tok_qwen25():
    try:
        return _tokenizer("qwen2.5")
    except Exception as e:
        pytest.skip(f"could not load the qwen2.5 tokenizer: {e}")


@pytest.fixture(scope="session")
def tok_qwen3():
    try:
        return _tokenizer("qwen3")
    except Exception as e:
        pytest.skip(f"could not load the qwen3 tokenizer: {e}")


def reference_path(name: str) -> str:
    """Absolute path of a file in student/reference; skip the calling test when it is missing."""
    path = os.path.join(REFERENCE_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"reference file {name} not found in {REFERENCE_DIR}")
    return path
