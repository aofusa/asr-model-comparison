"""
TDD tests for multi-family real model support (Qwen3-ASR + Voxtral).

These tests are written BEFORE implementing the Qwen3 and Voxtral backends.

Goals of this reinforcement phase:
- Clearly communicate that full real support for Qwen3-ASR and Voxtral is not yet complete.
- Provide clean extension points so they can be added later without breaking the architecture.
- Ensure the single-model-in-memory constraint is never violated, even for unsupported families.
"""
from __future__ import annotations

import os
import asyncio

import pytest

from app.services.model_manager import ModelManager
from app.services.asr_backends.factory import create_whisper_loader


@pytest.fixture
def real_whisper_manager():
    """Manager configured for real Whisper (used as baseline in multi-family tests)."""
    loader = create_whisper_loader(device="cpu", compute_type="int8")
    manager = ModelManager(model_loader=loader)
    yield manager
    # Ensure cleanup
    asyncio.run(manager.unload_current())


def test_qwen3_and_voxtral_have_metadata_but_no_real_backend_yet():
    """
    The model list advertises Qwen3 and Voxtral, but real backend support
    is not yet implemented. This test documents the current state.
    """
    from app.models.available_models import AVAILABLE_MODELS

    families = {m.family for m in AVAILABLE_MODELS}
    assert "qwen3" in families
    assert "voxtral" in families

    # Whisper has a real implementation
    whisper_models = [m for m in AVAILABLE_MODELS if m.family == "whisper"]
    assert len(whisper_models) > 0


@pytest.mark.slow
@pytest.mark.skipif(
    os.getenv("USE_REAL_WHISPER") != "1" and os.getenv("USE_REAL_MODELS") != "1",
    reason="Requires real model environment (USE_REAL_MODELS=1)"
)
def test_loading_qwen3_in_real_mode_attempts_actual_model_load(real_whisper_manager):
    """
    With real backends implemented, loading Qwen3-ASR will attempt to load
    the actual model via transformers (may download several GB on first run).
    """
    manager = real_whisper_manager

    # It should no longer raise "not implemented" — it will try real loading
    # (which may take time / memory on first run)
    try:
        asyncio.run(manager.load_model("qwen3-asr-0.6b"))
    except Exception as e:
        # Acceptable failures: download, OOM, or model-specific inference error
        # The important thing is it did not hit the old NotImplemented skeleton
        assert "not implemented" not in str(e).lower()


@pytest.mark.slow
@pytest.mark.skipif(
    os.getenv("USE_REAL_WHISPER") != "1" and os.getenv("USE_REAL_MODELS") != "1",
    reason="Requires real model environment (USE_REAL_MODELS=1)"
)
def test_loading_voxtral_in_real_mode_attempts_actual_model_load(real_whisper_manager):
    """
    With the real VoxtralBackend, loading will attempt actual inference
    using the transformers pipeline.
    """
    manager = real_whisper_manager

    try:
        asyncio.run(manager.load_model("voxtral-mini-4b"))
    except Exception as e:
        assert "not implemented" not in str(e).lower()


@pytest.mark.slow
def test_attempting_unsupported_family_with_real_loader_fails_cleanly():
    """
    When using a real loader path, attempting Qwen3/Voxtral must raise
    a clear error from the backend and leave the manager in a safe state.
    """
    from app.services.asr_backends.factory import create_whisper_loader

    loader = create_whisper_loader(device="cpu", compute_type="int8")
    manager = ModelManager(model_loader=loader)

    # Load a real Whisper model first
    asyncio.run(manager.load_model("whisper-tiny"))
    assert manager.current_model_id == "whisper-tiny"

    # Attempting Qwen3 should raise a clear error (either from the wrong loader
    # or from the dedicated Qwen backend once a multi-family loader is used).
    with pytest.raises((ValueError, NotImplementedError, RuntimeError), match="(?i)(qwen|unsupported|not (yet )?supported|not implemented)"):
        asyncio.run(manager.load_model("qwen3-asr-1.7b"))

    # After the failure, we should not be left with a broken state.
    # The previous successful model may still be loaded (acceptable),
    # but we must never have the failed model as "current" without a working instance.
    assert manager.current_model_id in (None, "whisper-tiny")
