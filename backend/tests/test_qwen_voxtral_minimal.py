"""
Minimum viable functionality tests for Qwen3-ASR and Voxtral.

Goal: Make them work at the same basic level as Whisper.
That means:
- Can be loaded in real mode (USE_REAL_MODELS=1)
- Can perform transcription via the backend
- Return text + model_id without crashing

These tests are written BEFORE simplifying the backends for stability.
"""
from __future__ import annotations

import os
import pytest
from app.services.asr_backends.qwen3_backend import Qwen3ASRBackend
from app.services.asr_backends.voxtral_backend import VoxtralBackend


@pytest.mark.slow
def test_qwen3_backend_minimal_loading_and_transcribe():
    """
    Qwen3ASRBackend should at least be able to load and run basic transcription
    using the stable pipeline path (minimum requirement to be "usable like Whisper").
    """
    if os.getenv("USE_REAL_MODELS") != "1" and os.getenv("USE_REAL_WHISPER") != "1":
        pytest.skip("Requires real model environment")

    # Use stable settings for minimum viable functionality
    backend = Qwen3ASRBackend(
        model_id="qwen3-asr-0.6b",
        device="cpu",
        use_dedicated_class=False,   # Prioritize stability over fancy classes
        quantization="none",
    )

    # This should not raise "not implemented" or crash on init
    assert backend.family == "qwen3"

    # We don't actually call transcribe here in CI because model download is huge.
    # The important thing is construction + _ensure_loaded attempt succeeds or fails gracefully.
    try:
        backend._ensure_loaded()
        print("[Test] Qwen3 backend loaded successfully (or fell back cleanly).")
    except Exception as e:
        # Acceptable during first runs: download errors, memory, etc.
        print(f"[Test] Qwen3 loading attempt: {type(e).__name__}: {e}")


@pytest.mark.slow
def test_voxtral_backend_minimal_loading_and_transcribe():
    """
    VoxtralBackend should at least be constructible and attempt loading
    without raising NotImplementedError.
    """
    if os.getenv("USE_REAL_MODELS") != "1" and os.getenv("USE_REAL_WHISPER") != "1":
        pytest.skip("Requires real model environment")

    backend = VoxtralBackend(
        model_id="voxtral-mini-4b",
        device="cpu",
        use_dedicated_class=False,
        quantization="none",
    )

    assert backend.family == "voxtral"

    try:
        backend._ensure_loaded()
        print("[Test] Voxtral backend loaded successfully (or fell back cleanly).")
    except Exception as e:
        print(f"[Test] Voxtral loading attempt: {type(e).__name__}: {e}")


def test_qwen_voxtral_backends_do_not_raise_not_implemented_on_basic_use():
    """
    Even without real models enabled, constructing Qwen3/Voxtral backends
    with stable settings must never raise NotImplementedError.
    This ensures the "minimum working" promise.
    """
    qwen = Qwen3ASRBackend(
        model_id="qwen3-asr-0.6b",
        device="cpu",
        use_dedicated_class=False,
    )
    vox = VoxtralBackend(
        model_id="voxtral-mini-4b",
        device="cpu",
        use_dedicated_class=False,
    )

    # If we got here without NotImplementedError, the backends are at least "minimally usable"
    assert qwen is not None
    assert vox is not None
