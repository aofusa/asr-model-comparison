"""
TDD tests for advanced backend features:
- Model-specific dedicated classes (vs generic pipeline)
- 4-bit quantization support
- Device optimization branching (cpu / cuda / mps)

These tests are written BEFORE the implementation.
"""
from __future__ import annotations

import pytest

from app.services.asr_backends.qwen3_backend import Qwen3ASRBackend
from app.services.asr_backends.voxtral_backend import VoxtralBackend


def test_qwen3_backend_accepts_quantization_and_device_params():
    """Qwen3ASRBackend must accept quantization and device configuration."""
    backend = Qwen3ASRBackend(
        model_id="qwen3-asr-0.6b",
        device="cpu",
        quantization="4bit",   # should be handled gracefully on non-CUDA
        use_dedicated_class=True,
    )
    assert backend.model_id == "qwen3-asr-0.6b"
    assert hasattr(backend, "_quantization")
    assert backend._device == "cpu"


def test_voxtral_backend_accepts_quantization_and_device_params():
    """VoxtralBackend must accept the same advanced configuration."""
    backend = VoxtralBackend(
        model_id="voxtral-mini-4b",
        device="mps",           # Apple Silicon
        quantization="none",
        use_dedicated_class=True,
        torch_dtype="float16",
    )
    assert backend.family == "voxtral"


@pytest.mark.slow
def test_qwen3_backend_prefers_dedicated_class_when_requested():
    """
    When use_dedicated_class=True, the backend should attempt to use
    the model-specific class (Qwen2AudioForConditionalGeneration or equivalent)
    instead of the generic ASR pipeline.
    """
    backend = Qwen3ASRBackend(
        model_id="qwen3-asr-0.6b",
        use_dedicated_class=True,
    )
    # We don't call _ensure_loaded here (it would download the model),
    # but the init should have prepared the correct loading path.
    assert backend._use_dedicated_class is True


def test_quantization_fallback_on_non_cuda():
    """
    Requesting 4-bit quantization on CPU or MPS should not crash at init time.
    The backend must fall back gracefully (e.g. to int8 or float16).
    """
    backend = Qwen3ASRBackend(
        model_id="qwen3-asr-0.6b",
        device="cpu",
        quantization="4bit",
    )
    # Should not raise during construction
    assert backend is not None


def test_device_branching_logic_exists():
    """
    The backend must expose or use logic to choose different loading strategies
    based on device (cuda vs mps vs cpu).
    """
    for dev in ["cpu", "cuda", "mps"]:
        b = Qwen3ASRBackend(model_id="qwen3-asr-0.6b", device=dev)
        assert b._device == dev
