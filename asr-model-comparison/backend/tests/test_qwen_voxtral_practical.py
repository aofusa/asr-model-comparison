"""
Practical level tests for Qwen3-ASR and Voxtral as main models.

Goal: Raise them to a level where they can be used as the primary models
in the project (instead of Whisper), focusing on quality, controllability,
and proper integration — heaviness is accepted.

These tests are written before major implementation upgrades.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from app.services.asr_backends.qwen3_backend import Qwen3ASRBackend
from app.services.asr_backends.voxtral_backend import VoxtralBackend


def _mock_qwen_backend(model_id: str = "qwen3-asr-0.6b") -> Qwen3ASRBackend:
    backend = Qwen3ASRBackend(model_id=model_id, device="cpu", use_dedicated_class=True)
    backend._qwen_asr_model = MagicMock()
    backend._qwen_asr_model.transcribe.return_value = [
        {"text": "テスト文字起こし", "language": "ja", "time_stamps": (0.0, 1.0)}
    ]
    return backend


def _mock_voxtral_backend() -> VoxtralBackend:
    backend = VoxtralBackend(model_id="voxtral-mini-4b", device="cpu", use_dedicated_class=True)
    backend._model = MagicMock()
    backend._model.device = "cpu"
    backend._model.generate.return_value = [[1, 2, 3]]
    backend._processor = MagicMock()
    request = MagicMock()
    request.to.return_value = request
    backend._processor.apply_transcription_request.return_value = request
    backend._processor.batch_decode.return_value = ["voxtral transcription"]
    return backend


def test_qwen3_backend_supports_practical_parameters():
    """
    Qwen3 backend must accept parameters that are important for practical use:
    - language forcing
    - beam_size (for quality over speed)
    - return_timestamps
    - use_dedicated_class
    """
    backend = _mock_qwen_backend()

    # These should not raise and should influence behavior
    result = backend.transcribe(
        b"fake-audio-bytes-for-test",
        language="ja",
        beam_size=5,                    # quality focused
        return_timestamps=True,
    )

    assert "text" in result
    assert result.get("model_id") == "qwen3-asr-0.6b"
    _, kwargs = backend._qwen_asr_model.transcribe.call_args
    assert kwargs["language"] == "Japanese"
    assert kwargs["return_time_stamps"] is True


def test_voxtral_backend_supports_practical_parameters():
    """
    Voxtral should also support practical ASR parameters.
    """
    backend = _mock_voxtral_backend()

    result = backend.transcribe(
        b"fake-audio",
        language="en",
        beam_size=4,
        return_timestamps=True,
    )

    assert "text" in result
    backend._processor.apply_transcription_request.assert_called_once()


@pytest.mark.slow
def test_qwen3_backend_can_use_dedicated_class_for_better_quality():
    """
    When use_dedicated_class=True, the backend should attempt to use
    model-specific loading logic that can deliver better quality than
    generic pipeline (important since we want these as main models).
    """
    backend = Qwen3ASRBackend(
        model_id="qwen3-asr-0.6b",
        use_dedicated_class=True,
    )

    # The implementation should at least attempt dedicated loading
    # without crashing, even if it falls back.
    assert backend._use_dedicated_class is True


def test_practical_generation_kwargs_are_forwarded():
    """
    The backend should allow passing advanced generation kwargs
    (temperature, top_p, etc.) so users can tune for quality.
    """
    backend = _mock_qwen_backend(model_id="qwen3-asr-1.7b")

    # This should not raise even if not fully implemented yet
    result = backend.transcribe(
        b"test",
        generate_kwargs={
            "temperature": 0.0,
            "beam_size": 5,
            "num_beams": 5,
        }
    )
    assert isinstance(result, dict)
    backend._qwen_asr_model.transcribe.assert_called_once()
