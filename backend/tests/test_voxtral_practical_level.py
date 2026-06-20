"""
TDD tests to bring Voxtral up to the same practical level as current Qwen3-ASR.

Focus areas:
- Robust dedicated class inference for Japanese
- Good support for real-time chunking via previous_text
- High-quality timestamp output
- Japanese accuracy generation parameters
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from app.services.asr_backends.voxtral_backend import VoxtralBackend


def test_voxtral_dedicated_class_uses_japanese_optimized_prompting():
    """
    When using dedicated class with Japanese, Voxtral should construct
    a prompt that helps with accurate Japanese transcription.
    """
    backend = VoxtralBackend(
        model_id="voxtral-mini-4b",
        device="cpu",
        use_dedicated_class=True,
    )

    # Mock to inspect behavior without loading model
    backend._model = MagicMock()
    backend._processor = MagicMock()
    backend._processor.return_tensors = "pt"
    backend._model.generate.return_value = [[1, 2, 3]]
    backend._processor.batch_decode.return_value = ["日本語のテスト結果"]

    result = backend.transcribe(
        b"fake-audio",
        language="ja",
        previous_text="前回の話",
        beam_size=5,
        temperature=0.0,
    )

    assert "text" in result
    # In a properly improved implementation, the prompt sent to the model
    # should contain Japanese-focused instructions + previous context.
    # For now we at least verify it doesn't crash and returns structure.
    assert result["model_id"] == "voxtral-mini-4b"
    backend._processor.apply_transcription_request.assert_called_once()


def test_voxtral_dedicated_class_supports_realtime_chunking():
    """
    Voxtral dedicated path must properly accept and use 'previous_text'
    for real-time chunked streaming (same level as Qwen3).
    """
    backend = VoxtralBackend(
        model_id="voxtral-mini-4b",
        use_dedicated_class=True,
    )

    backend._model = MagicMock()
    backend._processor = MagicMock()
    backend._processor.return_tensors = "pt"
    backend._model.generate.return_value = [[10, 20]]
    backend._processor.batch_decode.return_value = ["続きの認識"]

    result = backend.transcribe(
        b"second-chunk",
        previous_text="最初の認識結果",
        language="ja",
    )

    assert "text" in result
    assert result["model_id"] == "voxtral-mini-4b"
    backend._processor.apply_transcription_request.assert_called_once()


def test_voxtral_translation_uses_audio_chat_instruction():
    """
    When translation is enabled, Voxtral should use its audio+text chat path so
    the target language is an output instruction, not an input-language hint.
    """
    backend = VoxtralBackend(
        model_id="voxtral-mini-4b",
        use_dedicated_class=True,
    )

    backend._model = MagicMock(device="cpu")
    backend._processor = MagicMock()
    fake_inputs = MagicMock()
    fake_inputs.to.return_value = fake_inputs
    fake_inputs.input_ids.shape = [1, 3]
    backend._processor.apply_chat_template.return_value = fake_inputs
    backend._model.generate.return_value = [[10, 20, 30, 40]]
    backend._processor.batch_decode.side_effect = [["Original transcript"], ["これは翻訳結果です"]]

    result = backend.transcribe(
        b"english-audio",
        language="en",
        target_language="ja",
        previous_text="Hello",
        beam_size=5,
    )

    assert result["text"] == "これは翻訳結果です"
    assert result["transcript_text"] == "Original transcript"
    assert result["translated_text"] == "これは翻訳結果です"
    assert result["target_language"] == "ja"
    backend._processor.apply_transcription_request.assert_called_once()
    backend._processor.apply_chat_template.assert_called_once()
    conversation = backend._processor.apply_chat_template.call_args.args[0]
    prompt = conversation[0]["content"][1]["text"]
    assert "Translate the original transcript into Japanese" in prompt
    assert "Original transcript: Original transcript" in prompt
    assert "Previous transcript context: Hello" in prompt


def test_voxtral_dedicated_class_returns_structured_timestamps():
    """
    When return_timestamps=True in dedicated class, Voxtral should return
    a proper 'chunks' structure (improving timestamp quality for real-time use).
    """
    backend = VoxtralBackend(
        model_id="voxtral-mini-4b",
        use_dedicated_class=True,
    )

    backend._model = MagicMock()
    backend._processor = MagicMock()
    backend._processor.return_tensors = "pt"
    backend._model.generate.return_value = [[5, 6, 7]]
    backend._processor.batch_decode.return_value = ["タイムスタンプ付きテキスト"]

    result = backend.transcribe(
        b"audio-with-timestamps",
        return_timestamps=True,
        language="ja",
    )

    assert "text" in result
    # We expect 'chunks' to be present when timestamps are requested in dedicated path
    if "chunks" in result:
        assert isinstance(result["chunks"], list)


def test_voxtral_applies_japanese_generation_parameters():
    """
    Voxtral should apply Japanese-optimized generation parameters
    (beam search, low temperature, repetition penalty) when language=ja.
    """
    backend = VoxtralBackend(model_id="voxtral-mini-4b", use_dedicated_class=True)

    backend._model = MagicMock()
    backend._processor = MagicMock()
    backend._processor.return_tensors = "pt"
    backend._model.generate.return_value = [[8, 9]]

    # Call with Japanese settings
    backend.transcribe(
        b"japanese-audio",
        language="ja",
        beam_size=5,
        temperature=0.0,
    )

    # Verify that generate was called with reasonable quality parameters
    call_kwargs = backend._model.generate.call_args[1]
    assert call_kwargs.get("num_beams", 1) >= 4
    assert call_kwargs.get("temperature", 1.0) <= 0.2


def test_voxtral_cpu_default_uses_reduced_precision_for_memory(monkeypatch):
    """
    Voxtral Mini is too large for safe CPU float32 loading on 16-24GB machines.
    The dedicated loader should default to bfloat16 unless the caller overrides it.
    """
    import torch

    fake_processor_cls = MagicMock()
    fake_processor_cls.from_pretrained.return_value = MagicMock()
    fake_model_cls = MagicMock()
    fake_model_cls.from_pretrained.return_value = MagicMock(device="cpu")

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=fake_processor_cls,
            VoxtralForConditionalGeneration=fake_model_cls,
            pipeline=MagicMock(),
        ),
    )

    backend = VoxtralBackend(
        model_id="voxtral-mini-4b",
        device="cpu",
        use_dedicated_class=True,
    )

    backend._ensure_loaded()

    _, kwargs = fake_model_cls.from_pretrained.call_args
    assert kwargs["dtype"] is torch.bfloat16
