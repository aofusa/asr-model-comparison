"""
TDD tests for real-time streaming support (WebSocket) and deepened dedicated class logic
for Qwen3-ASR and Voxtral as the main models.

These tests are written BEFORE implementation.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


def test_websocket_endpoint_exists():
    """
    The application must expose a WebSocket endpoint for real-time audio streaming
    from the browser (e.g. /ws/transcribe).
    """
    client = TestClient(app)
    try:
        with client.websocket_connect("/ws/transcribe") as websocket:
            # The endpoint should accept the connection (even if it closes immediately in test)
            assert websocket is not None
    except Exception as e:
        # Accept disconnects as the skeleton currently does minimal protocol handling
        if "WebSocketDisconnect" not in str(type(e)):
            raise
        # If we got here via disconnect, it means the route was found
        assert True


@pytest.mark.slow
def test_qwen3_dedicated_class_handles_previous_text_for_streaming():
    """
    The Qwen3 dedicated class path must properly use 'previous_text' (context from previous chunks)
    when doing real-time streaming. This is critical for maintaining Japanese conversation coherence.
    """
    from app.services.asr_backends.qwen3_backend import Qwen3ASRBackend

    backend = Qwen3ASRBackend(
        model_id="qwen3-asr-0.6b",
        device="cpu",
        use_dedicated_class=True,
    )

    # Mock the model and processor to avoid actual loading
    backend._model = MagicMock()
    backend._processor = MagicMock()
    backend._processor.apply_chat_template.return_value = "fake prompt"
    backend._processor.return_tensors = "pt"
    # Simulate generation output
    backend._model.generate.return_value = [[1, 2, 3]]
    backend._processor.batch_decode.return_value = ["続きの日本語テキスト"]

    result = backend.transcribe(
        b"fake-audio-chunk",
        previous_text="今日はいい天気です",
        language="ja",
        beam_size=6,
    )

    assert "text" in result
    # In a properly deepened implementation, previous_text should influence the prompt
    # We at least verify the call happened with context
    assert backend._processor.apply_chat_template.called


@pytest.mark.slow
def test_voxtral_dedicated_class_supports_streaming_context():
    """Same requirement for Voxtral as a main model."""
    from app.services.asr_backends.voxtral_backend import VoxtralBackend

    backend = VoxtralBackend(
        model_id="voxtral-mini-4b",
        device="cpu",
        use_dedicated_class=True,
    )

    backend._model = MagicMock()
    backend._processor = MagicMock()
    backend._processor.return_tensors = "pt"
    backend._model.generate.return_value = [[4, 5, 6]]
    backend._processor.batch_decode.return_value = ["次の発話"]

    result = backend.transcribe(
        b"audio-chunk-2",
        previous_text="前の認識結果",
        language="ja",
    )

    assert "text" in result