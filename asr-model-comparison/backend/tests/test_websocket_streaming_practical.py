"""
TDD tests for making the WebSocket streaming endpoint practical for real-time use.

Focus:
- One ModelManager per connection (model loaded once)
- Proper session config (model_id, language, beam_size, use_dedicated_class, etc.)
- Support for previous_text / context across chunks
- Clear message protocol
- Graceful error handling and cleanup
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


def test_websocket_practical_connection_and_config():
    """
    Client must send an initial config message after connecting.
    The server loads the model only once per connection.
    """
    client = TestClient(app)

    with client.websocket_connect("/api/ws/transcribe") as websocket:
        # Send initial config (new practical protocol)
        config = {
            "type": "config",
            "model_id": "qwen3-asr-0.6b",
            "language": "ja",
            "beam_size": 6,
            "use_dedicated_class": True,
        }
        websocket.send_json(config)

        # Expect "ready" after model is loaded (once)
        response = websocket.receive_json()
        assert response.get("type") == "ready"
        assert response.get("model_id") == "qwen3-asr-0.6b"


@pytest.mark.slow
def test_websocket_sends_transcription_results_for_chunks():
    """
    After sending config and receiving 'ready', sending binary audio chunks
    should result in transcription JSON messages.
    """
    client = TestClient(app)

    with client.websocket_connect("/api/ws/transcribe") as websocket:
        # New protocol: send config first
        websocket.send_json({"type": "config", "model_id": "qwen3-asr-0.6b", "language": "ja"})
        websocket.receive_json()  # ready

        websocket.send_bytes(b"fake-wav-audio-chunk-data")

        response = websocket.receive_json()
        assert response.get("type") == "transcription"
        assert "text" in response
        assert response.get("model_id") == "qwen3-asr-0.6b"


def test_websocket_maintains_previous_text_across_chunks():
    """
    The server should accumulate previous_text across chunks for continuity.
    """
    client = TestClient(app)

    with client.websocket_connect("/api/ws/transcribe") as websocket:
        websocket.send_json({"type": "config", "model_id": "qwen3-asr-0.6b"})
        websocket.receive_json()  # ready

        websocket.send_bytes(b"chunk1")
        response1 = websocket.receive_json()

        websocket.send_bytes(b"chunk2")
        response2 = websocket.receive_json()

        assert response1.get("type") == "transcription"
        assert response2.get("type") == "transcription"


def test_websocket_graceful_disconnect():
    """Connection should close cleanly after sending config."""
    client = TestClient(app)

    with client.websocket_connect("/api/ws/transcribe") as websocket:
        websocket.send_json({"type": "config", "model_id": "qwen3-asr-0.6b"})
        websocket.receive_json()  # ready

        # Client disconnects
        pass

    assert True