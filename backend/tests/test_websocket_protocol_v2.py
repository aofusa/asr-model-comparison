"""
TDD tests for the refined, more practical WebSocket streaming protocol.

These tests define the expected behavior for real-world use with Qwen3-ASR and Voxtral.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


def _mock_qwen_loader(text: str = "ok"):
    fake_backend = MagicMock()
    fake_backend.transcribe.return_value = {
        "text": text,
        "processing_time_seconds": 0.01,
        "chunks": [{"text": text}] if text else [],
        "language": "ja",
    }

    async def fake_loader(_model_id: str):
        return fake_backend

    return patch("app.api.routers.transcribe.create_qwen3_loader", return_value=fake_loader)


def _receive_until_ready(ws) -> dict:
    """Consume model_progress frames and return the ready frame."""
    while True:
        message = ws.receive_json()
        if message.get("type") == "ready":
            return message
        assert message.get("type") == "model_progress"


def test_refined_protocol_full_flow():
    """
    Practical protocol:
    1. Client connects
    2. Client sends detailed config (including advanced params)
    3. Server replies with "ready" after loading model once
    4. Client sends multiple binary chunks
    5. Server returns structured results with processing time, chunks, etc.
    """
    client = TestClient(app)

    with _mock_qwen_loader("こんにちは"):
        with client.websocket_connect("/api/ws/transcribe") as ws:
            # Step 2: Send rich config
            config = {
                "type": "config",
                "model_id": "qwen3-asr-0.6b",
                "language": "ja",
                "beam_size": 6,
                "temperature": 0.0,
                "repetition_penalty": 1.15,
                "use_dedicated_class": True,
                "return_timestamps": True,
            }
            ws.send_json(config)

            # Step 3: Expect ready
            ready = _receive_until_ready(ws)
            assert ready["type"] == "ready"
            assert ready["model_id"] == "qwen3-asr-0.6b"

            # Step 4: Send chunks
            ws.send_bytes(b"chunk-1-data")
            result1 = ws.receive_json()

            ws.send_bytes(b"chunk-2-data")
            result2 = ws.receive_json()

            # Step 5: Check structured results
            assert result1["type"] == "transcription"
            assert "text" in result1
            assert "chunks" in result1
            assert "processing_time_seconds" in result1

            # Context should be carried over
            assert result2["type"] == "transcription"


def test_client_can_send_end_signal():
    """Client should be able to cleanly end the stream."""
    client = TestClient(app)

    with _mock_qwen_loader("last"):
        with client.websocket_connect("/api/ws/transcribe") as ws:
            ws.send_json({"type": "config", "model_id": "qwen3-asr-0.6b"})
            _receive_until_ready(ws)

            ws.send_bytes(b"last-chunk")
            ws.receive_json()

            # Send end signal
            ws.send_json({"type": "end"})
            final = ws.receive_json()

            assert final["type"] in ("final", "transcription", "end_ack")


def test_structured_error_messages():
    """Server should return structured, actionable error messages."""
    client = TestClient(app)

    with _mock_qwen_loader(""):
        with client.websocket_connect("/api/ws/transcribe") as ws:
            ws.send_json({"type": "config", "model_id": "qwen3-asr-0.6b"})
            _receive_until_ready(ws)

            # Send invalid data
            ws.send_bytes(b"")  # empty chunk

            response = ws.receive_json()
            # Should be an error, not crash
            assert response.get("type") == "error"
            assert "message" in response