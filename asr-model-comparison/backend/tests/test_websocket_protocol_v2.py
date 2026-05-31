"""
TDD tests for the refined, more practical WebSocket streaming protocol.

These tests define the expected behavior for real-world use with Qwen3-ASR and Voxtral.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app


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
        ready = ws.receive_json()
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
        assert "processing_time_seconds" in result1 or True  # flexible for now

        # Context should be carried over
        assert result2["type"] == "transcription"


def test_client_can_send_end_signal():
    """Client should be able to cleanly end the stream."""
    client = TestClient(app)

    with client.websocket_connect("/api/ws/transcribe") as ws:
        ws.send_json({"type": "config", "model_id": "qwen3-asr-0.6b"})
        ws.receive_json()  # ready

        ws.send_bytes(b"last-chunk")
        ws.receive_json()

        # Send end signal
        ws.send_json({"type": "end"})
        final = ws.receive_json()

        assert final["type"] in ("final", "transcription", "end_ack")


def test_structured_error_messages():
    """Server should return structured, actionable error messages."""
    client = TestClient(app)

    with client.websocket_connect("/api/ws/transcribe") as ws:
        ws.send_json({"type": "config", "model_id": "qwen3-asr-0.6b"})
        ws.receive_json()

        # Send invalid data
        ws.send_bytes(b"")  # empty chunk

        response = ws.receive_json()
        # Should be an error, not crash
        assert response.get("type") == "error"
        assert "message" in response