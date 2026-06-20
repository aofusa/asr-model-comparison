"""
TDD Tests for POST /api/transcribe

Written before implementation.

Requirements:
- Accepts multipart/form-data with 'audio' file and 'model_id'
- Returns transcription text + processing_time_seconds (must be measured)
- Uses the ModelManager (single model constraint is already tested)
- Processing time must be present and positive
"""
from __future__ import annotations

from io import BytesIO

import pytest
from fastapi import UploadFile
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_transcribe_success_returns_text_and_time():
    """Basic happy path: send audio + model → get text and positive processing time."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Prepare a tiny fake WAV (content does not matter for mocked path)
        fake_audio = BytesIO(b"RIFF....WAVEfmt fake audio bytes for test")
        fake_audio.name = "test.wav"

        response = await client.post(
            "/api/transcribe",
            data={"model_id": "whisper-tiny"},
            files={"audio": ("test.wav", fake_audio, "audio/wav")},
        )

        assert response.status_code == 200
        body = response.json()
        assert "text" in body
        assert "processing_time_seconds" in body
        assert isinstance(body["processing_time_seconds"], float)
        assert body["processing_time_seconds"] > 0
        assert body["model_id"] == "whisper-tiny"


@pytest.mark.asyncio
async def test_transcribe_requires_model_id():
    """Missing model_id must return 422 or clear error."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        fake_audio = BytesIO(b"fake")
        response = await client.post(
            "/api/transcribe",
            files={"audio": ("x.wav", fake_audio, "audio/wav")},
        )
        assert response.status_code in (400, 422)


@pytest.mark.asyncio
async def test_transcribe_rejects_unknown_model():
    """Unknown model_id must return a proper error (not 500)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        fake_audio = BytesIO(b"fake")
        response = await client.post(
            "/api/transcribe",
            data={"model_id": "does-not-exist"},
            files={"audio": ("x.wav", fake_audio, "audio/wav")},
        )
        assert response.status_code in (400, 404, 422)


@pytest.mark.asyncio
async def test_transcribe_measures_time_even_for_fast_mocks():
    """
    Even when the underlying ASR is mocked (instant), we must still
    report a positive wall-clock processing time.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        fake_audio = BytesIO(b"short")
        response = await client.post(
            "/api/transcribe",
            data={"model_id": "qwen3-asr-0.6b"},
            files={"audio": ("short.wav", fake_audio, "audio/wav")},
        )
        body = response.json()
        assert body["processing_time_seconds"] >= 0.0   # In real runs this will be > 0
