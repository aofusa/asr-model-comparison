"""
TDD Test: GET /api/models endpoint
This test is written BEFORE any implementation (per strict TDD requirement).

It defines the expected contract for the model list API.
"""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_get_models_returns_list():
    """Model list endpoint must return a non-empty list of model descriptors."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0


@pytest.mark.asyncio
async def test_get_models_contains_required_fields():
    """Each model entry must contain the minimum required metadata fields."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/models")
        data = response.json()

        for model in data:
            assert "id" in model
            assert "name" in model
            assert "family" in model  # whisper | qwen3 | voxtral
            assert "size" in model or "parameters" in model


@pytest.mark.asyncio
async def test_get_models_includes_all_three_families():
    """The platform must expose models from all three supported families."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/models")
        data = response.json()

        families = {m.get("family") for m in data}
        assert "whisper" in families
        assert "qwen3" in families
        assert "voxtral" in families


@pytest.mark.asyncio
async def test_get_models_no_duplicate_ids():
    """Model IDs must be unique."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/models")
        data = response.json()

        ids = [m["id"] for m in data]
        assert len(ids) == len(set(ids))
