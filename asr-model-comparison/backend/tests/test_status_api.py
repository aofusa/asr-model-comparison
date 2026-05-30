"""
TDD tests for the new model status endpoint.

Written before implementation.
"""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_status_endpoint_exists_and_returns_json():
    """The status endpoint must exist and return basic information."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)


@pytest.mark.asyncio
async def test_status_reports_loaded_model_state():
    """Status must report whether a model is loaded and which one."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/status")
        data = response.json()

        assert "loaded" in data or "current_model_id" in data
        assert "is_model_loaded" in data


@pytest.mark.asyncio
async def test_status_includes_memory_info_when_available():
    """Status should attempt to report memory usage (psutil is optional but recommended)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/status")
        data = response.json()

        assert "memory" in data
        # Either we have real numbers or a graceful note
        mem = data["memory"]
        assert "rss_mb" in mem or "note" in mem or "used_mb" in mem
