"""
TDD tests for static file serving and SPA fallback (manifest.json etc.).

Written BEFORE the fix per 修正指示書_静的配信_manifest_QRL_WS問題.md (Phase 1).

Current broken state:
- /manifest.json is served as text/html (SPA index fallback) instead of application/json.
- This test MUST fail in broken state and pass after middleware improvement.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


def test_manifest_json_served_correctly():
    """The manifest must be served as valid JSON at root, not as HTML fallback.

    This directly catches the 'Manifest: Line: 1, column: 1, Syntax error.' root cause.
    """
    client = TestClient(app)
    r = client.get("/manifest.json")
    assert r.status_code == 200
    content_type = r.headers.get("content-type", "")
    assert "application/json" in content_type.lower(), f"Expected JSON content-type, got: {content_type}"
    data = r.json()  # Will fail or give wrong data if HTML served
    assert data.get("name") == "AMCP - Real-time ASR"
    assert data.get("short_name") == "AMCP"


def test_spa_fallback_for_unknown_path_still_works():
    """Unknown paths (for SPA routing) must still return index.html."""
    client = TestClient(app)
    r = client.get("/this-path-does-not-exist-xyz-123")
    assert r.status_code == 200
    assert "<!DOCTYPE html>" in r.text
    # But manifest must not fall into this
    assert "ASR Real-time Comparison" in r.text or "root" in r.text.lower()


def test_root_static_files_like_manifest_not_overridden_by_spa_for_known_assets():
    """Ensure known root assets like manifest are preferred over SPA fallback."""
    client = TestClient(app)
    r = client.get("/manifest.json")
    # After fix this will be JSON; the assertion above already covers.
    # This test exists to document intent.
    assert r.status_code == 200
