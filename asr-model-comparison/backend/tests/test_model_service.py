"""
TDD Tests for the core ModelManager service.

Critical requirement (CLAUDE.md + specs):
- NEVER keep more than one ASR model in memory at the same time.
- Switching models MUST unload the previous model before loading the new one.

These tests are written BEFORE the service implementation.
We use dependency injection / callable loaders so that no real ASR libraries
are required during unit testing.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.model_manager import ModelManager


@pytest.fixture
def mock_loaders():
    """Provide fake loader callables that record calls and simulate loading."""
    load_history = []

    async def fake_whisper_loader(model_id: str):
        load_history.append(("load", model_id))
        mock_model = MagicMock(name=f"mock-{model_id}")
        mock_model.transcribe = AsyncMock(return_value={"text": f"[{model_id}] transcription"})
        return mock_model

    async def fake_unloader(model_instance):
        load_history.append(("unload", getattr(model_instance, "name", "unknown")))
        return None

    return {
        "load": fake_whisper_loader,
        "unload": fake_unloader,
        "history": load_history,
    }


@pytest.mark.asyncio
async def test_initial_state_no_model_loaded():
    """Manager must start with no model in memory."""
    manager = ModelManager()
    assert manager.current_model_id is None
    assert manager.is_model_loaded is False


@pytest.mark.asyncio
async def test_load_model_sets_current_id(mock_loaders):
    """Loading a valid model must update current_model_id."""
    manager = ModelManager(model_loader=mock_loaders["load"])
    await manager.load_model("whisper-tiny")

    assert manager.current_model_id == "whisper-tiny"
    assert manager.is_model_loaded is True


@pytest.mark.asyncio
async def test_switching_models_unloads_previous(mock_loaders):
    """
    THE MOST IMPORTANT TEST.

    Loading a different model must first unload the currently loaded model.
    This enforces the single-model-in-memory constraint.
    """
    manager = ModelManager(
        model_loader=mock_loaders["load"],
        model_unloader=mock_loaders["unload"],
    )

    await manager.load_model("whisper-tiny")
    await manager.load_model("qwen3-asr-0.6b")

    history = mock_loaders["history"]
    # We expect: load tiny, then (unload tiny + load qwen)
    unload_events = [h for h in history if h[0] == "unload"]
    assert len(unload_events) >= 1
    assert manager.current_model_id == "qwen3-asr-0.6b"


@pytest.mark.asyncio
async def test_loading_same_model_is_idempotent(mock_loaders):
    """Loading the exact same model ID again should not trigger unload + reload."""
    manager = ModelManager(
        model_loader=mock_loaders["load"],
        model_unloader=mock_loaders["unload"],
    )

    await manager.load_model("whisper-medium")
    await manager.load_model("whisper-medium")

    history = mock_loaders["history"]
    load_count = sum(1 for h in history if h[0] == "load")
    # Only one actual load should have happened
    assert load_count == 1
    assert manager.current_model_id == "whisper-medium"


@pytest.mark.asyncio
async def test_unload_clears_state(mock_loaders):
    """Explicit unload must clear the current model."""
    manager = ModelManager(
        model_loader=mock_loaders["load"],
        model_unloader=mock_loaders["unload"],
    )

    await manager.load_model("voxtral-mini-4b")
    await manager.unload_current()

    assert manager.current_model_id is None
    assert manager.is_model_loaded is False


@pytest.mark.asyncio
async def test_load_unknown_model_raises_error():
    """Requesting a model ID that is not in the registry must raise a clear error."""
    manager = ModelManager()

    with pytest.raises(ValueError, match="Unknown model"):
        await manager.load_model("non-existent-model-xyz")


# ------------------------------------------------------------------
# Strengthened real-model memory management tests (TDD for reinforcement)
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unload_is_idempotent_and_safe(mock_loaders):
    """Calling unload multiple times must be safe and not crash."""
    manager = ModelManager(
        model_loader=mock_loaders["load"],
        model_unloader=mock_loaders["unload"],
    )

    await manager.load_model("whisper-tiny")
    await manager.unload_current()
    await manager.unload_current()  # second call must not raise
    await manager.unload_current()

    assert manager.current_model_id is None
    assert manager.is_model_loaded is False


@pytest.mark.asyncio
async def test_switching_models_always_unloads_previous_even_on_error(mock_loaders):
    """
    Even if loading a new model fails, the previous model must have been unloaded.
    This protects limited memory devices (ROG Ally X 24GB, M4 24GB).
    """
    async def failing_loader(name: str):
        # The manager passes internal name; simulate failure for the qwen internal id
        if "Qwen" in name or "qwen" in name.lower():
            raise RuntimeError("Simulated model load failure for Qwen3")
        return await mock_loaders["load"](name)

    manager = ModelManager(
        model_loader=failing_loader,
        model_unloader=mock_loaders["unload"],
    )

    await manager.load_model("whisper-small")

    with pytest.raises(RuntimeError, match="Simulated"):
        await manager.load_model("qwen3-asr-0.6b")

    # Previous model must have been unloaded before attempting the failing load
    unload_events = [h for h in mock_loaders["history"] if h[0] == "unload"]
    assert any("small" in str(e) for e in unload_events)
    assert manager.current_model_id is None
    assert manager.is_model_loaded is False
