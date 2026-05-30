"""
Factory functions to create real ASR backend loaders for ModelManager.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from app.services.asr_backends.whisper_backend import WhisperBackend


def create_whisper_loader(
    device: str = "cpu",
    compute_type: str = "int8",
) -> Callable[[str], Awaitable[Any]]:
    """
    Returns an async loader function suitable for ModelManager.

    Usage:
        manager = ModelManager(model_loader=create_whisper_loader())
        await manager.load_model("whisper-tiny")
    """

    async def whisper_loader(internal_name: str) -> WhisperBackend:
        # internal_name here is our public model_id (e.g. "whisper-tiny")
        # because we pass the public id from load_model
        backend = WhisperBackend(
            model_id=internal_name,
            device=device,
            compute_type=compute_type,
        )
        # Trigger lazy load immediately so errors happen at load time
        backend._ensure_loaded()  # type: ignore[attr-defined]
        return backend

    return whisper_loader


async def whisper_unloader(instance: Any) -> None:
    """Standard unloader for WhisperBackend instances (with strong memory cleanup)."""
    if hasattr(instance, "unload"):
        instance.unload()
    else:
        # fallback for any other object
        try:
            del instance
        except Exception:
            pass

    # Extra aggressive cleanup for real model usage on memory-constrained hardware
    import gc
    gc.collect()
    try:
        import torch
        if hasattr(torch, "cuda"):
            torch.cuda.empty_cache()
    except Exception:
        pass
