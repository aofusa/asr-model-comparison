"""
Factory functions to create real ASR backend loaders for ModelManager.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from app.services.asr_backends.qwen3_backend import Qwen3ASRBackend
from app.services.asr_backends.voxtral_backend import VoxtralBackend
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

    async def whisper_loader(model_id: str) -> WhisperBackend:
        # model_id is the public id (e.g. "whisper-tiny")
        backend = WhisperBackend(
            model_id=model_id,
            device=device,
            compute_type=compute_type,
        )
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


def create_qwen3_loader(**kwargs) -> Callable[[str], Awaitable[Any]]:
    async def qwen_loader(model_id: str) -> Qwen3ASRBackend:
        backend = Qwen3ASRBackend(model_id=model_id, **kwargs)
        # This will immediately raise a clear, helpful NotImplementedError
        backend.transcribe(b"")
        return backend
    return qwen_loader


def create_voxtral_loader(**kwargs) -> Callable[[str], Awaitable[Any]]:
    async def voxtral_loader(model_id: str) -> VoxtralBackend:
        backend = VoxtralBackend(model_id=model_id, **kwargs)
        backend.transcribe(b"")
        return backend
    return voxtral_loader
