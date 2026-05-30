"""
Qwen3-ASR Backend Adapter (Skeleton / Placeholder)

Full real implementation is not yet complete.

When fully implemented, this will use the official Qwen3-ASR package
or a transformers-based loader for the 0.6B / 1.7B variants.
"""
from __future__ import annotations

from typing import Any

from app.services.asr_backends.base import ASRBackend


class Qwen3ASRBackend:
    """
    Placeholder adapter for Qwen3-ASR models.

    Currently raises a clear NotImplementedError with guidance.
    """

    def __init__(self, model_id: str, **kwargs):
        self.model_id = model_id
        self.family = "qwen3"
        self._kwargs = kwargs

    def _raise_not_implemented(self):
        raise NotImplementedError(
            f"Real-time inference for Qwen3-ASR ({self.model_id}) is not yet implemented.\n\n"
            "Current options:\n"
            "  1. Use Whisper models (whisper-tiny, small, medium, large-v3-turbo) which have full support.\n"
            "  2. Install additional dependencies (transformers + qwen-asr if available) and implement this adapter.\n"
            "  3. Contribute a Qwen3ASRBackend implementation following the ASRBackend protocol.\n\n"
            "See docs and INSTALL.md for the current state of multi-model support."
        )

    def transcribe(self, audio: bytes | str, **kwargs: Any) -> dict[str, Any]:
        self._raise_not_implemented()

    def unload(self) -> None:
        # Nothing to unload in the placeholder
        pass
