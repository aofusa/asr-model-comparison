"""
Voxtral Backend Adapter (Skeleton / Placeholder)

Voxtral Mini 4B Realtime support is not yet implemented.

This will eventually use Mistral's Voxtral via transformers or their dedicated runtime.
"""
from __future__ import annotations

from typing import Any

from app.services.asr_backends.base import ASRBackend


class VoxtralBackend:
    """
    Placeholder adapter for Voxtral models.
    """

    def __init__(self, model_id: str, **kwargs):
        self.model_id = model_id
        self.family = "voxtral"
        self._kwargs = kwargs

    def _raise_not_implemented(self):
        raise NotImplementedError(
            f"Real-time inference for Voxtral ({self.model_id}) is not yet implemented.\n\n"
            "Current options:\n"
            "  1. Use Whisper models which have production-ready support via faster-whisper.\n"
            "  2. Voxtral is a heavier 4B model — even on M4 it benefits from optimized runtimes.\n"
            "  3. Implement this adapter once the official Voxtral inference path is stable.\n\n"
            "See the project roadmap and backend/INSTALL.md."
        )

    def transcribe(self, audio: bytes | str, **kwargs: Any) -> dict[str, Any]:
        self._raise_not_implemented()

    def unload(self) -> None:
        pass
