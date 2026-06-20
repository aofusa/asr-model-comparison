"""
Common interface for all ASR backends.

This protocol lets the ModelManager (and transcription service) work
with any ASR implementation without knowing the concrete library
(faster-whisper, transformers, official qwen-asr, etc.).
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ASRBackend(Protocol):
    """Minimal interface every ASR adapter must implement."""

    model_id: str          # Our public id, e.g. "whisper-tiny"
    family: str            # "whisper", "qwen3", "voxtral"

    def transcribe(
        self,
        audio: bytes | str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Transcribe audio and return at least:
            {"text": str, "language": str | None, ...}
        """
        ...

    def unload(self) -> None:
        """Release GPU/CPU memory (important for single-model constraint)."""
        ...
