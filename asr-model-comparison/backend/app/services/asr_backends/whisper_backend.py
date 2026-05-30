"""
Real Whisper backend using faster-whisper.

This adapter implements the ASRBackend protocol and is the first
real model integration for the project.
"""
from __future__ import annotations

from typing import Any

from app.services.asr_backends.base import ASRBackend


class WhisperBackend:
    """
    Adapter for OpenAI Whisper models via faster-whisper.

    Supports: tiny, small, medium, large-v3-turbo (and others)
    Optimized for CPU (int8) by default — ideal for ROG Ally X (AMD).
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        compute_type: str = "int8",
        download_root: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.family = "whisper"
        self._device = device
        self._compute_type = compute_type
        self._download_root = download_root

        self._internal_size = self._map_to_whisper_size(model_id)
        self._model: Any = None  # faster_whisper.WhisperModel

    def _map_to_whisper_size(self, model_id: str) -> str:
        mapping = {
            "whisper-tiny": "tiny",
            "whisper-small": "small",
            "whisper-medium": "medium",
            "whisper-large-v3-turbo": "large-v3-turbo",
        }
        if model_id not in mapping:
            raise ValueError(f"Unsupported Whisper model_id for this backend: {model_id}")
        return mapping[model_id]

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        from faster_whisper import WhisperModel

        print(f"[WhisperBackend] Loading real model: {self.model_id} ({self._internal_size}) on {self._device}...")
        self._model = WhisperModel(
            self._internal_size,
            device=self._device,
            compute_type=self._compute_type,
            download_root=self._download_root,
        )
        print(f"[WhisperBackend] Model {self.model_id} loaded successfully.")

    def transcribe(self, audio: bytes | str, **kwargs: Any) -> dict[str, Any]:
        """
        Run transcription using the real faster-whisper model.

        Returns dict with at least: text, language, duration, segments (optional)
        """
        self._ensure_loaded()

        # faster-whisper accepts file path or file-like. We support both.
        if isinstance(audio, (bytes, bytearray)):
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio)
                tmp_path = tmp.name
            try:
                return self._do_transcribe(tmp_path, **kwargs)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        else:
            # Assume it's a path
            return self._do_transcribe(str(audio), **kwargs)

    def _do_transcribe(self, audio_path: str, **kwargs: Any) -> dict[str, Any]:
        assert self._model is not None

        # Reasonable defaults for our use case
        beam_size = kwargs.get("beam_size", 1)
        language = kwargs.get("language", None)  # let it auto-detect unless specified

        segments, info = self._model.transcribe(
            audio_path,
            beam_size=beam_size,
            language=language,
            vad_filter=kwargs.get("vad_filter", True),
        )

        text = " ".join(segment.text.strip() for segment in segments).strip()

        result = {
            "text": text,
            "language": info.language,
            "language_probability": getattr(info, "language_probability", None),
            "duration": getattr(info, "duration", None),
            "model_id": self.model_id,
        }

        # Optionally include detailed segments
        if kwargs.get("return_segments", False):
            result["segments"] = [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text.strip(),
                }
                for s in segments
            ]

        return result

    def unload(self) -> None:
        """Release the model to free memory (important for single-model rule on 24GB devices)."""
        if self._model is not None:
            try:
                # faster-whisper / CTranslate2 models benefit from explicit deletion
                del self._model
            except Exception:
                pass
            self._model = None

            import gc
            gc.collect()

            # Best-effort: clear any torch CUDA cache (harmless on CPU)
            try:
                import torch
                if hasattr(torch, "cuda"):
                    torch.cuda.empty_cache()
            except Exception:
                pass

            print(f"[WhisperBackend] Model {self.model_id} unloaded and memory cleaned.")
