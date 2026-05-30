"""
Voxtral Real Backend Implementation

Uses transformers pipeline for Mistral's Voxtral Mini models.
Provides actual model inference for "voxtral-mini-4b".
"""
from __future__ import annotations

import tempfile
import os
from typing import Any

import torch
from transformers import pipeline

from app.services.asr_backends.base import ASRBackend


class VoxtralBackend:
    """
    Real adapter for Voxtral models using transformers.

    Currently targets Voxtral Mini variants.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        torch_dtype: torch.dtype | None = None,
        quantization: str = "none",
        use_dedicated_class: bool = True,
        **kwargs: Any,
    ):
        self.model_id = model_id
        self.family = "voxtral"
        self._device = device
        self._quantization = quantization
        self._use_dedicated_class = use_dedicated_class
        self._kwargs = kwargs

        if quantization == "4bit" and device != "cuda":
            print(f"[VoxtralBackend] Warning: 4-bit requested on {device}. Falling back.")
            self._quantization = "none"

        default_dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        self._torch_dtype = torch_dtype or default_dtype

        self._pipe = None
        self._hf_model_id = self._resolve_hf_model_id(model_id)

    def _resolve_hf_model_id(self, model_id: str) -> str:
        mapping = {
            "voxtral-mini-4b": "mistralai/Voxtral-Mini-3B-2507",  # adjust if official id changes
        }
        return mapping.get(model_id, model_id)

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return

        print(f"[VoxtralBackend] Loading real model (stable pipeline mode): {self.model_id} ({self._hf_model_id}) on {self._device}...")

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self._hf_model_id,
            device=0 if self._device == "cuda" else -1,
            torch_dtype=self._torch_dtype,
            model_kwargs={"low_cpu_mem_usage": True},
        )
        print(f"[VoxtralBackend] Model {self.model_id} loaded successfully via pipeline.")

    def transcribe(self, audio: bytes | str, **kwargs: Any) -> dict[str, Any]:
        self._ensure_loaded()

        if isinstance(audio, (bytes, bytearray)):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio)
                tmp_path = tmp.name
            try:
                return self._run_pipeline(tmp_path, **kwargs)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        else:
            return self._run_pipeline(str(audio), **kwargs)

    def _run_pipeline(self, audio_path: str, **kwargs: Any) -> dict[str, Any]:
        assert self._pipe is not None
        generate_kwargs = kwargs.get("generate_kwargs", {})
        if "language" in kwargs and kwargs["language"]:
            generate_kwargs["language"] = kwargs["language"]

        result = self._pipe(
            audio_path,
            return_timestamps=kwargs.get("return_timestamps", False),
            generate_kwargs=generate_kwargs or None,
        )
        text = result.get("text", "").strip() if isinstance(result, dict) else str(result)
        return {"text": text, "model_id": self.model_id}

    def unload(self) -> None:
        if self._pipe is not None:
            try:
                del self._pipe
            except Exception:
                pass
            self._pipe = None

        import gc
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"[VoxtralBackend] Model {self.model_id} unloaded.")
