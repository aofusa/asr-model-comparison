"""
Qwen3-ASR Real Backend Implementation

Uses the official qwen_asr backend for Qwen3-ASR 0.6B and 1.7B variants.

Note: First run will download the model (several GB). 
Recommended on machines with at least 16-24GB RAM.
"""
from __future__ import annotations

import importlib
import os
import tempfile
from typing import Any

from app.services.asr_backends.base import ASRBackend
from app.services.translation import translate_text
from app.utils.server_logging import server_log


LANGUAGE_NAMES = {
    "ja": "Japanese",
    "en": "English",
    "zh": "Chinese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "tr": "Turkish",
    "nl": "Dutch",
    "pl": "Polish",
    "sv": "Swedish",
}


def _language_name(code: str | None) -> str:
    return LANGUAGE_NAMES.get(code or "", "the detected language")


def _asr_output_language(language: str | None) -> str | None:
    """Qwen3-ASR uses the language argument to force the ASR output language."""
    if language in (None, "", "auto"):
        return None
    return _language_name(language)


def _is_translation_enabled(target_language: str | None) -> bool:
    return target_language not in (None, "", "none", "auto")


class Qwen3ASRBackend:
    """
    Real adapter for Qwen3-ASR models using transformers.

    Supports:
      - qwen3-asr-0.6b
      - qwen3-asr-1.7b
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        torch_dtype: Any | None = None,
        quantization: str = "none",           # "none", "4bit", "8bit"
        use_dedicated_class: bool = True,     # use specific Qwen audio classes when possible
        **kwargs: Any,
    ):
        self.model_id = model_id
        self.family = "qwen3"
        self._device = device
        self._quantization = quantization
        self._use_dedicated_class = use_dedicated_class
        self._kwargs = kwargs

        if quantization == "4bit" and device != "cuda":
            server_log(f"[Qwen3ASRBackend] Warning: 4-bit quantization requested on {device}. Falling back.")
            self._quantization = "none"

        self._torch_dtype = torch_dtype

        self._pipe = None
        self._qwen_asr_model = None
        self._hf_model_id = self._resolve_hf_model_id(model_id)

    def _resolve_hf_model_id(self, model_id: str) -> str:
        mapping = {
            "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
            "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
        }
        return mapping.get(model_id, model_id)

    def _ensure_loaded(self) -> None:
        if (
            self._pipe is not None
            or self._qwen_asr_model is not None
            or getattr(self, "_model", None) is not None
        ):
            return

        try:
            server_log(f"[Qwen3ASRBackend] Loading real model: {self.model_id} ({self._hf_model_id}) on {self._device}...")
        except Exception:
            pass

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-ASR requires optional heavy dependencies: torch and qwen-asr. "
                "Install them before loading Qwen models."
            ) from exc

        torch_dtype = self._torch_dtype or (
            torch.float16 if self._device in ("cuda", "mps") else torch.float32
        )

        quantization_config = None
        if self._quantization == "4bit" and self._device == "cuda":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )

        device_map = "auto" if self._device in ("cuda", "mps") else "cpu"

        if self._use_dedicated_class:
            try:
                qwen3_asr_module = importlib.import_module("qwen_asr.inference.qwen3_asr")
                qwen3_asr_model_cls = getattr(qwen3_asr_module, "Qwen3ASRModel")
            except ImportError as exc:
                raise RuntimeError(
                    "Qwen3-ASR requires the official qwen-asr package. "
                    "The generic transformers ASR pipeline cannot load Qwen3-ASR checkpoints "
                    "and produces the observed `qwen3_asr` model type error. "
                    "Install qwen-asr in the backend environment, then restart the server."
                ) from exc

            model_kwargs = {
                "device_map": device_map,
                "low_cpu_mem_usage": True,
            }
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config

            # qwen_asr forwards kwargs to Transformers. Newer Transformers
            # accepts `dtype`; older versions may still expect `torch_dtype`.
            if torch_dtype is not None:
                model_kwargs["dtype"] = torch_dtype

            try:
                server_log("[Qwen3ASRBackend] Attempting official qwen_asr Qwen3ASRModel...")
                try:
                    self._qwen_asr_model = qwen3_asr_model_cls.from_pretrained(
                        self._hf_model_id,
                        max_new_tokens=self._kwargs.get("max_new_tokens", 512),
                        **model_kwargs,
                    )
                except TypeError as exc:
                    if "dtype" not in str(exc):
                        raise
                    model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
                    self._qwen_asr_model = qwen3_asr_model_cls.from_pretrained(
                        self._hf_model_id,
                        max_new_tokens=self._kwargs.get("max_new_tokens", 512),
                        **model_kwargs,
                    )
                server_log("[Qwen3ASRBackend] Successfully loaded using official qwen_asr backend.")
                return
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Qwen3-ASR with the official qwen_asr backend: {e}"
                ) from e

        raise RuntimeError(
            "Generic transformers pipeline fallback is not supported for Qwen3-ASR. "
            "Set use_dedicated_class=True and install the official qwen-asr package."
        )

    def transcribe(self, audio: bytes | str, **kwargs: Any) -> dict[str, Any]:
        try:
            self._ensure_loaded()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Qwen3-ASR model '{self.model_id}'. "
                f"This can happen with very large models on limited RAM or first-time download issues. "
                f"Original error: {e}"
            ) from e

        if not audio:
            raise ValueError("Empty audio data provided to transcription.")

        if isinstance(audio, (bytes, bytearray)):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio)
                tmp_path = tmp.name
            try:
                return self._run_inference(tmp_path, **kwargs)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        else:
            return self._run_inference(str(audio), **kwargs)

    def _run_inference(self, audio_path: str, **kwargs: Any) -> dict[str, Any]:
        # Official qwen_asr path. This is the supported Qwen3-ASR inference API;
        # generic transformers ASR pipeline does not understand qwen3_asr configs.
        if self._qwen_asr_model is not None:
            language = kwargs.get("language")
            target_language = kwargs.get("target_language")
            previous_text = kwargs.get("previous_text", "")
            output_language = _asr_output_language(language)

            try:
                result = self._qwen_asr_model.transcribe(
                    audio=audio_path,
                    context=previous_text or "",
                    language=output_language,
                    return_time_stamps=kwargs.get("return_timestamps", False),
                )
            except ValueError as exc:
                if "return_time_stamps=True requires `forced_aligner`" not in str(exc):
                    raise
                server_log(
                    "[Qwen3ASRBackend] forced_aligner is not configured; "
                    "retrying transcription without timestamps."
                )
                result = self._qwen_asr_model.transcribe(
                    audio=audio_path,
                    context=previous_text or "",
                    language=output_language,
                    return_time_stamps=False,
                )

            item = result[0] if isinstance(result, list) and result else result
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                detected_language = item.get("language") or language
                time_stamps = item.get("time_stamps") or item.get("timestamps")
            else:
                text = str(getattr(item, "text", "")).strip()
                detected_language = getattr(item, "language", None) or language
                time_stamps = getattr(item, "time_stamps", None)

            transcript_text = text
            translated_text = None
            if _is_translation_enabled(target_language) and transcript_text:
                source_language = language or detected_language
                translated = translate_text(transcript_text, source_language, target_language)
                if translated:
                    translated_text = translated

            normalized = {
                "text": translated_text or transcript_text,
                "transcript_text": transcript_text,
                "translated_text": translated_text,
                "model_id": self.model_id,
                "language": detected_language,
                "target_language": target_language,
            }

            if kwargs.get("return_timestamps") and time_stamps is not None:
                normalized["chunks"] = [{"text": transcript_text, "timestamp": time_stamps}]

            return normalized

        # Pipeline fallback (also apply Japanese quality settings)
        assert self._pipe is not None
        generate_kwargs = kwargs.get("generate_kwargs", {}) or {}

        language = kwargs.get("language")
        if language == "ja":
            if "num_beams" not in generate_kwargs and "beam_size" not in generate_kwargs:
                generate_kwargs["num_beams"] = kwargs.get("beam_size", 6)
            generate_kwargs.setdefault("temperature", 0.0)
            generate_kwargs.setdefault("repetition_penalty", 1.15)
        else:
            if "num_beams" not in generate_kwargs and "beam_size" not in generate_kwargs:
                generate_kwargs["num_beams"] = kwargs.get("beam_size", 5)

        result = self._pipe(
            audio_path,
            return_timestamps=kwargs.get("return_timestamps", False),
            generate_kwargs=generate_kwargs,
        )

        text = result.get("text", "").strip() if isinstance(result, dict) else str(result)
        out = {
            "text": text,
            "transcript_text": text,
            "translated_text": None,
            "model_id": self.model_id,
            "language": result.get("language") if isinstance(result, dict) else None,
            "target_language": kwargs.get("target_language"),
        }

        if kwargs.get("return_timestamps") and isinstance(result, dict) and "chunks" in result:
            out["chunks"] = result.get("chunks", [])

        return out

    def unload(self) -> None:
        if self._qwen_asr_model is not None:
            try:
                del self._qwen_asr_model
            except Exception:
                pass
            self._qwen_asr_model = None

        if self._pipe is not None:
            try:
                del self._pipe
            except Exception:
                pass
            self._pipe = None

        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        server_log(f"[Qwen3ASRBackend] Model {self.model_id} unloaded.")
