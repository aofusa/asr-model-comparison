"""
Voxtral Real Backend Implementation

Uses transformers pipeline for Mistral's Voxtral Mini models.
Provides actual model inference for "voxtral-mini-4b".
"""
from __future__ import annotations

import tempfile
import os
from typing import Any

from app.services.asr_backends.base import ASRBackend
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


def _is_translation_enabled(target_language: str | None) -> bool:
    return target_language not in (None, "", "none", "auto")


class VoxtralBackend:
    """
    Real adapter for Voxtral models using transformers.

    Currently targets Voxtral Mini variants.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        torch_dtype: Any | None = None,
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
            server_log(f"[VoxtralBackend] Warning: 4-bit requested on {device}. Falling back.")
            self._quantization = "none"

        self._torch_dtype = torch_dtype

        self._pipe = None
        self._hf_model_id = self._resolve_hf_model_id(model_id)

    def _resolve_hf_model_id(self, model_id: str) -> str:
        mapping = {
            "voxtral-mini-4b": "mistralai/Voxtral-Mini-3B-2507",  # adjust if official id changes
        }
        return mapping.get(model_id, model_id)

    def _ensure_loaded(self) -> None:
        if self._pipe is not None or getattr(self, "_model", None) is not None:
            return

        server_log(f"[VoxtralBackend] Loading real model: {self.model_id} ({self._hf_model_id}) on {self._device}...")

        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Voxtral requires optional heavy dependencies: torch and transformers. "
                "Install them before loading Voxtral models."
            ) from exc

        # Voxtral is large enough that CPU float32 loading can crash the host
        # process on 16-24GB machines. The model cards recommend reduced
        # precision; bfloat16 is the safest CPU default when available.
        torch_dtype = self._torch_dtype or (
            torch.float16 if self._device in ("cuda", "mps") else torch.bfloat16
        )

        quantization_config = None
        if self._quantization == "4bit" and self._device == "cuda":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        device_map = "auto" if self._device in ("cuda", "mps") else "cpu"

        if self._use_dedicated_class:
            try:
                from transformers import AutoProcessor, VoxtralForConditionalGeneration
                server_log("[VoxtralBackend] Attempting dedicated speech model class...")
                self._processor = AutoProcessor.from_pretrained(self._hf_model_id)
                self._model = VoxtralForConditionalGeneration.from_pretrained(
                    self._hf_model_id,
                    dtype=torch_dtype,
                    device_map=device_map,
                    quantization_config=quantization_config,
                    low_cpu_mem_usage=True,
                )
                server_log("[VoxtralBackend] Loaded using dedicated class.")
                return
            except Exception as e:
                server_log(f"[VoxtralBackend] Dedicated class failed ({type(e).__name__}). Falling back to pipeline...")

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self._hf_model_id,
            device=0 if self._device == "cuda" else -1,
            torch_dtype=torch_dtype,
            model_kwargs={"low_cpu_mem_usage": True},
        )
        server_log(f"[VoxtralBackend] Model {self.model_id} loaded via pipeline (fallback).")

    def transcribe(self, audio: bytes | str, **kwargs: Any) -> dict[str, Any]:
        try:
            self._ensure_loaded()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Voxtral model '{self.model_id}'. "
                "Large speech models can fail on first download or due to memory limits. "
                f"Original error: {e}"
            ) from e

        if not audio:
            raise ValueError("Empty audio data provided for transcription.")

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
        if getattr(self, "_model", None) is not None and getattr(self, "_processor", None) is not None:
            # Deepened dedicated class path for practical real-time Japanese use
            language = kwargs.get("language")
            target_language = kwargs.get("target_language")
            translation_enabled = _is_translation_enabled(target_language)
            previous_text = str(kwargs.get("previous_text", "") or "").strip()

            asr_inputs = self._processor.apply_transcription_request(
                language=language or "auto",
                audio=audio_path,
                model_id=self._hf_model_id,
                return_tensors="pt",
            )
            asr_inputs = asr_inputs.to(self._model.device)

            asr_gen_kwargs = {
                "max_new_tokens": 512,
                "num_beams": kwargs.get("beam_size", 5),
                "temperature": kwargs.get("temperature", 0.0),
                "repetition_penalty": kwargs.get("repetition_penalty", 1.12),
                "do_sample": False,
            }

            import torch
            with torch.no_grad():
                asr_generated_ids = self._model.generate(**asr_inputs, **asr_gen_kwargs)

            try:
                asr_generated_ids = asr_generated_ids[:, asr_inputs.input_ids.shape[1]:]
            except Exception:
                # Unit tests use simple list/MagicMock values instead of tensors.
                pass
            transcript_text = self._processor.batch_decode(asr_generated_ids, skip_special_tokens=True)[0].strip()

            if translation_enabled:
                target_name = _language_name(target_language)
                source_hint = (
                    f"The source speech language is {_language_name(language)}. "
                    if language not in (None, "", "auto")
                    else ""
                )
                context_hint = (
                    f"Previous transcript context: {previous_text}\n"
                    if previous_text
                    else ""
                )
                prompt = (
                    f"{source_hint}{context_hint}"
                    f"Original transcript: {transcript_text}\n"
                    f"Translate the original transcript into {target_name}. "
                    "Return only the translated text."
                )
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "path": audio_path},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

                inputs = self._processor.apply_chat_template(conversation)
                inputs = inputs.to(self._model.device)

                gen_kwargs = {
                    "max_new_tokens": 512,
                    "num_beams": kwargs.get("beam_size", 5),
                    "temperature": kwargs.get("temperature", 0.0),
                    "repetition_penalty": kwargs.get("repetition_penalty", 1.12),
                    "do_sample": False,
                }

                with torch.no_grad():
                    generated_ids = self._model.generate(**inputs, **gen_kwargs)

                try:
                    generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
                except Exception:
                    pass

                text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                result = {
                    "text": text,
                    "transcript_text": transcript_text,
                    "translated_text": text,
                    "model_id": self.model_id,
                    "language": language,
                    "target_language": target_language,
                }
                if kwargs.get("return_timestamps"):
                    result["chunks"] = [{"text": transcript_text, "timestamp": (0.0, None)}]
                return result

            result = {
                "text": transcript_text,
                "transcript_text": transcript_text,
                "translated_text": None,
                "model_id": self.model_id,
                "language": language,
                "target_language": target_language,
            }

            if kwargs.get("return_timestamps"):
                # Better structured timestamps for real-time applications
                # (In a more advanced implementation we could parse forced alignment or use model outputs)
                result["chunks"] = [
                    {
                        "text": transcript_text,
                        "timestamp": (0.0, None)
                    }
                ]

            return result

        # Pipeline fallback (with Japanese quality defaults)
        assert self._pipe is not None
        generate_kwargs = kwargs.get("generate_kwargs", {}) or {}

        language = kwargs.get("language")
        if language == "ja":
            if "num_beams" not in generate_kwargs:
                generate_kwargs["num_beams"] = kwargs.get("beam_size", 5)
            generate_kwargs.setdefault("temperature", 0.0)
            generate_kwargs.setdefault("repetition_penalty", 1.12)
        else:
            if "num_beams" not in generate_kwargs:
                generate_kwargs["num_beams"] = kwargs.get("beam_size", 4)

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
            "target_language": kwargs.get("target_language"),
        }
        if kwargs.get("return_timestamps") and isinstance(result, dict) and "chunks" in result:
            out["chunks"] = result.get("chunks", [])
        return out

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
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        server_log(f"[VoxtralBackend] Model {self.model_id} unloaded.")
