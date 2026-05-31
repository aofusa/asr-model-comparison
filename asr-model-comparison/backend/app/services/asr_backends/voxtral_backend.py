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
        if self._pipe is not None or getattr(self, "_model", None) is not None:
            return

        print(f"[VoxtralBackend] Loading real model: {self.model_id} ({self._hf_model_id}) on {self._device}...")

        quantization_config = None
        if self._quantization == "4bit" and self._device == "cuda":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        device_map = "auto" if self._device in ("cuda", "mps") else "cpu"

        if self._use_dedicated_class:
            try:
                from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
                print(f"[VoxtralBackend] Attempting dedicated speech model class...")
                self._processor = AutoProcessor.from_pretrained(self._hf_model_id)
                self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self._hf_model_id,
                    torch_dtype=self._torch_dtype,
                    device_map=device_map,
                    quantization_config=quantization_config,
                    low_cpu_mem_usage=True,
                )
                print(f"[VoxtralBackend] Loaded using dedicated class.")
                return
            except Exception as e:
                print(f"[VoxtralBackend] Dedicated class failed ({type(e).__name__}). Falling back to pipeline...")

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self._hf_model_id,
            device=0 if self._device == "cuda" else -1,
            torch_dtype=self._torch_dtype,
            model_kwargs={"low_cpu_mem_usage": True},
        )
        print(f"[VoxtralBackend] Model {self.model_id} loaded via pipeline (fallback).")

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
            previous_text = kwargs.get("previous_text", "")
            language = kwargs.get("language", "ja")

            # Deepened dedicated class prompting for practical real-time Japanese use
            if language == "ja":
                base_prompt = "以下の音声を正確で自然な日本語に書き起こしてください。"
                if previous_text:
                    base_prompt += f"\n\n前の音声チャンクからの文脈（連続性を保つため）:\n{previous_text}"
            else:
                base_prompt = "Transcribe the audio below into accurate and natural English."
                if previous_text:
                    base_prompt += f"\n\nPrevious context (for continuity with previous audio chunk):\n{previous_text}"

            # Use the processor in a more deliberate way for dedicated class
            inputs = self._processor(
                audio=audio_path,
                text=base_prompt,
                return_tensors="pt",
                padding=True
            )
            inputs = inputs.to(self._model.device)

            # Japanese accuracy focused generation (heaviness accepted)
            gen_kwargs = {
                "max_length": 1024,
                "num_beams": kwargs.get("beam_size", 5),
                "temperature": kwargs.get("temperature", 0.0),
                "repetition_penalty": kwargs.get("repetition_penalty", 1.12),
                "do_sample": False,
            }

            with torch.no_grad():
                generated_ids = self._model.generate(**inputs, **gen_kwargs)

            text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            result = {"text": text, "model_id": self.model_id, "language": language}

            if kwargs.get("return_timestamps"):
                # Better structured timestamps for real-time applications
                # (In a more advanced implementation we could parse forced alignment or use model outputs)
                result["chunks"] = [
                    {
                        "text": text,
                        "timestamp": (0.0, None)
                    }
                ]

            return result

        # Pipeline fallback (with Japanese quality defaults)
        assert self._pipe is not None
        generate_kwargs = kwargs.get("generate_kwargs", {}) or {}

        language = kwargs.get("language", "ja")
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
        out = {"text": text, "model_id": self.model_id}
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"[VoxtralBackend] Model {self.model_id} unloaded.")
