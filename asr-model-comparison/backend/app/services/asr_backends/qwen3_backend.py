"""
Qwen3-ASR Real Backend Implementation

Uses transformers pipeline for automatic speech recognition.
This provides actual model usage for Qwen3-ASR 0.6B and 1.7B variants.

Note: First run will download the model (several GB). 
Recommended on machines with at least 16-24GB RAM.
"""
from __future__ import annotations

import tempfile
import os
from typing import Any

from app.services.asr_backends.base import ASRBackend


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
            print(f"[Qwen3ASRBackend] Warning: 4-bit quantization requested on {device}. Falling back.")
            self._quantization = "none"

        self._torch_dtype = torch_dtype

        self._pipe = None
        self._hf_model_id = self._resolve_hf_model_id(model_id)

    def _resolve_hf_model_id(self, model_id: str) -> str:
        mapping = {
            "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
            "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
        }
        return mapping.get(model_id, model_id)

    def _ensure_loaded(self) -> None:
        if self._pipe is not None or getattr(self, "_model", None) is not None:
            return

        try:
            print(f"[Qwen3ASRBackend] Loading real model: {self.model_id} ({self._hf_model_id}) on {self._device}...")
        except Exception:
            pass

        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-ASR requires optional heavy dependencies: torch and transformers. "
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
                from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

                print(f"[Qwen3ASRBackend] Attempting dedicated Qwen2Audio class...")
                self._processor = AutoProcessor.from_pretrained(self._hf_model_id)
                self._model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    self._hf_model_id,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    quantization_config=quantization_config,
                    low_cpu_mem_usage=True,
                )
                print(f"[Qwen3ASRBackend] Successfully loaded using dedicated class.")
                return
            except Exception as e:
                print(f"[Qwen3ASRBackend] Dedicated class loading failed ({type(e).__name__}: {e}). Falling back to pipeline for stability...")

        # Stable pipeline fallback
        device_arg = 0 if self._device == "cuda" else -1
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self._hf_model_id,
            device=device_arg,
            torch_dtype=torch_dtype,
            model_kwargs={"low_cpu_mem_usage": True},
        )
        print(f"[Qwen3ASRBackend] Model {self.model_id} loaded via pipeline (fallback).")

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
        # Dedicated class path (higher quality for real-time Japanese use)
        if getattr(self, "_model", None) is not None and getattr(self, "_processor", None) is not None:
            language = kwargs.get("language", "ja")
            previous_text = kwargs.get("previous_text", "")

            # Improved prompting for Japanese ASR quality with real-time chunk context
            user_text = "Transcribe the following audio into Japanese accurately and naturally."
            if previous_text:
                user_text += f"\nPrevious transcript context (for continuity): {previous_text}"

            conversation = [
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": audio_path},
                    {"type": "text", "text": user_text},
                ]}
            ]

            text_prompt = self._processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )

            inputs = self._processor(text=text_prompt, audio=audio_path, return_tensors="pt", padding=True)
            inputs = inputs.to(self._model.device)

            # Japanese accuracy focused generation parameters (heaviness accepted)
            # Recommended for real-time Japanese: low temperature + decent beams
            gen_kwargs = {
                "max_length": 1024,
                "num_beams": kwargs.get("beam_size", 6),
                "temperature": kwargs.get("temperature", 0.0),
                "repetition_penalty": kwargs.get("repetition_penalty", 1.15),
                "do_sample": False,
            }

            import torch
            with torch.no_grad():
                generated_ids = self._model.generate(**inputs, **gen_kwargs)

            text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            result = {"text": text, "model_id": self.model_id, "language": language}

            if kwargs.get("return_timestamps"):
                # Structured timestamps (improved in dedicated path)
                result["chunks"] = [{"text": text, "timestamp": (0.0, None)}]

            return result

        # Pipeline fallback (also apply Japanese quality settings)
        assert self._pipe is not None
        generate_kwargs = kwargs.get("generate_kwargs", {}) or {}

        language = kwargs.get("language", "ja")
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
            "model_id": self.model_id,
            "language": result.get("language") if isinstance(result, dict) else None,
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

        print(f"[Qwen3ASRBackend] Model {self.model_id} unloaded.")
