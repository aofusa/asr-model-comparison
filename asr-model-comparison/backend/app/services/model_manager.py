"""
Core model management service.

Enforces the critical constraint:
    "Only one ASR model may be resident in memory at any time."

All model loading/unloading goes through this class.
Real ASR engine loading is injected via callables so that:
- Unit tests can use cheap mocks
- Production code can supply the actual faster-whisper / transformers loaders
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from app.models.available_models import AVAILABLE_MODELS
from app.models.schemas import ModelInfo

# Type aliases for dependency injection (makes testing trivial)
ModelLoader = Callable[[str], Awaitable[Any]]          # model_id -> loaded engine instance
ModelUnloader = Callable[[Any], Awaitable[None]]       # engine_instance -> None


class ModelManager:
    """
    Manages the lifecycle of exactly one ASR model in memory.
    """

    # Map from our public model_id to the "internal" name the actual library expects.
    # This can grow as we add real loaders.
    MODEL_ID_TO_INTERNAL: dict[str, str] = {
        "whisper-tiny": "tiny",
        "whisper-small": "small",
        "whisper-medium": "medium",
        "whisper-large-v3-turbo": "large-v3-turbo",
        "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
        "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
        "voxtral-mini-4b": "mistralai/Voxtral-Mini-3B-2507",  # example id
    }

    def __init__(
        self,
        model_loader: ModelLoader | None = None,
        model_unloader: ModelUnloader | None = None,
    ) -> None:
        self._current_model_id: str | None = None
        self._current_instance: Any = None

        # Default no-op loaders (safe for early tests)
        self._model_loader = model_loader or self._default_noop_loader
        self._model_unloader = model_unloader or self._default_noop_unloader

        # Build a quick lookup set for validation
        self._valid_model_ids = {m.id for m in AVAILABLE_MODELS}

    @property
    def current_model_id(self) -> str | None:
        return self._current_model_id

    @property
    def is_model_loaded(self) -> bool:
        return self._current_model_id is not None and self._current_instance is not None

    async def load_model(self, model_id: str) -> None:
        """
        Load the requested model.

        If a different model is already loaded, it is unloaded first.
        Loading the same model is a no-op (idempotent).
        """
        if model_id not in self._valid_model_ids:
            raise ValueError(f"Unknown model id: {model_id}. Valid: {self._valid_model_ids}")

        if self._current_model_id == model_id:
            # Already loaded — do nothing (important for performance)
            return

        # Must unload previous model before loading a new one (core constraint)
        if self._current_model_id is not None:
            await self.unload_current()

        # Perform the actual (expensive) load via injected loader
        internal_name = self.MODEL_ID_TO_INTERNAL.get(model_id, model_id)
        self._current_instance = await self._model_loader(internal_name)
        self._current_model_id = model_id

    async def unload_current(self) -> None:
        """Explicitly unload whatever model is currently resident (if any)."""
        if self._current_instance is not None:
            await self._model_unloader(self._current_instance)

        self._current_instance = None
        self._current_model_id = None

    async def get_current_instance(self) -> Any:
        """Return the raw loaded engine (for transcription service to use)."""
        if not self.is_model_loaded:
            raise RuntimeError("No model is currently loaded. Call load_model() first.")
        return self._current_instance

    def get_status(self) -> dict:
        """
        Return lightweight status information about the currently loaded model.

        Used by the /api/status endpoint for UI feedback and debugging on
        memory-constrained devices.
        """
        status: dict = {
            "current_model_id": self._current_model_id,
            "is_model_loaded": self.is_model_loaded,
            "family": None,
        }

        if self._current_model_id:
            # Try to enrich with family info
            for m in AVAILABLE_MODELS:
                if m.id == self._current_model_id:
                    status["family"] = m.family
                    break

        # Memory usage (best effort, using psutil if available)
        try:
            import psutil
            process = psutil.Process()
            mem = process.memory_info()
            status["memory"] = {
                "rss_mb": round(mem.rss / (1024 * 1024), 1),
                "vms_mb": round(mem.vms / (1024 * 1024), 1),
            }
        except Exception:
            status["memory"] = {"note": "psutil not available or failed"}

        return status

    async def transcribe(
        self,
        audio: bytes | str,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        High-level transcription entry point.

        Ensures the requested model (or currently loaded one) is resident,
        then delegates to the loaded engine's transcribe method.

        Returns a dict containing at minimum {"text": str, "processing_time_seconds": float}.
        Real implementations will populate more fields (language, segments, etc.).
        """
        import time

        if model_id and model_id != self.current_model_id:
            await self.load_model(model_id)

        if not self.is_model_loaded:
            # Auto-load first available model as last resort (for early dev)
            first_model = next(iter(self._valid_model_ids))
            await self.load_model(first_model)

        start = time.perf_counter()

        instance = await self.get_current_instance()

        # The actual engine may expose .transcribe(audio, ...) or be a callable
        if hasattr(instance, "transcribe"):
            result = instance.transcribe(audio, **kwargs)
            if hasattr(result, "__await__"):
                result = await result
        else:
            # Fallback for our noop mock
            result = {"text": f"[mock transcription from {self.current_model_id}]"}

        elapsed = time.perf_counter() - start

        if isinstance(result, dict):
            result.setdefault("processing_time_seconds", elapsed)
            result.setdefault("model_id", self.current_model_id)
        else:
            result = {
                "text": str(result),
                "model_id": self.current_model_id,
                "processing_time_seconds": elapsed,
            }

        return result


    # --- Default safe no-op implementations (used when no real loader is provided) ---

    async def _default_noop_loader(self, internal_name: str) -> Any:
        """Placeholder that simulates a loaded model without any ML work."""
        mock = type("NoopModel", (), {"name": internal_name, "transcribe": lambda *a, **k: None})()
        return mock

    async def _default_noop_unloader(self, instance: Any) -> None:
        """Placeholder unloader with best-effort memory cleanup."""
        try:
            del instance
        except Exception:
            pass

        import gc
        gc.collect()
        try:
            import torch
            if hasattr(torch, "cuda"):
                torch.cuda.empty_cache()
        except Exception:
            pass
