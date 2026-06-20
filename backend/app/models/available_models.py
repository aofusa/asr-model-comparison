"""
Canonical list of supported ASR models for the platform.

This file exists to avoid circular imports between:
- routers (API layer)
- services/model_manager (core logic)

It should only contain static data + the ModelInfo schema.
"""
from __future__ import annotations

from app.models.schemas import ModelInfo

AVAILABLE_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="whisper-tiny",
        name="Whisper Tiny",
        family="whisper",
        size="tiny",
        parameters="39M",
        description="最速の軽量モデル。精度は低めだがリソース消費が非常に少ない。",
        recommended_vram_gb=2.0,
    ),
    ModelInfo(
        id="whisper-small",
        name="Whisper Small",
        family="whisper",
        size="small",
        parameters="244M",
        description="バランスの取れた軽量モデル。",
        recommended_vram_gb=4.0,
    ),
    ModelInfo(
        id="whisper-medium",
        name="Whisper Medium",
        family="whisper",
        size="medium",
        parameters="769M",
        description="精度と速度のバランスが良い標準モデル。",
        recommended_vram_gb=7.0,
    ),
    ModelInfo(
        id="whisper-large-v3-turbo",
        name="Whisper Large-v3 Turbo",
        family="whisper",
        size="large-v3-turbo",
        parameters="~1.6B",
        description="高速化された高精度モデル。",
        recommended_vram_gb=9.0,
    ),
    ModelInfo(
        id="qwen3-asr-0.6b",
        name="Qwen3-ASR 0.6B",
        family="qwen3",
        size="0.6B",
        parameters="0.6B",
        description="Qwen3シリーズの軽量高性能モデル。日本語に強い。",
        recommended_vram_gb=5.0,
    ),
    ModelInfo(
        id="qwen3-asr-1.7b",
        name="Qwen3-ASR 1.7B",
        family="qwen3",
        size="1.7B",
        parameters="1.7B",
        description="Qwen3シリーズのバランスモデル。精度と速度の優秀なトレードオフ。",
        recommended_vram_gb=10.0,
    ),
    ModelInfo(
        id="voxtral-mini-4b",
        name="Voxtral Mini 4B Realtime",
        family="voxtral",
        size="Mini 4B",
        parameters="4B",
        description="Mistral製の高精度Realtime向けモデル。",
        recommended_vram_gb=16.0,
    ),
]
