"""
Pluggable ASR backend adapters.

Each real model family (Whisper via faster-whisper, Qwen3, Voxtral)
should have its own adapter that implements a common interface.

This allows the ModelManager to stay independent of any specific ML library.
"""
from __future__ import annotations
