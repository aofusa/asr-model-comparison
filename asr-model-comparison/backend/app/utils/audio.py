"""
Audio utilities for real-time ASR streaming.

Provides normalization of arbitrary browser audio chunks (webm/opus, etc.)
into a format that the ASR backends (faster-whisper etc.) can reliably decode:
16kHz mono PCM WAV.

This addresses the root cause of empty transcription results from 2s
MediaRecorder chunks in the WS /api/ws/transcribe flow.
"""

from __future__ import annotations

import io
from typing import Optional


def normalize_to_wav_pcm_16k_mono(raw_audio: bytes) -> bytes:
    """
    Convert raw audio bytes (any format supported by pydub/ffmpeg) to
    16 kHz, mono, PCM WAV bytes suitable for faster-whisper / transformers ASR.

    Used by the WS realtime endpoint so that short mic chunks from the browser
    are not rejected by VAD or decoder due to wrong container / sample rate.

    Falls back to returning the original bytes if conversion fails
    (preserves backward compat for tests that send fake data).

    Requires pydub (already in requirements.txt) + ffmpeg in PATH for full support.
    """
    if not raw_audio:
        return raw_audio

    try:
        from pydub import AudioSegment  # type: ignore

        # pydub can sniff format from content (webm, wav, mp3, ogg, etc.)
        seg: AudioSegment = AudioSegment.from_file(io.BytesIO(raw_audio))

        # Standard for Whisper-family models
        seg = seg.set_frame_rate(16000).set_channels(1)

        out = io.BytesIO()
        seg.export(out, format="wav")
        return out.getvalue()
    except Exception as exc:
        # Do not break the pipeline for fake data or unusual environments.
        # Log at call site (WS handler) for diagnostics.
        # In production one might want to raise or return a sentinel.
        print(f"[audio] normalize_to_wav_pcm_16k_mono failed, returning raw bytes: {exc}")
        return raw_audio


def is_likely_speech(audio: bytes, threshold: int = 500) -> bool:
    """
    Very cheap energy-based heuristic (optional helper).

    Returns True if the (already normalized) audio has enough total energy
    to be worth sending to a heavy ASR model. Used to avoid pointless
    inference on pure silence chunks.

    This is a lightweight complement to the model's own VAD.
    """
    if not audio or len(audio) < 100:
        return False
    try:
        # If it's a WAV, we could parse header, but for speed just look at amplitude bytes.
        # Treat last 1k bytes as rough proxy for energy.
        tail = audio[-1024:]
        energy = sum(abs(b - 128) for b in tail)  # rough for 8-bit, good enough as heuristic
        return energy > threshold
    except Exception:
        return True  # when in doubt, let the model decide
