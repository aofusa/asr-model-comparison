"""
Optional text translation helper for ASR results.

Qwen3-ASR is an ASR model, not a speech translation model. When the UI requests
translation for Qwen output, use a dedicated translation model instead of
Gemma/Bonsai-style post-processing.
"""
from __future__ import annotations

from functools import lru_cache
import re


LANGUAGE_ALIASES = {
    "ja": "ja",
    "japanese": "ja",
    "en": "en",
    "english": "en",
}

TRANSLATION_MODELS = {
    ("ja", "en"): "Helsinki-NLP/opus-mt-ja-en",
}

JAPANESE_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def _normalize_language_code(language: str | None) -> str | None:
    if language in (None, "", "auto", "none"):
        return None
    return LANGUAGE_ALIASES.get(str(language).strip().lower())


def _parse_small_japanese_number(value: str) -> int | None:
    if not value:
        return None
    if value in JAPANESE_DIGITS:
        return JAPANESE_DIGITS[value]
    if "十" not in value:
        return None

    tens_text, _, ones_text = value.partition("十")
    tens = 1 if not tens_text else JAPANESE_DIGITS.get(tens_text)
    ones = 0 if not ones_text else JAPANESE_DIGITS.get(ones_text)
    if tens is None or ones is None:
        return None
    return tens * 10 + ones


def normalize_text_for_translation(text: str, source_language: str | None, target_language: str | None) -> str:
    source = _normalize_language_code(source_language)
    target = _normalize_language_code(target_language)
    if source != "ja" or target != "en":
        return text

    def replace_number(match: re.Match[str]) -> str:
        parsed = _parse_small_japanese_number(match.group(0))
        return str(parsed) if parsed is not None else match.group(0)

    return re.sub(r"[零一二三四五六七八九十]{1,3}", replace_number, text)


@lru_cache(maxsize=4)
def _load_translation_model(model_id: str):
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Translation requires transformers and tokenizer dependencies. "
            "Install backend requirements, then restart the server."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.eval()
    return tokenizer, model


def translate_text(text: str, source_language: str | None, target_language: str | None) -> str:
    text = (text or "").strip()
    if not text:
        return text

    source = _normalize_language_code(source_language)
    target = _normalize_language_code(target_language)
    if not source or not target or source == target:
        return text

    model_id = TRANSLATION_MODELS.get((source, target))
    if model_id is None:
        raise RuntimeError(f"Translation pair is not supported yet: {source}->{target}")

    normalized_text = normalize_text_for_translation(text, source, target)
    tokenizer, model = _load_translation_model(model_id)
    inputs = tokenizer(normalized_text, return_tensors="pt", truncation=True)

    try:
        import torch

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256)
    except ImportError:
        outputs = model.generate(**inputs, max_new_tokens=256)

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
