"""Translate text for the Rust backend command translation boundary.

The Rust app sends a JSON request on stdin:
{"text": "...", "source_language": "ja", "target_language": "en"}

This helper keeps parity with the legacy Python backend by defaulting to
Helsinki-NLP/opus-mt-ja-en for Japanese -> English translation.
"""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache


DEFAULT_MODELS = {
    ("ja", "en"): "Helsinki-NLP/opus-mt-ja-en",
}


def normalize_language(language: str | None) -> str | None:
    if language in (None, "", "auto", "none"):
        return None
    language = str(language).strip().lower()
    return {
        "japanese": "ja",
        "english": "en",
    }.get(language, language)


@lru_cache(maxsize=4)
def load_model(model_id: str):
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "translate_hf.py requires torch and transformers. "
            "Install them in the Python environment used by AMCP_TRANSLATION_COMMAND."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.eval()
    return tokenizer, model, torch


def translate(text: str, source_language: str | None, target_language: str | None) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    source = normalize_language(source_language)
    target = normalize_language(target_language)
    if not source or not target or source == target:
        return text

    model_id = os.environ.get("AMCP_TRANSLATION_MODEL") or DEFAULT_MODELS.get((source, target))
    if not model_id:
        raise RuntimeError(f"Translation pair is not supported: {source}->{target}")

    tokenizer, model, torch = load_model(model_id)
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=int(os.environ.get("AMCP_TRANSLATION_MAX_TOKENS", "256")))
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def main() -> int:
    try:
        request = json.load(sys.stdin)
        translated_text = translate(
            request.get("text", ""),
            request.get("source_language"),
            request.get("target_language"),
        )
        print(json.dumps({"translated_text": translated_text}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
