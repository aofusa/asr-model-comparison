from __future__ import annotations

from app.services.translation import normalize_text_for_translation
from app.api.routers.transcribe import _normalize_transcription_result


def test_normalize_text_for_translation_converts_small_japanese_numbers_for_ja_en():
    text = "最高気温は二十三度です。夕方から少し雲が増えるでしょう。"

    assert (
        normalize_text_for_translation(text, "ja", "en")
        == "最高気温は23度です。夕方から少し雲が増えるでしょう。"
    )


def test_normalize_text_for_translation_keeps_non_translation_text_unchanged():
    text = "最高気温は二十三度です。"

    assert normalize_text_for_translation(text, "ja", "ja") == text


def test_normalize_transcription_result_preserves_original_and_translation():
    result = _normalize_transcription_result(
        {
            "text": "Translated text",
            "transcript_text": "原文です",
            "translated_text": "Translated text",
            "language": "ja",
            "target_language": "en",
            "processing_time_seconds": 0.2,
        }
    )

    assert result["text"] == "Translated text"
    assert result["transcript_text"] == "原文です"
    assert result["translated_text"] == "Translated text"
    assert result["target_language"] == "en"


def test_normalize_transcription_result_without_translation_uses_text_as_transcript():
    result = _normalize_transcription_result(
        {
            "text": "認識結果",
            "translated_text": None,
            "processing_time_seconds": 0.1,
        }
    )

    assert result["text"] == "認識結果"
    assert result["transcript_text"] == "認識結果"
    assert result["translated_text"] is None
