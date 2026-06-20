from __future__ import annotations

from app.services.translation import normalize_text_for_translation


def test_normalize_text_for_translation_converts_small_japanese_numbers_for_ja_en():
    text = "最高気温は二十三度です。夕方から少し雲が増えるでしょう。"

    assert (
        normalize_text_for_translation(text, "ja", "en")
        == "最高気温は23度です。夕方から少し雲が増えるでしょう。"
    )


def test_normalize_text_for_translation_keeps_non_translation_text_unchanged():
    text = "最高気温は二十三度です。"

    assert normalize_text_for_translation(text, "ja", "ja") == text
