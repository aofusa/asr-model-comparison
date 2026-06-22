use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TranslationOutcome {
    pub transcript_text: String,
    pub translated_text: Option<String>,
    pub target_language: Option<String>,
    pub engine: String,
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TranslationRuntimeStatus {
    pub configured: bool,
    pub engine: &'static str,
    pub supported_pairs: Vec<String>,
    pub reason: String,
}

pub fn translate_optional(
    text: &str,
    source_language: Option<&str>,
    target_language: Option<&str>,
) -> TranslationOutcome {
    let transcript_text = text.trim().to_string();
    let source = normalize_language_code(source_language);
    let target = normalize_language_code(target_language);

    if transcript_text.is_empty() {
        return TranslationOutcome {
            transcript_text,
            translated_text: None,
            target_language: target,
            engine: "none".to_string(),
            note: None,
        };
    }

    if target.is_none() || source == target {
        return TranslationOutcome {
            transcript_text,
            translated_text: None,
            target_language: target,
            engine: "none".to_string(),
            note: None,
        };
    }

    let normalized =
        normalize_text_for_translation(&transcript_text, source.as_deref(), target.as_deref());

    TranslationOutcome {
        transcript_text: normalized,
        translated_text: None,
        target_language: target,
        engine: "rust-native-unavailable".to_string(),
        note: Some(
            "Rust-native text translation is unavailable; use Qwen3-ASR or Voxtral for model-native speech translation."
                .to_string(),
        ),
    }
}

impl TranslationOutcome {
    pub fn from_backend_translation(
        transcript_text: String,
        translated_text: String,
        target_language: Option<String>,
        engine: impl Into<String>,
    ) -> Self {
        TranslationOutcome {
            transcript_text,
            translated_text: Some(translated_text),
            target_language,
            engine: engine.into(),
            note: None,
        }
    }
}

pub fn runtime_status() -> TranslationRuntimeStatus {
    TranslationRuntimeStatus {
        configured: true,
        engine: "model-native",
        supported_pairs: vec![
            "qwen3-asr:speech->target-language".to_string(),
            "voxtral:speech->target-language".to_string(),
        ],
        reason: "external translation commands are disabled; Qwen3-ASR uses Rust/Candle language-prompt translation and Voxtral uses patched llama.cpp model-native text generation when target_language is set".to_string(),
    }
}

pub fn normalize_language_code(language: Option<&str>) -> Option<String> {
    let language = language?.trim().to_ascii_lowercase();
    match language.as_str() {
        "" | "auto" | "none" => None,
        "ja" | "japanese" => Some("ja".to_string()),
        "en" | "english" => Some("en".to_string()),
        other => Some(other.to_string()),
    }
}

pub fn normalize_text_for_translation(
    text: &str,
    source_language: Option<&str>,
    target_language: Option<&str>,
) -> String {
    if normalize_language_code(source_language) != Some("ja".to_string())
        || normalize_language_code(target_language) != Some("en".to_string())
    {
        return text.to_string();
    }

    replace_japanese_numbers(text)
}

fn replace_japanese_numbers(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut buffer = String::new();

    for ch in text.chars() {
        if is_japanese_number_char(ch) {
            buffer.push(ch);
            continue;
        }

        flush_number_buffer(&mut output, &mut buffer);
        output.push(ch);
    }
    flush_number_buffer(&mut output, &mut buffer);

    output
}

fn flush_number_buffer(output: &mut String, buffer: &mut String) {
    if buffer.is_empty() {
        return;
    }

    if buffer.chars().count() <= 3 {
        if let Some(number) = parse_small_japanese_number(buffer) {
            output.push_str(&number.to_string());
            buffer.clear();
            return;
        }
    }

    output.push_str(buffer);
    buffer.clear();
}

fn is_japanese_number_char(ch: char) -> bool {
    matches!(
        ch,
        '零' | '一' | '二' | '三' | '四' | '五' | '六' | '七' | '八' | '九' | '十'
    )
}

fn parse_small_japanese_number(value: &str) -> Option<u32> {
    if value.is_empty() {
        return None;
    }
    if let Some(digit) = japanese_digit(value) {
        return Some(digit);
    }
    let (tens_text, ones_text) = value.split_once('十').unwrap_or((value, ""));
    if !value.contains('十') {
        return None;
    }

    let tens = if tens_text.is_empty() {
        1
    } else {
        japanese_digit(tens_text)?
    };
    let ones = if ones_text.is_empty() {
        0
    } else {
        japanese_digit(ones_text)?
    };

    Some(tens * 10 + ones)
}

fn japanese_digit(value: &str) -> Option<u32> {
    match value {
        "零" => Some(0),
        "一" => Some(1),
        "二" => Some(2),
        "三" => Some(3),
        "四" => Some(4),
        "五" => Some(5),
        "六" => Some(6),
        "七" => Some(7),
        "八" => Some(8),
        "九" => Some(9),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_small_japanese_numbers_for_ja_to_en() {
        assert_eq!(
            normalize_text_for_translation("今日は二十三人です", Some("ja"), Some("en")),
            "今日は23人です"
        );
    }

    #[test]
    fn unsupported_translation_preserves_contract_without_fake_translation() {
        let outcome = translate_optional("今日は二人です", Some("ja"), Some("en"));

        assert_eq!(outcome.transcript_text, "今日は2人です");
        assert_eq!(outcome.translated_text, None);
        assert_eq!(outcome.target_language.as_deref(), Some("en"));
        assert_eq!(outcome.engine, "rust-native-unavailable");
    }

    #[test]
    fn same_language_does_not_translate() {
        let outcome = translate_optional("hello", Some("en"), Some("en"));

        assert_eq!(outcome.transcript_text, "hello");
        assert_eq!(outcome.translated_text, None);
        assert_eq!(outcome.engine, "none");
    }
}
