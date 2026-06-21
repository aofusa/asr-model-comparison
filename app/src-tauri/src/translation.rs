use serde::{Deserialize, Serialize};
use std::io::Write;
use std::process::{Command, Stdio};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TranslationOutcome {
    pub transcript_text: String,
    pub translated_text: Option<String>,
    pub target_language: Option<String>,
    pub engine: &'static str,
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct TranslationCommandRequest<'a> {
    text: &'a str,
    source_language: Option<&'a str>,
    target_language: &'a str,
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
            engine: "none",
            note: None,
        };
    }

    if target.is_none() || source == target {
        return TranslationOutcome {
            transcript_text,
            translated_text: None,
            target_language: target,
            engine: "none",
            note: None,
        };
    }

    let normalized =
        normalize_text_for_translation(&transcript_text, source.as_deref(), target.as_deref());

    match translate_with_command(&normalized, source.as_deref(), target.as_deref()) {
        Ok(Some(translated_text)) => TranslationOutcome {
            transcript_text: normalized,
            translated_text: Some(translated_text),
            target_language: target,
            engine: "command",
            note: None,
        },
        Ok(None) => TranslationOutcome {
            transcript_text: normalized,
            translated_text: None,
            target_language: target,
            engine: "unavailable",
            note: Some("translation engine is not configured in the Rust backend".to_string()),
        },
        Err(error) => TranslationOutcome {
            transcript_text: normalized,
            translated_text: None,
            target_language: target,
            engine: "error",
            note: Some(error),
        },
    }
}

fn translate_with_command(
    text: &str,
    source_language: Option<&str>,
    target_language: Option<&str>,
) -> Result<Option<String>, String> {
    let Some(target_language) = target_language else {
        return Ok(None);
    };
    let source_language = source_language.unwrap_or("auto");
    let command_key = format!(
        "AMCP_TRANSLATION_{}_{}_COMMAND",
        source_language.to_ascii_uppercase(),
        target_language.to_ascii_uppercase()
    );
    let command = std::env::var(&command_key)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| std::env::var("AMCP_TRANSLATION_COMMAND").ok())
        .filter(|value| !value.trim().is_empty());
    let Some(command) = command else {
        return Ok(None);
    };
    let args = std::env::var(format!(
        "AMCP_TRANSLATION_{}_{}_ARGS",
        source_language.to_ascii_uppercase(),
        target_language.to_ascii_uppercase()
    ))
    .ok()
    .or_else(|| std::env::var("AMCP_TRANSLATION_ARGS").ok())
    .map(|args| split_command_args(&args))
    .unwrap_or_default();

    let normalized_source = normalize_language_code(Some(source_language));
    let request = TranslationCommandRequest {
        text,
        source_language: normalized_source.as_deref(),
        target_language,
    };
    let request_json = serde_json::to_vec(&request)
        .map_err(|error| format!("failed to serialize translation request: {error}"))?;
    let mut child = Command::new(&command)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("failed to start translation command `{command}`: {error}"))?;

    if let Some(stdin) = child.stdin.as_mut() {
        stdin
            .write_all(&request_json)
            .map_err(|error| format!("failed to write translation request: {error}"))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|error| format!("failed to wait for translation command: {error}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(if stderr.is_empty() {
            format!("translation command exited with {}", output.status)
        } else {
            format!(
                "translation command exited with {}: {stderr}",
                output.status
            )
        });
    }

    let stdout = String::from_utf8(output.stdout)
        .map_err(|error| format!("translation command output is not UTF-8: {error}"))?;
    parse_translation_output(&stdout)
        .map(Some)
        .map_err(|error| format!("invalid translation command output: {error}"))
}

fn parse_translation_output(output: &str) -> Result<String, String> {
    let output = output.trim();
    if output.is_empty() {
        return Ok(String::new());
    }
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(output) {
        if let Some(text) = value
            .get("translated_text")
            .or_else(|| value.get("translation"))
            .or_else(|| value.get("text"))
            .and_then(|value| value.as_str())
        {
            return Ok(text.trim().to_string());
        }
        return Err("JSON output must contain translated_text, translation, or text".to_string());
    }
    Ok(output.to_string())
}

fn split_command_args(args: &str) -> Vec<String> {
    args.split_whitespace()
        .filter(|arg| !arg.trim().is_empty())
        .map(ToString::to_string)
        .collect()
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
        assert_eq!(outcome.engine, "unavailable");
    }

    #[test]
    fn parses_json_translation_command_output() {
        assert_eq!(
            parse_translation_output(r#"{"translated_text":"It is sunny."}"#).unwrap(),
            "It is sunny."
        );
    }

    #[test]
    fn parses_plain_translation_command_output() {
        assert_eq!(
            parse_translation_output("It is sunny.\n").unwrap(),
            "It is sunny."
        );
    }

    #[test]
    fn same_language_does_not_translate() {
        let outcome = translate_optional("hello", Some("en"), Some("en"));

        assert_eq!(outcome.transcript_text, "hello");
        assert_eq!(outcome.translated_text, None);
        assert_eq!(outcome.engine, "none");
    }
}
