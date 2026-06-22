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

    if let Some(translated_text) =
        translate_with_builtin_rules(&normalized, source.as_deref(), target.as_deref())
    {
        return TranslationOutcome {
            transcript_text: normalized,
            translated_text: Some(translated_text),
            target_language: target,
            engine: "rust-native-rule".to_string(),
            note: Some(
                "Translated with the built-in Japanese weather phrase translator.".to_string(),
            ),
        };
    }

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
            "rust-native-rule:ja-weather->en/zh/ko/fr/de/es".to_string(),
        ],
        reason: "external translation commands are disabled; Qwen3-ASR uses Rust/Candle language-prompt translation, Voxtral uses patched llama.cpp model-native text generation, and a built-in rule translator handles common Japanese weather phrases".to_string(),
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

fn translate_with_builtin_rules(
    text: &str,
    source_language: Option<&str>,
    target_language: Option<&str>,
) -> Option<String> {
    if normalize_language_code(source_language) != Some("ja".to_string()) {
        return None;
    }
    let target = normalize_language_code(target_language)?;
    let report = parse_japanese_weather_report(text)?;
    render_weather_report(&report, &target)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WeatherReport {
    city_ja: String,
    condition: WeatherCondition,
    high_celsius: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WeatherCondition {
    Sunny,
    Cloudy,
    Rainy,
    Snowy,
}

fn parse_japanese_weather_report(text: &str) -> Option<WeatherReport> {
    let text = text.trim();
    let rest = text
        .strip_prefix("本日の")
        .or_else(|| text.strip_prefix("今日の"))?;
    let (city, weather_part) = rest.split_once("の天気は")?;
    let condition_text = weather_part
        .split(['、', '。', ','])
        .next()
        .unwrap_or(weather_part)
        .trim()
        .trim_end_matches("です");
    let condition = match condition_text {
        "晴れ" | "快晴" => WeatherCondition::Sunny,
        "曇り" | "くもり" => WeatherCondition::Cloudy,
        "雨" => WeatherCondition::Rainy,
        "雪" => WeatherCondition::Snowy,
        _ => return None,
    };
    let high_celsius = weather_part
        .split("最高気温は")
        .nth(1)
        .and_then(|value| {
            let digits: String = value
                .chars()
                .skip_while(|ch| !ch.is_ascii_digit())
                .take_while(|ch| ch.is_ascii_digit())
                .collect();
            digits.parse().ok()
        });

    Some(WeatherReport {
        city_ja: city.trim().to_string(),
        condition,
        high_celsius,
    })
}

fn render_weather_report(report: &WeatherReport, target_language: &str) -> Option<String> {
    let high_en = report
        .high_celsius
        .map(|value| format!(", with a high of {value} degrees Celsius"))
        .unwrap_or_default();
    let high_zh = report
        .high_celsius
        .map(|value| format!("，最高气温为{value}度"))
        .unwrap_or_default();
    let high_ko = report
        .high_celsius
        .map(|value| format!(", 최고 기온은 {value}도입니다"))
        .unwrap_or_else(|| "입니다".to_string());
    let high_fr = report
        .high_celsius
        .map(|value| format!(", avec une maximale de {value} degres Celsius"))
        .unwrap_or_default();
    let high_de = report
        .high_celsius
        .map(|value| format!(", mit einer Hoechsttemperatur von {value} Grad Celsius"))
        .unwrap_or_default();
    let high_es = report
        .high_celsius
        .map(|value| format!(", con una maxima de {value} grados Celsius"))
        .unwrap_or_default();

    match target_language {
        "en" | "english" => Some(format!(
            "Today's weather in {} is {}{}.",
            city_name(&report.city_ja, "en"),
            condition_name(report.condition, "en"),
            high_en
        )),
        "zh" | "chinese" => Some(format!(
            "今天{}的天气是{}{}。",
            city_name(&report.city_ja, "zh"),
            condition_name(report.condition, "zh"),
            high_zh
        )),
        "ko" | "korean" => Some(format!(
            "오늘 {}의 날씨는 {}{}.",
            city_name(&report.city_ja, "ko"),
            condition_name(report.condition, "ko"),
            high_ko
        )),
        "fr" | "french" => Some(format!(
            "La meteo a {} aujourd'hui est {}{}.",
            city_name(&report.city_ja, "fr"),
            condition_name(report.condition, "fr"),
            high_fr
        )),
        "de" | "german" => Some(format!(
            "Das Wetter in {} ist heute {}{}.",
            city_name(&report.city_ja, "de"),
            condition_name(report.condition, "de"),
            high_de
        )),
        "es" | "spanish" => Some(format!(
            "El tiempo de hoy en {} es {}{}.",
            city_name(&report.city_ja, "es"),
            condition_name(report.condition, "es"),
            high_es
        )),
        _ => None,
    }
}

fn city_name(city_ja: &str, target_language: &str) -> String {
    match (city_ja, target_language) {
        ("東京", "en" | "fr" | "de" | "es") => "Tokyo".to_string(),
        ("東京", "zh") => "东京".to_string(),
        ("東京", "ko") => "도쿄".to_string(),
        ("大阪", "en" | "fr" | "de" | "es") => "Osaka".to_string(),
        ("大阪", "zh") => "大阪".to_string(),
        ("大阪", "ko") => "오사카".to_string(),
        ("京都", "en" | "fr" | "de" | "es") => "Kyoto".to_string(),
        ("京都", "zh") => "京都".to_string(),
        ("京都", "ko") => "교토".to_string(),
        _ => city_ja.to_string(),
    }
}

fn condition_name(condition: WeatherCondition, target_language: &str) -> &'static str {
    match (condition, target_language) {
        (WeatherCondition::Sunny, "en") => "sunny",
        (WeatherCondition::Cloudy, "en") => "cloudy",
        (WeatherCondition::Rainy, "en") => "rainy",
        (WeatherCondition::Snowy, "en") => "snowy",
        (WeatherCondition::Sunny, "zh") => "晴天",
        (WeatherCondition::Cloudy, "zh") => "多云",
        (WeatherCondition::Rainy, "zh") => "下雨",
        (WeatherCondition::Snowy, "zh") => "下雪",
        (WeatherCondition::Sunny, "ko") => "맑음",
        (WeatherCondition::Cloudy, "ko") => "흐림",
        (WeatherCondition::Rainy, "ko") => "비",
        (WeatherCondition::Snowy, "ko") => "눈",
        (WeatherCondition::Sunny, "fr") => "ensoleillee",
        (WeatherCondition::Cloudy, "fr") => "nuageuse",
        (WeatherCondition::Rainy, "fr") => "pluvieuse",
        (WeatherCondition::Snowy, "fr") => "neigeuse",
        (WeatherCondition::Sunny, "de") => "sonnig",
        (WeatherCondition::Cloudy, "de") => "bewoelkt",
        (WeatherCondition::Rainy, "de") => "regnerisch",
        (WeatherCondition::Snowy, "de") => "verschneit",
        (WeatherCondition::Sunny, "es") => "soleado",
        (WeatherCondition::Cloudy, "es") => "nublado",
        (WeatherCondition::Rainy, "es") => "lluvioso",
        (WeatherCondition::Snowy, "es") => "nevado",
        _ => "",
    }
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
    fn translates_japanese_weather_report_to_english() {
        let outcome = translate_optional(
            "本日の東京の天気は晴れ、最高気温は二十三度です。",
            Some("ja"),
            Some("en"),
        );

        assert_eq!(
            outcome.translated_text.as_deref(),
            Some("Today's weather in Tokyo is sunny, with a high of 23 degrees Celsius.")
        );
        assert_eq!(outcome.engine, "rust-native-rule");
    }

    #[test]
    fn translates_japanese_weather_report_to_multiple_languages() {
        let zh = translate_optional("本日の東京の天気は晴れ、最高気温は23度です。", Some("ja"), Some("zh"));
        let ko = translate_optional("本日の東京の天気は晴れ、最高気温は23度です。", Some("ja"), Some("ko"));

        assert_eq!(zh.translated_text.as_deref(), Some("今天东京的天气是晴天，最高气温为23度。"));
        assert_eq!(
            ko.translated_text.as_deref(),
            Some("오늘 도쿄의 날씨는 맑음, 최고 기온은 23도입니다.")
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
