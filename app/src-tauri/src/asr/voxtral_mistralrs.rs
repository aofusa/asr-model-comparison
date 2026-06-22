use super::audio::PreprocessedAudio;
use crate::accelerator::HardwareBackend;
use serde::{Deserialize, Serialize};
#[cfg(feature = "voxtral")]
use std::path::{Path, PathBuf};

const DEFAULT_MISTRALRS_MODEL_ID: &str = "mistralai/Voxtral-Mini-4B-Realtime-2602";
const DEFAULT_MISTRALRS_API_MODEL: &str = "default";
const DEFAULT_MISTRALRS_TIMEOUT_SECONDS: u64 = 300;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VoxtralMistralRsConfig {
    pub url: Option<String>,
    pub model_id: String,
    pub api_model: String,
    pub timeout_seconds: u64,
    pub providers: Vec<HardwareBackend>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VoxtralMistralRsTranscription {
    pub text: String,
    pub transcript_text: String,
    pub translated_text: Option<String>,
    pub target_language: Option<String>,
    pub model_id: String,
    pub url: Option<String>,
}

pub fn configure_voxtral_mistralrs(
    available_backends: &[HardwareBackend],
) -> VoxtralMistralRsConfig {
    VoxtralMistralRsConfig {
        url: env_string("AMCP_VOXTRAL_MISTRALRS_URL"),
        model_id: env_string("AMCP_VOXTRAL_MISTRALRS_MODEL_ID")
            .unwrap_or_else(|| DEFAULT_MISTRALRS_MODEL_ID.to_string()),
        api_model: env_string("AMCP_VOXTRAL_MISTRALRS_API_MODEL")
            .unwrap_or_else(|| DEFAULT_MISTRALRS_API_MODEL.to_string()),
        timeout_seconds: std::env::var("AMCP_VOXTRAL_MISTRALRS_TIMEOUT_SECONDS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(DEFAULT_MISTRALRS_TIMEOUT_SECONDS)
            .clamp(1, 3600),
        providers: mistralrs_provider_plan(available_backends),
    }
}

pub fn voxtral_runtime_preference() -> Option<String> {
    env_string("AMCP_VOXTRAL_RUNTIME").map(|runtime| runtime.to_ascii_lowercase())
}

pub fn is_mistralrs_requested() -> bool {
    voxtral_runtime_preference().as_deref() == Some("mistralrs")
}

pub fn should_route_to_mistralrs(language: Option<&str>, target_language: Option<&str>) -> bool {
    should_route_to_mistralrs_with_runtime(
        language,
        target_language,
        voxtral_runtime_preference().as_deref(),
    )
}

pub fn should_route_to_mistralrs_with_runtime(
    language: Option<&str>,
    target_language: Option<&str>,
    runtime: Option<&str>,
) -> bool {
    if runtime.is_some_and(|runtime| runtime.eq_ignore_ascii_case("mistralrs")) {
        return true;
    }
    is_japanese_language(language) || is_japanese_language(target_language)
}

pub fn mistralrs_provider_plan(available_backends: &[HardwareBackend]) -> Vec<HardwareBackend> {
    let mut providers = Vec::new();
    for backend in [
        HardwareBackend::Cuda,
        HardwareBackend::Vulkan,
        HardwareBackend::Metal,
        HardwareBackend::Cpu,
    ] {
        if available_backends.contains(&backend) || backend == HardwareBackend::Cpu {
            providers.push(backend);
        }
    }
    providers
}

#[cfg(feature = "voxtral")]
pub fn transcribe_voxtral_audio(
    audio: &PreprocessedAudio,
    language: Option<&str>,
    target_language: Option<&str>,
    previous_text: Option<&str>,
    available_backends: &[HardwareBackend],
) -> Result<Option<VoxtralMistralRsTranscription>, String> {
    let config = configure_voxtral_mistralrs(available_backends);
    let Some(url) = config.url.as_deref() else {
        return Err(
            "Voxtral mistral.rs runtime is selected, but AMCP_VOXTRAL_MISTRALRS_URL is not configured. Start mistralrs serve and set the local server URL."
                .to_string(),
        );
    };

    let audio_path = write_temp_wav(audio)?;
    let transcript = (|| {
        let transcript_prompt = transcription_prompt(language, previous_text);
        let transcript_text = run_mistralrs_chat(&config, url, &audio_path, &transcript_prompt)?;
        let target_language = normalize_target_language(target_language);
        let translated_text = if let Some(target_language) = target_language.as_deref() {
            let instruction =
                translation_prompt(language, target_language, &transcript_text, previous_text);
            let translated = run_mistralrs_chat(&config, url, &audio_path, &instruction)?;
            if translated.is_empty() {
                None
            } else {
                Some(translated)
            }
        } else {
            None
        };

        Ok(VoxtralMistralRsTranscription {
            text: translated_text
                .clone()
                .unwrap_or_else(|| transcript_text.clone()),
            transcript_text,
            translated_text,
            target_language,
            model_id: config.model_id.clone(),
            url: config.url.clone(),
        })
    })();
    let _ = std::fs::remove_file(&audio_path);
    transcript.map(Some)
}

#[cfg(not(feature = "voxtral"))]
pub fn transcribe_voxtral_audio(
    _audio: &PreprocessedAudio,
    _language: Option<&str>,
    _target_language: Option<&str>,
    _previous_text: Option<&str>,
    _available_backends: &[HardwareBackend],
) -> Result<Option<VoxtralMistralRsTranscription>, String> {
    Ok(None)
}

#[cfg(feature = "voxtral")]
fn run_mistralrs_chat(
    config: &VoxtralMistralRsConfig,
    url: &str,
    audio_path: &Path,
    prompt: &str,
) -> Result<String, String> {
    let endpoint = chat_completions_url(url);
    let audio_url = file_url(audio_path);
    let request = serde_json::json!({
        "model": config.api_model,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": { "url": audio_url }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }],
        "temperature": 0.0,
        "max_tokens": max_tokens(),
    });

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(config.timeout_seconds))
        .build()
        .map_err(|error| format!("failed to create mistral.rs HTTP client: {error}"))?;
    let response = client
        .post(endpoint)
        .json(&request)
        .send()
        .map_err(|error| format!("mistral.rs request failed: {error}"))?;
    let status = response.status();
    let body = response
        .text()
        .map_err(|error| format!("failed to read mistral.rs response: {error}"))?;
    if !status.is_success() {
        return Err(format!("mistral.rs returned HTTP {status}: {body}"));
    }

    parse_chat_completion_content(&body)
}

#[cfg(any(feature = "voxtral", test))]
fn parse_chat_completion_content(body: &str) -> Result<String, String> {
    let value = serde_json::from_str::<serde_json::Value>(body).map_err(|error| {
        format!("failed to parse mistral.rs JSON response: {error}; body={body}")
    })?;
    let content = value
        .get("choices")
        .and_then(|choices| choices.as_array())
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .ok_or_else(|| format!("mistral.rs response has no choices[0].message.content: {body}"))?;
    if let Some(text) = content.as_str() {
        return Ok(text.trim().to_string());
    }
    if let Some(parts) = content.as_array() {
        let text = parts
            .iter()
            .filter_map(|part| {
                part.get("text")
                    .and_then(|text| text.as_str())
                    .or_else(|| part.as_str())
            })
            .collect::<Vec<_>>()
            .join("");
        return Ok(text.trim().to_string());
    }
    Err(format!(
        "mistral.rs response content is not a string or text array: {content}"
    ))
}

#[cfg(feature = "voxtral")]
fn write_temp_wav(audio: &PreprocessedAudio) -> Result<PathBuf, String> {
    let path = std::env::temp_dir().join(format!(
        "amcp-voxtral-mistralrs-{}-{}.wav",
        std::process::id(),
        monotonic_nanos()
    ));
    let mut bytes = Vec::with_capacity(44 + audio.samples.len() * 2);
    let data_len = (audio.samples.len() * 2) as u32;
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.extend_from_slice(&audio.sample_rate.to_le_bytes());
    bytes.extend_from_slice(&(audio.sample_rate * 2).to_le_bytes());
    bytes.extend_from_slice(&2_u16.to_le_bytes());
    bytes.extend_from_slice(&16_u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    for sample in &audio.samples {
        let pcm = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        bytes.extend_from_slice(&pcm.to_le_bytes());
    }
    std::fs::write(&path, bytes)
        .map_err(|error| format!("failed to write temporary Voxtral WAV: {error}"))?;
    Ok(path)
}

#[cfg(feature = "voxtral")]
fn monotonic_nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0)
}

#[cfg(feature = "voxtral")]
fn transcription_prompt(language: Option<&str>, previous_text: Option<&str>) -> String {
    let language = language_name(language)
        .map(|language| format!(" in {language}"))
        .unwrap_or_default();
    let previous = previous_text
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(|text| format!(" Previous transcript context: {text}"))
        .unwrap_or_default();
    format!("Transcribe this audio{language}. Return only the transcription.{previous}")
}

#[cfg(any(feature = "voxtral", test))]
fn translation_prompt(
    source_language: Option<&str>,
    target_language: &str,
    transcript_text: &str,
    previous_text: Option<&str>,
) -> String {
    let source = language_name(source_language)
        .map(|language| format!("The source speech language is {language}. "))
        .unwrap_or_default();
    let previous = previous_text
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(|text| format!("Previous transcript context: {text}\n"))
        .unwrap_or_default();
    let target =
        language_name(Some(target_language)).unwrap_or_else(|| target_language.to_string());
    format!("{source}{previous}Original transcript: {transcript_text}\nTranslate the audio and original transcript into {target}. Return only the translated text.")
}

#[cfg(feature = "voxtral")]
fn normalize_target_language(language: Option<&str>) -> Option<String> {
    let language = language?.trim().to_ascii_lowercase();
    match language.as_str() {
        "" | "auto" | "none" => None,
        "ja" | "japanese" => Some("ja".to_string()),
        "en" | "english" => Some("en".to_string()),
        "zh" | "chinese" => Some("zh".to_string()),
        other => Some(other.to_string()),
    }
}

#[cfg(any(feature = "voxtral", test))]
fn language_name(language: Option<&str>) -> Option<String> {
    match language?.trim().to_ascii_lowercase().as_str() {
        "" | "auto" | "none" => None,
        "ja" | "japanese" => Some("Japanese".to_string()),
        "en" | "english" => Some("English".to_string()),
        "zh" | "chinese" => Some("Chinese".to_string()),
        "ko" | "korean" => Some("Korean".to_string()),
        "fr" | "french" => Some("French".to_string()),
        "de" | "german" => Some("German".to_string()),
        "es" | "spanish" => Some("Spanish".to_string()),
        "it" | "italian" => Some("Italian".to_string()),
        "pt" | "portuguese" => Some("Portuguese".to_string()),
        "nl" | "dutch" => Some("Dutch".to_string()),
        "hi" | "hindi" => Some("Hindi".to_string()),
        other => Some(other.to_string()),
    }
}

fn is_japanese_language(language: Option<&str>) -> bool {
    matches!(
        language
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("ja" | "japanese")
    )
}

#[cfg(any(feature = "voxtral", test))]
fn chat_completions_url(base_url: &str) -> String {
    format!(
        "{}/v1/chat/completions",
        base_url.trim().trim_end_matches('/')
    )
}

#[cfg(feature = "voxtral")]
fn file_url(path: &Path) -> String {
    let path = path.to_string_lossy().replace('\\', "/");
    if path.starts_with('/') {
        format!("file://{path}")
    } else {
        format!("file:///{path}")
    }
}

#[cfg(feature = "voxtral")]
fn max_tokens() -> usize {
    std::env::var("AMCP_VOXTRAL_MISTRALRS_MAX_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .or_else(|| {
            std::env::var("AMCP_VOXTRAL_MAX_TOKENS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
        })
        .unwrap_or(512)
        .clamp(1, 4096)
}

fn env_string(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_japanese_requests_to_mistralrs() {
        assert!(should_route_to_mistralrs_with_runtime(
            Some("ja"),
            None,
            None
        ));
        assert!(should_route_to_mistralrs_with_runtime(
            Some("en"),
            Some("ja"),
            None
        ));
        assert!(should_route_to_mistralrs_with_runtime(
            Some("en"),
            None,
            Some("mistralrs")
        ));
        assert!(!should_route_to_mistralrs_with_runtime(
            Some("en"),
            None,
            None
        ));
    }

    #[test]
    fn builds_mistralrs_chat_endpoint_url() {
        assert_eq!(
            chat_completions_url("http://127.0.0.1:1234/"),
            "http://127.0.0.1:1234/v1/chat/completions"
        );
    }

    #[test]
    fn parses_openai_chat_completion_text() {
        let body = r#"{"choices":[{"message":{"content":" こんにちは "}}]}"#;
        assert_eq!(parse_chat_completion_content(body).unwrap(), "こんにちは");
    }

    #[test]
    fn prompt_mentions_target_translation_language() {
        let prompt = translation_prompt(Some("en"), "ja", "hello", Some("previous"));
        assert!(prompt.contains("source speech language is English"));
        assert!(prompt.contains("Previous transcript context: previous"));
        assert!(prompt.contains("Original transcript: hello"));
        assert!(prompt.contains("into Japanese"));
    }
}
