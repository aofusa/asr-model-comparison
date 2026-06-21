use crate::accelerator::HardwareBackend;
#[cfg(feature = "qwen")]
use crate::qwen_native::{
    AsrInference, SpeechTranslationResult, TranscribeOptions, TranscribeResult,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QwenCandleConfig {
    pub model_dir: PathBuf,
    pub model_id: String,
    pub backends: Vec<HardwareBackend>,
    pub auto_download: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QwenCandleValidation {
    pub model_dir: PathBuf,
    pub config_exists: bool,
    pub tokenizer_exists: bool,
    pub weights_exist: bool,
    pub auto_download: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QwenCandleTranscription {
    pub text: String,
    pub language: Option<String>,
    pub model_dir: PathBuf,
    pub device: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QwenCandleSpeechTranslation {
    pub transcript_text: String,
    pub translated_text: Option<String>,
    pub language: Option<String>,
    pub target_language: Option<String>,
    pub model_dir: PathBuf,
    pub device: String,
}

pub fn configure_qwen_candle(
    model_id: &str,
    available_backends: &[HardwareBackend],
) -> QwenCandleConfig {
    let hf_model_id = qwen_hf_model_id(model_id);
    let explicit_dir = qwen_explicit_model_dir(&hf_model_id);
    let model_dir = explicit_dir.unwrap_or_else(|| {
        std::env::var("AMCP_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| default_qwen_model_root())
            .join("qwen")
            .join(hf_model_id.replace('/', "--"))
    });

    QwenCandleConfig {
        model_dir,
        model_id: hf_model_id,
        backends: qwen_backend_plan(available_backends),
        auto_download: qwen_auto_download_enabled(),
    }
}

fn qwen_explicit_model_dir(hf_model_id: &str) -> Option<PathBuf> {
    if cfg!(test) {
        return None;
    }
    let env_var = if hf_model_id == "voiceping-ai/qwen3-asr-ja-en-speech-translation" {
        "AMCP_QWEN_TRANSLATION_MODEL_DIR"
    } else {
        "AMCP_QWEN_MODEL_DIR"
    };
    std::env::var(env_var)
        .ok()
        .filter(|path| !path.trim().is_empty())
        .map(PathBuf::from)
}

fn default_qwen_model_root() -> PathBuf {
    if cfg!(test) {
        return std::env::temp_dir().join("amcp-qwen-test-models");
    }
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("models")
}

pub fn qwen_backend_plan(available_backends: &[HardwareBackend]) -> Vec<HardwareBackend> {
    let mut backends = Vec::new();
    for backend in [
        HardwareBackend::Cuda,
        HardwareBackend::Metal,
        HardwareBackend::Cpu,
    ] {
        if available_backends.contains(&backend) || backend == HardwareBackend::Cpu {
            backends.push(backend);
        }
    }
    backends
}

pub fn validate_qwen_candle(config: &QwenCandleConfig) -> Result<QwenCandleValidation, String> {
    let config_path = config.model_dir.join("config.json");
    let tokenizer_path = config.model_dir.join("tokenizer.json");
    Ok(QwenCandleValidation {
        model_dir: config.model_dir.clone(),
        config_exists: config_path.is_file(),
        tokenizer_exists: tokenizer_path.is_file(),
        weights_exist: qwen_weights_exist(&config.model_dir),
        auto_download: config.auto_download,
    })
}

#[cfg(feature = "qwen")]
pub fn transcribe_qwen_audio(
    samples: &[f32],
    model_id: &str,
    language: Option<&str>,
    _previous_text: Option<&str>,
    available_backends: &[HardwareBackend],
) -> Result<Option<QwenCandleTranscription>, String> {
    if samples.is_empty() {
        return Ok(Some(QwenCandleTranscription {
            text: String::new(),
            language: language.map(str::to_string),
            model_dir: PathBuf::new(),
            device: "none".to_string(),
        }));
    }

    let samples = samples.to_vec();
    let model_id = model_id.to_string();
    let language = language.map(str::to_string);
    let available_backends = available_backends.to_vec();
    return std::thread::spawn(move || {
        transcribe_qwen_audio_blocking(
            &samples,
            &model_id,
            language.as_deref(),
            &available_backends,
        )
    })
    .join()
    .map_err(|_| "Qwen Candle inference thread panicked".to_string())?;
}

#[cfg(feature = "qwen")]
pub fn transcribe_qwen_audio_with_translation(
    samples: &[f32],
    model_id: &str,
    language: Option<&str>,
    target_language: Option<&str>,
    _previous_text: Option<&str>,
    available_backends: &[HardwareBackend],
) -> Result<Option<QwenCandleSpeechTranslation>, String> {
    if samples.is_empty() {
        return Ok(Some(QwenCandleSpeechTranslation {
            transcript_text: String::new(),
            translated_text: None,
            language: language.map(str::to_string),
            target_language: normalize_target_language(target_language),
            model_dir: PathBuf::new(),
            device: "none".to_string(),
        }));
    }

    let samples = samples.to_vec();
    let model_id = model_id.to_string();
    let language = language.map(str::to_string);
    let target_language = target_language.map(str::to_string);
    let available_backends = available_backends.to_vec();
    std::thread::spawn(move || {
        transcribe_qwen_audio_with_translation_blocking(
            &samples,
            &model_id,
            language.as_deref(),
            target_language.as_deref(),
            &available_backends,
        )
    })
    .join()
    .map_err(|_| "Qwen Candle translation inference thread panicked".to_string())?
}

#[cfg(feature = "qwen")]
fn transcribe_qwen_audio_blocking(
    samples: &[f32],
    model_id: &str,
    language: Option<&str>,
    available_backends: &[HardwareBackend],
) -> Result<Option<QwenCandleTranscription>, String> {
    let config = configure_qwen_candle(model_id, available_backends);
    if !qwen_model_ready(&config.model_dir) && !config.auto_download {
        return Ok(None);
    }
    let model_dir = ensure_qwen_model_dir(&config)?;
    let device = qwen_device(&config.backends);
    let device_label = qwen_device_label(&device);
    let engine = AsrInference::load(&model_dir, device)
        .map_err(|error| format!("failed to load Qwen Candle model: {error}"))?;
    let mut options = TranscribeOptions::default().with_max_new_tokens(512);
    if let Some(language) = qwen_language_name(language) {
        options = options.with_language(language);
    }
    let result = engine
        .transcribe_samples(samples, options)
        .map_err(|error| format!("Qwen Candle transcription failed: {error}"))?;

    Ok(Some(QwenCandleTranscription {
        text: clean_qwen_text(&result.text),
        language: Some(result.language),
        model_dir,
        device: device_label,
    }))
}

#[cfg(feature = "qwen")]
fn transcribe_qwen_audio_with_translation_blocking(
    samples: &[f32],
    model_id: &str,
    language: Option<&str>,
    target_language: Option<&str>,
    available_backends: &[HardwareBackend],
) -> Result<Option<QwenCandleSpeechTranslation>, String> {
    let config = configure_qwen_candle(model_id, available_backends);
    if !qwen_model_ready(&config.model_dir) && !config.auto_download {
        return Ok(None);
    }
    let model_dir = ensure_qwen_model_dir(&config)?;
    let device = qwen_device(&config.backends);
    let device_label = qwen_device_label(&device);
    let engine = load_qwen_engine(&model_dir, device.clone())?;

    let target_language = normalize_target_language(target_language);
    let source_language = normalize_target_language(language);
    let (transcript, translated_text) =
        if target_language.is_some() && target_language != source_language {
            let translation_model_id =
                qwen_translation_model_id(source_language.as_deref(), target_language.as_deref())
                    .unwrap_or_else(|| config.model_id.clone());

            if translation_model_id == config.model_id {
                let result = run_qwen_speech_translation(
                    &engine,
                    samples,
                    language,
                    target_language.as_deref(),
                )?;
                let translated_text = result
                    .translated
                    .map(|translated| translated.text)
                    .filter(|text| !text.trim().is_empty());
                (result.transcript, translated_text)
            } else {
                let transcript = run_qwen_inference(&engine, samples, language)?;
                let translation_config =
                    configure_qwen_candle(&translation_model_id, available_backends);
                let translation_model_dir = ensure_qwen_model_dir(&translation_config)?;
                let translation_engine = load_qwen_engine(&translation_model_dir, device)?;
                let translated =
                    run_qwen_inference(&translation_engine, samples, target_language.as_deref())?;
                let translated_text = if translated.text.trim().is_empty() {
                    None
                } else {
                    Some(translated.text)
                };
                (transcript, translated_text)
            }
        } else {
            (run_qwen_inference(&engine, samples, language)?, None)
        };

    Ok(Some(QwenCandleSpeechTranslation {
        transcript_text: transcript.text,
        translated_text,
        language: Some(transcript.language),
        target_language,
        model_dir,
        device: device_label,
    }))
}

#[cfg(feature = "qwen")]
fn load_qwen_engine(
    model_dir: &std::path::Path,
    device: candle_core::Device,
) -> Result<AsrInference, String> {
    AsrInference::load(model_dir, device)
        .map_err(|error| format!("failed to load Qwen Candle model: {error}"))
}

#[cfg(feature = "qwen")]
fn run_qwen_inference(
    engine: &AsrInference,
    samples: &[f32],
    language: Option<&str>,
) -> Result<TranscribeResult, String> {
    let mut options = TranscribeOptions::default().with_max_new_tokens(512);
    if let Some(language) = qwen_language_name(language) {
        options = options.with_language(language);
    }
    let mut result = engine
        .transcribe_samples(samples, options)
        .map_err(|error| format!("Qwen Candle transcription failed: {error}"))?;
    result.text = clean_qwen_text(&result.text);
    Ok(result)
}

#[cfg(feature = "qwen")]
fn run_qwen_speech_translation(
    engine: &AsrInference,
    samples: &[f32],
    language: Option<&str>,
    target_language: Option<&str>,
) -> Result<SpeechTranslationResult, String> {
    let source_language = qwen_language_name(language);
    let target_language = qwen_language_name(target_language);
    let mut result = engine
        .transcribe_samples_with_translation(
            samples,
            source_language.as_deref(),
            target_language.as_deref(),
            512,
        )
        .map_err(|error| format!("Qwen Candle speech translation failed: {error}"))?;
    result.transcript.text = clean_qwen_text(&result.transcript.text);
    if let Some(translated) = &mut result.translated {
        translated.text = clean_qwen_text(&translated.text);
    }
    Ok(result)
}

#[cfg(not(feature = "qwen"))]
pub fn transcribe_qwen_audio(
    _samples: &[f32],
    _model_id: &str,
    _language: Option<&str>,
    _previous_text: Option<&str>,
    _available_backends: &[HardwareBackend],
) -> Result<Option<QwenCandleTranscription>, String> {
    Ok(None)
}

#[cfg(not(feature = "qwen"))]
pub fn transcribe_qwen_audio_with_translation(
    _samples: &[f32],
    _model_id: &str,
    _language: Option<&str>,
    _target_language: Option<&str>,
    _previous_text: Option<&str>,
    _available_backends: &[HardwareBackend],
) -> Result<Option<QwenCandleSpeechTranslation>, String> {
    Ok(None)
}

#[cfg(feature = "qwen")]
fn ensure_qwen_model_dir(config: &QwenCandleConfig) -> Result<PathBuf, String> {
    if qwen_model_ready(&config.model_dir) {
        return Ok(config.model_dir.clone());
    }
    if !config.auto_download {
        return Err(format!(
            "Qwen model files are missing in {} and auto-download is disabled.",
            config.model_dir.display()
        ));
    }
    let cache_dir = config
        .model_dir
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| config.model_dir.clone());
    AsrInference::from_pretrained(&config.model_id, &cache_dir, qwen_device(&config.backends))
        .map(|_| cache_dir.join(config.model_id.replace('/', "--")))
        .map_err(|error| {
            format!(
                "failed to download/load Qwen model {}: {error}",
                config.model_id
            )
        })
}

#[cfg(feature = "qwen")]
fn qwen_model_ready(model_dir: &std::path::Path) -> bool {
    model_dir.join("config.json").is_file()
        && model_dir.join("tokenizer.json").is_file()
        && qwen_weights_exist(model_dir)
}

fn qwen_weights_exist(model_dir: &std::path::Path) -> bool {
    model_dir.join("model.safetensors").is_file()
        || model_dir.join("model.safetensors.index.json").is_file()
        || std::fs::read_dir(model_dir)
            .ok()
            .into_iter()
            .flat_map(|entries| entries.filter_map(Result::ok))
            .any(|entry| {
                entry
                    .path()
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.ends_with(".safetensors"))
            })
}

fn qwen_auto_download_enabled() -> bool {
    std::env::var("AMCP_QWEN_DISABLE_DOWNLOAD")
        .map(|value| !matches!(value.trim(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(!cfg!(test))
}

#[cfg(feature = "qwen")]
fn qwen_device(backends: &[HardwareBackend]) -> candle_core::Device {
    if backends.contains(&HardwareBackend::Cuda) {
        #[cfg(feature = "qwen-cuda")]
        if let Ok(device) = candle_core::Device::new_cuda(0) {
            return device;
        }
    }
    candle_core::Device::Cpu
}

#[cfg(feature = "qwen")]
fn qwen_device_label(device: &candle_core::Device) -> String {
    match device {
        candle_core::Device::Cpu => "cpu".to_string(),
        other => format!("{other:?}"),
    }
}

fn qwen_hf_model_id(model_id: &str) -> String {
    if model_id.contains('/') {
        return model_id.to_string();
    }
    match model_id {
        "qwen3-asr-1.7b" => "Qwen/Qwen3-ASR-1.7B".to_string(),
        "qwen3-asr-ja-en-translation" => {
            "voiceping-ai/qwen3-asr-ja-en-speech-translation".to_string()
        }
        _ => "Qwen/Qwen3-ASR-0.6B".to_string(),
    }
}

#[cfg(any(test, feature = "qwen"))]
fn qwen_translation_model_id(
    source_language: Option<&str>,
    target_language: Option<&str>,
) -> Option<String> {
    if let Ok(model_id) = std::env::var("AMCP_QWEN_TRANSLATION_MODEL_ID") {
        let model_id = model_id.trim();
        if !model_id.is_empty() {
            return Some(model_id.to_string());
        }
    }

    match (source_language, target_language) {
        (Some("ja"), Some("en")) | (Some("en"), Some("ja")) => {
            Some("voiceping-ai/qwen3-asr-ja-en-speech-translation".to_string())
        }
        _ => None,
    }
}

#[cfg(any(test, feature = "qwen"))]
fn qwen_language_name(language: Option<&str>) -> Option<String> {
    match language?.trim().to_ascii_lowercase().as_str() {
        "" | "auto" | "none" => None,
        "ja" | "japanese" => Some("japanese".to_string()),
        "en" | "english" => Some("english".to_string()),
        "zh" | "chinese" => Some("chinese".to_string()),
        "ko" | "korean" => Some("korean".to_string()),
        "fr" | "french" => Some("french".to_string()),
        "de" | "german" => Some("german".to_string()),
        "es" | "spanish" => Some("spanish".to_string()),
        "it" | "italian" => Some("italian".to_string()),
        "pt" | "portuguese" => Some("portuguese".to_string()),
        "ru" | "russian" => Some("russian".to_string()),
        "ar" | "arabic" => Some("arabic".to_string()),
        "hi" | "hindi" => Some("hindi".to_string()),
        "vi" | "vietnamese" => Some("vietnamese".to_string()),
        "th" | "thai" => Some("thai".to_string()),
        "id" | "indonesian" => Some("indonesian".to_string()),
        "tr" | "turkish" => Some("turkish".to_string()),
        "nl" | "dutch" => Some("dutch".to_string()),
        "pl" | "polish" => Some("polish".to_string()),
        "sv" | "swedish" => Some("swedish".to_string()),
        _ => None,
    }
}

#[cfg(any(test, feature = "qwen"))]
fn normalize_target_language(language: Option<&str>) -> Option<String> {
    match language?.trim().to_ascii_lowercase().as_str() {
        "" | "auto" | "none" => None,
        "ja" | "japanese" => Some("ja".to_string()),
        "en" | "english" => Some("en".to_string()),
        "zh" | "chinese" => Some("zh".to_string()),
        "ko" | "korean" => Some("ko".to_string()),
        "fr" | "french" => Some("fr".to_string()),
        "de" | "german" => Some("de".to_string()),
        "es" | "spanish" => Some("es".to_string()),
        "it" | "italian" => Some("it".to_string()),
        "pt" | "portuguese" => Some("pt".to_string()),
        "ru" | "russian" => Some("ru".to_string()),
        other if qwen_language_name(Some(other)).is_some() => Some(other.to_string()),
        _ => None,
    }
}

#[cfg(any(test, feature = "qwen"))]
fn clean_qwen_text(text: &str) -> String {
    text.replace("<asr_text>", "")
        .replace("<|im_end|>", "")
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_plan_keeps_cpu_fallback() {
        let backends = qwen_backend_plan(&[HardwareBackend::Wgpu]);

        assert_eq!(backends, vec![HardwareBackend::Cpu]);
    }

    #[test]
    fn backend_plan_uses_candle_supported_accelerators_only() {
        let backends = qwen_backend_plan(&[
            HardwareBackend::DirectMl,
            HardwareBackend::Cuda,
            HardwareBackend::Wgpu,
            HardwareBackend::Metal,
            HardwareBackend::Cpu,
        ]);

        assert_eq!(
            backends,
            vec![
                HardwareBackend::Cuda,
                HardwareBackend::Metal,
                HardwareBackend::Cpu
            ]
        );
    }

    #[test]
    fn maps_language_codes_to_qwen_names() {
        assert_eq!(qwen_language_name(Some("ja")).as_deref(), Some("japanese"));
        assert_eq!(qwen_language_name(Some("auto")), None);
    }

    #[test]
    fn cleans_qwen_asr_markers() {
        assert_eq!(clean_qwen_text("<asr_text>hello<|im_end|>"), "hello");
    }

    #[test]
    fn normalizes_translation_targets() {
        assert_eq!(
            normalize_target_language(Some("English")).as_deref(),
            Some("en")
        );
        assert_eq!(normalize_target_language(Some("none")), None);
    }

    #[test]
    fn qwen_translation_uses_translation_checkpoint_for_ja_en() {
        assert_eq!(
            qwen_translation_model_id(Some("ja"), Some("en")).as_deref(),
            Some("voiceping-ai/qwen3-asr-ja-en-speech-translation")
        );
    }
}
