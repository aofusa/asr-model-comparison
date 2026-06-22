use super::audio::PreprocessedAudio;
#[cfg(feature = "voxtral-llamacpp-native")]
use super::audio::TARGET_SAMPLE_RATE;
use crate::accelerator::HardwareBackend;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const DEFAULT_LLAMA_MODEL_ID: &str = "mistralai/Voxtral-Mini-4B-Realtime-2602-GGUF";
const DEFAULT_CONTEXT_SIZE: u32 = 4096;
const DEFAULT_BATCH_SIZE: u32 = 512;
const DEFAULT_MAX_TOKENS: usize = 512;
const DEFAULT_GPU_LAYERS: u32 = 999;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VoxtralLlamaCppConfig {
    pub model_path: Option<PathBuf>,
    pub mmproj_path: Option<PathBuf>,
    pub model_id: String,
    pub context_size: u32,
    pub batch_size: u32,
    pub max_tokens: usize,
    pub n_gpu_layers: u32,
    pub use_gpu: bool,
    pub print_timings: bool,
    pub providers: Vec<HardwareBackend>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VoxtralLlamaCppValidation {
    pub model_exists: bool,
    pub mmproj_exists: bool,
    pub configured: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VoxtralLlamaCppTranscription {
    pub text: String,
    pub transcript_text: String,
    pub translated_text: Option<String>,
    pub target_language: Option<String>,
    pub model_id: String,
    pub model_path: PathBuf,
    pub mmproj_path: PathBuf,
    pub n_gpu_layers: u32,
}

pub fn configure_voxtral_llamacpp(available_backends: &[HardwareBackend]) -> VoxtralLlamaCppConfig {
    let providers = llamacpp_provider_plan(available_backends);
    let use_gpu = env_bool("AMCP_VOXTRAL_LLAMA_USE_GPU").unwrap_or_else(|| {
        cfg!(feature = "voxtral-llamacpp-vulkan")
            && providers
                .iter()
                .any(|backend| *backend != HardwareBackend::Cpu)
    });
    VoxtralLlamaCppConfig {
        model_path: env_path("AMCP_VOXTRAL_LLAMA_MODEL_PATH"),
        mmproj_path: env_path("AMCP_VOXTRAL_LLAMA_MMPROJ_PATH"),
        model_id: env_string("AMCP_VOXTRAL_LLAMA_MODEL_ID")
            .unwrap_or_else(|| DEFAULT_LLAMA_MODEL_ID.to_string()),
        context_size: env_u32("AMCP_VOXTRAL_LLAMA_CONTEXT_SIZE")
            .unwrap_or(DEFAULT_CONTEXT_SIZE)
            .clamp(512, 262_144),
        batch_size: env_u32("AMCP_VOXTRAL_LLAMA_BATCH_SIZE")
            .unwrap_or(DEFAULT_BATCH_SIZE)
            .clamp(1, 8192),
        max_tokens: max_tokens(),
        n_gpu_layers: env_u32("AMCP_VOXTRAL_LLAMA_GPU_LAYERS").unwrap_or(DEFAULT_GPU_LAYERS),
        use_gpu,
        print_timings: env_bool("AMCP_VOXTRAL_LLAMA_PRINT_TIMINGS").unwrap_or(false),
        providers,
    }
}

pub fn validate_voxtral_llamacpp_config(
    config: &VoxtralLlamaCppConfig,
) -> VoxtralLlamaCppValidation {
    let model_exists = config
        .model_path
        .as_ref()
        .is_some_and(|path| path.is_file());
    let mmproj_exists = config
        .mmproj_path
        .as_ref()
        .is_some_and(|path| path.is_file());
    VoxtralLlamaCppValidation {
        model_exists,
        mmproj_exists,
        configured: model_exists && mmproj_exists,
    }
}

pub fn voxtral_runtime_preference() -> Option<String> {
    env_string("AMCP_VOXTRAL_RUNTIME").map(|runtime| runtime.to_ascii_lowercase())
}

pub fn is_llamacpp_requested() -> bool {
    voxtral_runtime_preference()
        .as_deref()
        .is_some_and(is_llamacpp_runtime)
}

pub fn should_route_to_llamacpp(language: Option<&str>, target_language: Option<&str>) -> bool {
    should_route_to_llamacpp_with_runtime(
        language,
        target_language,
        voxtral_runtime_preference().as_deref(),
    )
}

pub fn should_route_to_llamacpp_with_runtime(
    language: Option<&str>,
    target_language: Option<&str>,
    runtime: Option<&str>,
) -> bool {
    if runtime.is_some_and(is_llamacpp_runtime) {
        return true;
    }
    is_japanese_language(language) || is_japanese_language(target_language)
}

pub fn llamacpp_provider_plan(available_backends: &[HardwareBackend]) -> Vec<HardwareBackend> {
    let mut providers = Vec::new();
    for backend in [
        HardwareBackend::Vulkan,
        HardwareBackend::Cuda,
        HardwareBackend::Metal,
        HardwareBackend::Cpu,
    ] {
        if available_backends.contains(&backend) || backend == HardwareBackend::Cpu {
            providers.push(backend);
        }
    }
    providers
}

#[cfg(feature = "voxtral-llamacpp-native")]
pub fn transcribe_voxtral_audio(
    audio: &PreprocessedAudio,
    language: Option<&str>,
    target_language: Option<&str>,
    previous_text: Option<&str>,
    available_backends: &[HardwareBackend],
) -> Result<Option<VoxtralLlamaCppTranscription>, String> {
    let config = configure_voxtral_llamacpp(available_backends);
    let validation = validate_voxtral_llamacpp_config(&config);
    if !validation.configured {
        return Err(missing_config_error(&config, &validation));
    }
    if audio.sample_rate != TARGET_SAMPLE_RATE {
        return Err(format!(
            "Voxtral llama.cpp expects {TARGET_SAMPLE_RATE}Hz mono PCM, got {}Hz.",
            audio.sample_rate
        ));
    }

    let transcript_prompt = transcription_prompt(language, previous_text);
    let transcript_text = run_llamacpp_generation(audio, &config, &transcript_prompt)?;
    let target_language = normalize_target_language(target_language);
    let translated_text = if let Some(target_language) = target_language.as_deref() {
        let instruction =
            translation_prompt(language, target_language, &transcript_text, previous_text);
        let translated = run_llamacpp_generation(audio, &config, &instruction)?;
        if translated.is_empty() {
            None
        } else {
            Some(translated)
        }
    } else {
        None
    };

    Ok(Some(VoxtralLlamaCppTranscription {
        text: translated_text
            .clone()
            .unwrap_or_else(|| transcript_text.clone()),
        transcript_text,
        translated_text,
        target_language,
        model_id: config.model_id,
        model_path: config.model_path.expect("validated model path exists"),
        mmproj_path: config.mmproj_path.expect("validated mmproj path exists"),
        n_gpu_layers: config.n_gpu_layers,
    }))
}

#[cfg(not(feature = "voxtral-llamacpp-native"))]
pub fn transcribe_voxtral_audio(
    _audio: &PreprocessedAudio,
    _language: Option<&str>,
    _target_language: Option<&str>,
    _previous_text: Option<&str>,
    _available_backends: &[HardwareBackend],
) -> Result<Option<VoxtralLlamaCppTranscription>, String> {
    Err(
        "Voxtral llama.cpp runtime is selected, but the voxtral-llamacpp-native feature is not enabled."
            .to_string(),
    )
}

#[cfg(feature = "voxtral-llamacpp-native")]
fn run_llamacpp_generation(
    audio: &PreprocessedAudio,
    config: &VoxtralLlamaCppConfig,
    instruction: &str,
) -> Result<String, String> {
    use llama_cpp_4::{
        context::params::LlamaContextParams,
        llama_backend::LlamaBackend,
        llama_batch::LlamaBatch,
        model::{params::LlamaModelParams, LlamaModel, Special},
        mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputChunks, MtmdInputText},
        sampling::LlamaSampler,
    };
    use std::num::NonZeroU32;

    let model_path = config
        .model_path
        .as_ref()
        .ok_or_else(|| "AMCP_VOXTRAL_LLAMA_MODEL_PATH is not configured.".to_string())?;
    let mmproj_path = config
        .mmproj_path
        .as_ref()
        .ok_or_else(|| "AMCP_VOXTRAL_LLAMA_MMPROJ_PATH is not configured.".to_string())?;

    let backend = LlamaBackend::init().map_err(|error| {
        format!("failed to initialize llama.cpp backend for Voxtral llama.cpp: {error}")
    })?;
    let model_params = LlamaModelParams::default().with_n_gpu_layers(config.n_gpu_layers);
    let model =
        LlamaModel::load_from_file(&backend, model_path, &model_params).map_err(|error| {
            format!(
                "failed to load Voxtral GGUF model at {}: {error}",
                model_path.display()
            )
        })?;
    let n_batch = config.batch_size.min(config.context_size).max(1);
    let mut lctx = model
        .new_context(
            &backend,
            LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(config.context_size))
                .with_n_batch(n_batch)
                .with_n_ubatch(n_batch),
        )
        .map_err(|error| format!("failed to create Voxtral llama.cpp context: {error}"))?;

    MtmdContext::void_helper_logs();
    let mtmd_params = MtmdContextParams::default()
        .use_gpu(config.use_gpu)
        .print_timings(config.print_timings);
    let mtmd_ctx =
        MtmdContext::init_from_file(mmproj_path, &model, mtmd_params).map_err(|error| {
            format!(
                "failed to load Voxtral mmproj at {}: {error}",
                mmproj_path.display()
            )
        })?;
    if !mtmd_ctx.supports_audio() {
        return Err(format!(
            "Voxtral mmproj at {} does not report audio support.",
            mmproj_path.display()
        ));
    }

    let bitmap = MtmdBitmap::from_audio(&audio.samples)
        .map_err(|error| format!("failed to create Voxtral audio bitmap: {error}"))?;
    let prompt = build_llamacpp_prompt(mtmd_ctx.marker(), instruction);
    let text = MtmdInputText::try_new(&prompt, true, true)
        .map_err(|error| format!("invalid Voxtral llama.cpp prompt: {error}"))?;
    let mut chunks = MtmdInputChunks::new();
    mtmd_ctx
        .tokenize(&text, &[&bitmap], &mut chunks)
        .map_err(|error| format!("failed to tokenize Voxtral llama.cpp prompt/audio: {error}"))?;

    let mut n_past = 0_i32;
    mtmd_ctx
        .eval_chunks(
            lctx.as_ptr(),
            &chunks,
            0,
            0,
            i32::try_from(n_batch).unwrap_or(i32::MAX),
            true,
            &mut n_past,
        )
        .map_err(|error| format!("failed to evaluate Voxtral llama.cpp prompt/audio: {error}"))?;

    let mut sampler = LlamaSampler::greedy();
    let vocab = model.get_vocab();
    let mut output = Vec::new();
    for _ in 0..config.max_tokens {
        let token = sampler.sample(&lctx, -1);
        if vocab.is_eog(token) {
            break;
        }
        sampler.accept(token);
        if !vocab.is_control(token) {
            let bytes = model
                .token_to_bytes(token, Special::Plaintext)
                .map_err(|error| format!("failed to decode Voxtral token: {error}"))?;
            output.extend(bytes);
        }

        let mut batch = LlamaBatch::new(1, 1);
        batch
            .add(token, n_past, &[0], true)
            .map_err(|error| format!("failed to append Voxtral generated token: {error}"))?;
        lctx.decode(&mut batch)
            .map_err(|error| format!("failed to decode Voxtral generated token: {error}"))?;
        n_past += 1;
    }

    Ok(clean_model_output(&String::from_utf8_lossy(&output)))
}

#[cfg(feature = "voxtral-llamacpp-native")]
fn build_llamacpp_prompt(marker: &str, instruction: &str) -> String {
    format!("[INST] {marker}\n{} [/INST]", instruction.trim())
}

#[cfg(any(feature = "voxtral-llamacpp-native", test))]
fn missing_config_error(
    config: &VoxtralLlamaCppConfig,
    validation: &VoxtralLlamaCppValidation,
) -> String {
    format!(
        "Voxtral llama.cpp runtime is selected, but GGUF/mmproj files are not configured or missing. \
Set AMCP_VOXTRAL_LLAMA_MODEL_PATH and AMCP_VOXTRAL_LLAMA_MMPROJ_PATH. \
model_path={:?} model_exists={} mmproj_path={:?} mmproj_exists={}",
        config.model_path, validation.model_exists, config.mmproj_path, validation.mmproj_exists
    )
}

#[cfg(feature = "voxtral-llamacpp-native")]
fn transcription_prompt(language: Option<&str>, previous_text: Option<&str>) -> String {
    let language = language_name(language)
        .map(|language| format!(" in {language}"))
        .unwrap_or_else(|| " in the spoken language".to_string());
    let previous = previous_text
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(|text| format!(" Previous transcript context: {text}"))
        .unwrap_or_default();
    format!("Transcribe this audio{language}. Return only the transcription.{previous}")
}

#[cfg(any(feature = "voxtral-llamacpp-native", test))]
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

#[cfg(feature = "voxtral-llamacpp-native")]
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

#[cfg(any(feature = "voxtral-llamacpp-native", test))]
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

fn is_llamacpp_runtime(runtime: &str) -> bool {
    matches!(
        runtime.trim().to_ascii_lowercase().as_str(),
        "llamacpp" | "llama.cpp" | "llama-cpp" | "llama_cpp"
    )
}

#[cfg(feature = "voxtral-llamacpp-native")]
fn clean_model_output(text: &str) -> String {
    text.replace("[/INST]", "")
        .replace("[INST]", "")
        .trim()
        .to_string()
}

fn max_tokens() -> usize {
    std::env::var("AMCP_VOXTRAL_LLAMA_MAX_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .or_else(|| {
            std::env::var("AMCP_VOXTRAL_MAX_TOKENS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
        })
        .unwrap_or(DEFAULT_MAX_TOKENS)
        .clamp(1, 4096)
}

fn env_string(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn env_path(key: &str) -> Option<PathBuf> {
    env_string(key).map(PathBuf::from)
}

fn env_u32(key: &str) -> Option<u32> {
    env_string(key).and_then(|value| value.parse::<u32>().ok())
}

fn env_bool(key: &str) -> Option<bool> {
    env_string(key).and_then(|value| match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_japanese_requests_to_llamacpp() {
        assert!(should_route_to_llamacpp_with_runtime(
            Some("ja"),
            None,
            None
        ));
        assert!(should_route_to_llamacpp_with_runtime(
            Some("en"),
            Some("ja"),
            None
        ));
        assert!(should_route_to_llamacpp_with_runtime(
            Some("en"),
            None,
            Some("llamacpp")
        ));
        assert!(!should_route_to_llamacpp_with_runtime(
            Some("en"),
            None,
            None
        ));
    }

    #[test]
    fn reports_missing_model_and_mmproj() {
        let config = VoxtralLlamaCppConfig {
            model_path: None,
            mmproj_path: None,
            model_id: DEFAULT_LLAMA_MODEL_ID.to_string(),
            context_size: DEFAULT_CONTEXT_SIZE,
            batch_size: DEFAULT_BATCH_SIZE,
            max_tokens: DEFAULT_MAX_TOKENS,
            n_gpu_layers: DEFAULT_GPU_LAYERS,
            use_gpu: false,
            print_timings: false,
            providers: vec![HardwareBackend::Cpu],
        };
        let validation = validate_voxtral_llamacpp_config(&config);

        assert!(!validation.model_exists);
        assert!(!validation.mmproj_exists);
        assert!(!validation.configured);
        assert!(
            missing_config_error(&config, &validation).contains("AMCP_VOXTRAL_LLAMA_MODEL_PATH")
        );
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
