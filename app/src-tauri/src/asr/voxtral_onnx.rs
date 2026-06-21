use super::audio::PreprocessedAudio;
use crate::accelerator::HardwareBackend;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[cfg(feature = "voxtral")]
use ort::value::Tensor;

const VOXTRAL_ONNX_REPO_ID: &str = "onnx-community/Voxtral-Mini-3B-2507-ONNX";
#[cfg(feature = "voxtral")]
const VOXTRAL_BOS_TOKEN_ID: i64 = 1;
#[cfg(feature = "voxtral")]
const VOXTRAL_EOS_TOKEN_ID: i64 = 2;
#[cfg(feature = "voxtral")]
const VOXTRAL_INST_START_TOKEN_ID: i64 = 3;
#[cfg(feature = "voxtral")]
const VOXTRAL_INST_END_TOKEN_ID: i64 = 4;
#[cfg(feature = "voxtral")]
const VOXTRAL_AUDIO_TOKEN_ID: i64 = 24;
#[cfg(feature = "voxtral")]
const VOXTRAL_BEGIN_AUDIO_TOKEN_ID: i64 = 25;
#[cfg(feature = "voxtral")]
const VOXTRAL_TRANSCRIBE_TOKEN_ID: i64 = 34;
#[cfg(feature = "voxtral")]
const VOXTRAL_SUPPORTED_TRANSCRIPTION_LANGUAGES: &[&str] =
    &["en", "fr", "de", "es", "it", "pt", "nl", "hi"];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VoxtralOnnxConfig {
    pub model_path: PathBuf,
    pub audio_encoder_path: Option<PathBuf>,
    pub embed_tokens_path: Option<PathBuf>,
    pub decoder_path: Option<PathBuf>,
    pub tokenizer_path: Option<PathBuf>,
    pub providers: Vec<HardwareBackend>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VoxtralOnnxSessionInfo {
    pub model_path: PathBuf,
    pub providers: Vec<HardwareBackend>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VoxtralOnnxTranscription {
    pub text: String,
    pub transcript_text: String,
    pub translated_text: Option<String>,
    pub target_language: Option<String>,
    pub generated_tokens: usize,
    pub audio_frames: usize,
    pub audio_encoder_path: PathBuf,
    pub decoder_path: PathBuf,
}

pub fn resolve_voxtral_model_path() -> Option<PathBuf> {
    std::env::var("AMCP_VOXTRAL_ONNX_MODEL_PATH")
        .ok()
        .filter(|path| !path.trim().is_empty())
        .map(PathBuf::from)
        .or_else(|| Some(default_voxtral_model_dir().join("model.onnx")))
}

pub fn configure_voxtral_onnx(available_backends: &[HardwareBackend]) -> Option<VoxtralOnnxConfig> {
    let split = resolve_voxtral_split_paths();
    let model_path = split
        .as_ref()
        .map(|paths| paths.decoder_path.clone())
        .or_else(resolve_voxtral_model_path)?;
    Some(VoxtralOnnxConfig {
        model_path,
        audio_encoder_path: split.as_ref().map(|paths| paths.audio_encoder_path.clone()),
        embed_tokens_path: split.as_ref().map(|paths| paths.embed_tokens_path.clone()),
        decoder_path: split.as_ref().map(|paths| paths.decoder_path.clone()),
        tokenizer_path: split.as_ref().map(|paths| paths.tokenizer_path.clone()),
        providers: voxtral_provider_plan(available_backends),
    })
}

#[derive(Debug, Clone)]
struct VoxtralSplitPaths {
    audio_encoder_path: PathBuf,
    embed_tokens_path: PathBuf,
    decoder_path: PathBuf,
    tokenizer_path: PathBuf,
}

fn resolve_voxtral_split_paths() -> Option<VoxtralSplitPaths> {
    let explicit_audio_encoder = env_path("AMCP_VOXTRAL_AUDIO_ENCODER_PATH");
    let explicit_embed_tokens = env_path("AMCP_VOXTRAL_EMBED_TOKENS_PATH");
    let explicit_decoder = env_path("AMCP_VOXTRAL_DECODER_PATH");
    let explicit_tokenizer = env_path("AMCP_VOXTRAL_TOKENIZER_PATH");
    if explicit_audio_encoder.is_some()
        || explicit_embed_tokens.is_some()
        || explicit_decoder.is_some()
        || explicit_tokenizer.is_some()
    {
        return Some(VoxtralSplitPaths {
            audio_encoder_path: explicit_audio_encoder?,
            embed_tokens_path: explicit_embed_tokens?,
            decoder_path: explicit_decoder?,
            tokenizer_path: explicit_tokenizer?,
        });
    }

    let model_dir = default_voxtral_model_dir();
    let onnx_dir = model_dir.join("onnx");
    let base_dir = if onnx_dir.is_dir() {
        onnx_dir
    } else {
        model_dir
    };
    Some(VoxtralSplitPaths {
        audio_encoder_path: resolve_voxtral_onnx_path(
            &base_dir,
            "audio_encoder",
            &["q4", "q4f16", "fp16", "int8", "quantized", "uint8", "bnb4"],
        ),
        embed_tokens_path: resolve_voxtral_onnx_path(
            &base_dir,
            "embed_tokens",
            &["q4", "fp16", "quantized"],
        ),
        decoder_path: resolve_voxtral_onnx_path(
            &base_dir,
            "decoder_model_merged",
            &["q4", "q4f16", "fp16"],
        ),
        tokenizer_path: base_dir
            .parent()
            .map(|parent| parent.join("tokenizer.json"))
            .filter(|path| path.is_file())
            .unwrap_or_else(|| base_dir.join("tokenizer.json")),
    })
}

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var(key)
        .ok()
        .filter(|path| !path.trim().is_empty())
        .map(PathBuf::from)
}

pub fn default_voxtral_model_dir() -> PathBuf {
    env_path("AMCP_VOXTRAL_MODEL_DIR").unwrap_or_else(|| {
        if let Some(model_root) = env_path("AMCP_MODEL_DIR") {
            return model_root.join("voxtral");
        }
        existing_hf_snapshot_dir().unwrap_or_else(default_hf_repo_cache_dir)
    })
}

#[cfg(feature = "voxtral")]
pub fn ensure_default_voxtral_model_cached() -> Result<Option<PathBuf>, String> {
    if has_explicit_voxtral_paths() || env_path("AMCP_VOXTRAL_MODEL_DIR").is_some() {
        return Ok(None);
    }
    if env_path("AMCP_MODEL_DIR").is_some() {
        return Ok(None);
    }

    let variant = voxtral_onnx_variant();
    let api = hf_hub::api::sync::Api::new()
        .map_err(|error| format!("failed to create Hugging Face Hub API: {error}"))?;
    let repo = api.repo(hf_hub::Repo::new(
        VOXTRAL_ONNX_REPO_ID.to_string(),
        hf_hub::RepoType::Model,
    ));
    let tokenizer_path = repo
        .get("tokenizer.json")
        .map_err(|error| format!("failed to resolve Voxtral tokenizer: {error}"))?;
    let model_dir = tokenizer_path
        .parent()
        .ok_or_else(|| "Voxtral tokenizer path has no parent directory".to_string())?
        .to_path_buf();

    for path in voxtral_hf_required_files(&variant) {
        repo.get(&path)
            .map_err(|error| format!("failed to resolve Voxtral asset {path}: {error}"))?;
    }
    for path in voxtral_hf_optional_external_data(&variant) {
        let _ = repo.get(&path);
    }

    Ok(Some(model_dir))
}

#[cfg(feature = "voxtral")]
fn has_explicit_voxtral_paths() -> bool {
    [
        "AMCP_VOXTRAL_AUDIO_ENCODER_PATH",
        "AMCP_VOXTRAL_EMBED_TOKENS_PATH",
        "AMCP_VOXTRAL_DECODER_PATH",
        "AMCP_VOXTRAL_TOKENIZER_PATH",
        "AMCP_VOXTRAL_ONNX_MODEL_PATH",
    ]
    .iter()
    .any(|key| env_path(key).is_some())
}

#[cfg(feature = "voxtral")]
fn voxtral_onnx_variant() -> String {
    std::env::var("AMCP_VOXTRAL_ONNX_VARIANT")
        .ok()
        .map(|variant| variant.trim().to_string())
        .filter(|variant| !variant.is_empty())
        .unwrap_or_else(|| "q4".to_string())
}

#[cfg(feature = "voxtral")]
fn voxtral_hf_required_files(variant: &str) -> Vec<String> {
    [
        format!("onnx/audio_encoder_{variant}.onnx"),
        format!("onnx/embed_tokens_{variant}.onnx"),
        format!("onnx/decoder_model_merged_{variant}.onnx"),
    ]
    .into_iter()
    .collect()
}

#[cfg(feature = "voxtral")]
fn voxtral_hf_optional_external_data(variant: &str) -> Vec<String> {
    let mut files = vec![
        format!("onnx/audio_encoder_{variant}.onnx_data"),
        format!("onnx/embed_tokens_{variant}.onnx_data"),
        format!("onnx/decoder_model_merged_{variant}.onnx_data"),
    ];
    if variant == "q4" {
        files.push("onnx/decoder_model_merged_q4.onnx_data_1".to_string());
    }
    files
}

fn existing_hf_snapshot_dir() -> Option<PathBuf> {
    let repo_dir = default_hf_repo_cache_dir();
    let revision = std::fs::read_to_string(repo_dir.join("refs").join("main")).ok()?;
    let revision = revision.trim();
    if revision.is_empty() {
        return None;
    }
    let snapshot = repo_dir.join("snapshots").join(revision);
    snapshot.is_dir().then_some(snapshot)
}

fn default_hf_repo_cache_dir() -> PathBuf {
    default_hf_hub_cache_dir().join(format!(
        "models--{}",
        VOXTRAL_ONNX_REPO_ID.replace('/', "--")
    ))
}

fn default_hf_hub_cache_dir() -> PathBuf {
    if let Some(cache) = env_path("HF_HUB_CACHE") {
        return cache;
    }
    let hf_home = env_path("HF_HOME").unwrap_or_else(|| {
        env_path("XDG_CACHE_HOME")
            .unwrap_or_else(default_user_cache_dir)
            .join("huggingface")
    });
    hf_home.join("hub")
}

fn default_user_cache_dir() -> PathBuf {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(".cache")
}

fn resolve_voxtral_onnx_path(base_dir: &Path, stem: &str, fallback_variants: &[&str]) -> PathBuf {
    if let Ok(variant) = std::env::var("AMCP_VOXTRAL_ONNX_VARIANT") {
        let variant = variant.trim();
        if !variant.is_empty() {
            return variant_path(base_dir, stem, variant);
        }
    }

    let canonical = base_dir.join(format!("{stem}.onnx"));
    if canonical.is_file() {
        return canonical;
    }

    fallback_variants
        .iter()
        .map(|variant| variant_path(base_dir, stem, variant))
        .find(|path| path.is_file())
        .unwrap_or(canonical)
}

fn variant_path(base_dir: &Path, stem: &str, variant: &str) -> PathBuf {
    base_dir.join(format!("{stem}_{variant}.onnx"))
}

pub fn voxtral_provider_plan(available_backends: &[HardwareBackend]) -> Vec<HardwareBackend> {
    let mut providers = Vec::new();
    for backend in [
        HardwareBackend::Cuda,
        HardwareBackend::DirectMl,
        HardwareBackend::CoreMl,
        HardwareBackend::NnApi,
        HardwareBackend::Vulkan,
        HardwareBackend::Cpu,
    ] {
        if available_backends.contains(&backend) || backend == HardwareBackend::Cpu {
            providers.push(backend);
        }
    }
    providers
}

#[cfg(feature = "voxtral")]
pub fn validate_voxtral_session(
    config: &VoxtralOnnxConfig,
) -> Result<VoxtralOnnxSessionInfo, String> {
    if let Some(paths) = split_paths_from_config(config) {
        validate_split_files(&paths)?;
        return create_split_session_info(&paths, &config.providers);
    }

    if !config.model_path.is_file() {
        return Err(format!(
            "Voxtral ONNX model file does not exist: {}",
            config.model_path.display()
        ));
    }

    create_session_info(&config.model_path, &config.providers)
}

#[cfg(not(feature = "voxtral"))]
pub fn validate_voxtral_session(
    _config: &VoxtralOnnxConfig,
) -> Result<VoxtralOnnxSessionInfo, String> {
    Err("voxtral feature is not enabled".to_string())
}

#[cfg(feature = "voxtral")]
pub fn transcribe_voxtral_audio(
    audio: &PreprocessedAudio,
    language: Option<&str>,
    target_language: Option<&str>,
    previous_text: Option<&str>,
    available_backends: &[HardwareBackend],
) -> Result<Option<VoxtralOnnxTranscription>, String> {
    validate_voxtral_transcription_language(language)?;
    let _ = ensure_default_voxtral_model_cached()?;
    let Some(config) = configure_voxtral_onnx(available_backends) else {
        return Ok(None);
    };
    let Some(paths) = split_paths_from_config(&config) else {
        return Ok(None);
    };
    validate_split_files(&paths)?;

    let tokenizer = tokenizers::Tokenizer::from_file(&paths.tokenizer_path)
        .map_err(|error| format!("failed to load Voxtral tokenizer: {error}"))?;
    let mut audio_encoder = create_session(&paths.audio_encoder_path, &config.providers)?;
    let mut embed_tokens = create_session(&paths.embed_tokens_path, &config.providers)?;
    let mut decoder = create_session(&paths.decoder_path, &config.providers)?;

    let (audio_embeds, audio_frames, embed_dim) =
        encode_audio_embeddings(audio, &mut audio_encoder)?;
    if audio_frames == 0 || embed_dim == 0 {
        return Ok(Some(VoxtralOnnxTranscription {
            text: String::new(),
            transcript_text: String::new(),
            translated_text: None,
            target_language: normalize_target_language(target_language),
            generated_tokens: 0,
            audio_frames,
            audio_encoder_path: paths.audio_encoder_path,
            decoder_path: paths.decoder_path,
        }));
    }

    let max_tokens = std::env::var("AMCP_VOXTRAL_MAX_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(256)
        .clamp(1, 1024);
    let transcription_prompt =
        build_voxtral_transcription_prompt_tokens(&tokenizer, audio_frames, language)?;
    let (transcript_text, transcript_tokens) = run_voxtral_generation(
        &tokenizer,
        &mut embed_tokens,
        &mut decoder,
        &audio_embeds,
        audio_frames,
        embed_dim,
        transcription_prompt,
        max_tokens,
    )?;
    let target_language = normalize_target_language(target_language);
    let mut translation_tokens = 0usize;
    let translated_text = if let Some(target_language) = target_language.as_deref() {
        let instruction =
            translation_instruction(language, target_language, &transcript_text, previous_text);
        let translation_prompt =
            build_voxtral_chat_prompt_tokens(&tokenizer, audio_frames, &instruction)?;
        let (translated, tokens) = run_voxtral_generation(
            &tokenizer,
            &mut embed_tokens,
            &mut decoder,
            &audio_embeds,
            audio_frames,
            embed_dim,
            translation_prompt,
            max_tokens,
        )?;
        translation_tokens = tokens;
        if translated.is_empty() {
            None
        } else {
            Some(translated)
        }
    } else {
        None
    };

    Ok(Some(VoxtralOnnxTranscription {
        text: translated_text
            .clone()
            .unwrap_or_else(|| transcript_text.clone()),
        transcript_text,
        translated_text,
        target_language,
        generated_tokens: transcript_tokens + translation_tokens,
        audio_frames,
        audio_encoder_path: paths.audio_encoder_path,
        decoder_path: paths.decoder_path,
    }))
}

#[cfg(feature = "voxtral")]
fn run_voxtral_generation(
    tokenizer: &tokenizers::Tokenizer,
    embed_tokens: &mut ort::session::Session,
    decoder: &mut ort::session::Session,
    audio_embeds: &[f32],
    audio_frames: usize,
    embed_dim: usize,
    prompt_tokens: Vec<i64>,
    max_tokens: usize,
) -> Result<(String, usize), String> {
    let mut inputs_embeds = embed_input_ids(embed_tokens, &prompt_tokens)?;
    splice_audio_embeddings(&mut inputs_embeds, audio_embeds, audio_frames, embed_dim)?;

    let generated_ids = generate_tokens(
        decoder,
        embed_tokens,
        inputs_embeds,
        prompt_tokens.len(),
        max_tokens,
    )?;
    let generated_u32 = generated_ids
        .iter()
        .map(|id| *id as u32)
        .collect::<Vec<_>>();
    let text = tokenizer
        .decode(&generated_u32, true)
        .map_err(|error| format!("failed to decode Voxtral tokens: {error}"))?
        .trim()
        .to_string();
    Ok((text, generated_ids.len()))
}

#[cfg(feature = "voxtral")]
fn build_voxtral_transcription_prompt_tokens(
    tokenizer: &tokenizers::Tokenizer,
    audio_frames: usize,
    language: Option<&str>,
) -> Result<Vec<i64>, String> {
    let instruction_tokens = if let Some(language) = normalize_transcription_language(language) {
        encode_prompt_text(tokenizer, &format!("lang:{language} [TRANSCRIBE]"))?
    } else {
        vec![VOXTRAL_TRANSCRIBE_TOKEN_ID]
    };
    Ok(build_voxtral_transcription_prompt_from_instruction_tokens(
        audio_frames,
        &instruction_tokens,
    ))
}

#[cfg(feature = "voxtral")]
fn build_voxtral_transcription_prompt_from_instruction_tokens(
    audio_frames: usize,
    instruction_tokens: &[i64],
) -> Vec<i64> {
    let mut prompt_tokens = build_voxtral_audio_inst_prefix(audio_frames);
    prompt_tokens.extend_from_slice(instruction_tokens);
    prompt_tokens.push(VOXTRAL_INST_END_TOKEN_ID);
    prompt_tokens
}

#[cfg(feature = "voxtral")]
fn build_voxtral_chat_prompt_tokens(
    tokenizer: &tokenizers::Tokenizer,
    audio_frames: usize,
    instruction: &str,
) -> Result<Vec<i64>, String> {
    let instruction_ids = encode_prompt_text(tokenizer, instruction)?;
    let mut prompt_tokens = build_voxtral_audio_inst_prefix(audio_frames);
    prompt_tokens.extend(instruction_ids);
    prompt_tokens.push(VOXTRAL_INST_END_TOKEN_ID);
    Ok(prompt_tokens)
}

#[cfg(feature = "voxtral")]
fn build_voxtral_audio_inst_prefix(audio_frames: usize) -> Vec<i64> {
    let mut prompt_tokens = vec![
        VOXTRAL_BOS_TOKEN_ID,
        VOXTRAL_INST_START_TOKEN_ID,
        VOXTRAL_BEGIN_AUDIO_TOKEN_ID,
    ];
    prompt_tokens.extend(std::iter::repeat(VOXTRAL_AUDIO_TOKEN_ID).take(audio_frames));
    prompt_tokens
}

#[cfg(feature = "voxtral")]
fn encode_prompt_text(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<Vec<i64>, String> {
    tokenizer
        .encode(text, false)
        .map_err(|error| format!("failed to encode Voxtral prompt: {error}"))
        .map(|encoding| encoding.get_ids().iter().map(|id| *id as i64).collect())
}

#[cfg(feature = "voxtral")]
fn normalize_transcription_language(language: Option<&str>) -> Option<String> {
    let language = normalize_target_language(language)?;
    Some(match language.as_str() {
        "japanese" => "ja".to_string(),
        "english" => "en".to_string(),
        "chinese" => "zh".to_string(),
        other => other.to_string(),
    })
}

#[cfg(feature = "voxtral")]
fn validate_voxtral_transcription_language(language: Option<&str>) -> Result<(), String> {
    let Some(language) = normalize_transcription_language(language) else {
        return Ok(());
    };
    if VOXTRAL_SUPPORTED_TRANSCRIPTION_LANGUAGES.contains(&language.as_str()) {
        return Ok(());
    }
    Err(format!(
        "Voxtral Mini ONNX transcription does not support language '{language}'. Supported languages are: {}. Use Qwen3-ASR for Japanese transcription.",
        VOXTRAL_SUPPORTED_TRANSCRIPTION_LANGUAGES.join(", ")
    ))
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

#[cfg(feature = "voxtral")]
fn translation_instruction(
    source_language: Option<&str>,
    target_language: &str,
    transcript_text: &str,
    previous_text: Option<&str>,
) -> String {
    let source_hint = language_name(source_language)
        .map(|name| format!("The source speech language is {name}. "))
        .unwrap_or_default();
    let context_hint = previous_text
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(|text| format!("Previous transcript context: {text}\n"))
        .unwrap_or_default();
    format!(
        "{source_hint}{context_hint}Original transcript: {transcript_text}\nTranslate the original transcript into {target}. Return only the translated text.",
        target = language_name(Some(target_language)).unwrap_or_else(|| target_language.to_string())
    )
}

#[cfg(feature = "voxtral")]
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

#[cfg(not(feature = "voxtral"))]
pub fn transcribe_voxtral_audio(
    _audio: &PreprocessedAudio,
    _language: Option<&str>,
    _target_language: Option<&str>,
    _previous_text: Option<&str>,
    _available_backends: &[HardwareBackend],
) -> Result<Option<VoxtralOnnxTranscription>, String> {
    Ok(None)
}

#[cfg(feature = "voxtral")]
fn split_paths_from_config(config: &VoxtralOnnxConfig) -> Option<VoxtralSplitPaths> {
    Some(VoxtralSplitPaths {
        audio_encoder_path: config.audio_encoder_path.clone()?,
        embed_tokens_path: config.embed_tokens_path.clone()?,
        decoder_path: config.decoder_path.clone()?,
        tokenizer_path: config.tokenizer_path.clone()?,
    })
}

#[cfg(feature = "voxtral")]
fn validate_split_files(paths: &VoxtralSplitPaths) -> Result<(), String> {
    for (label, path) in [
        ("audio_encoder", &paths.audio_encoder_path),
        ("embed_tokens", &paths.embed_tokens_path),
        ("decoder", &paths.decoder_path),
        ("tokenizer", &paths.tokenizer_path),
    ] {
        if !path.is_file() {
            return Err(format!(
                "Voxtral {label} file does not exist: {}",
                path.display()
            ));
        }
    }
    Ok(())
}

#[cfg(feature = "voxtral")]
fn create_session(
    model_path: &Path,
    _providers: &[HardwareBackend],
) -> Result<ort::session::Session, String> {
    let mut builder = ort::session::Session::builder()
        .map_err(|error| format!("failed to create ONNX Runtime session builder: {error}"))?;

    #[cfg(any(feature = "voxtral-cuda", feature = "voxtral-directml"))]
    {
        let execution_providers = build_execution_providers(_providers);
        if !execution_providers.is_empty() {
            builder = builder
                .with_execution_providers(execution_providers)
                .map_err(|error| format!("failed to register ONNX execution providers: {error}"))?;
        }
    }

    builder.commit_from_file(model_path).map_err(|error| {
        format!(
            "failed to load ONNX model {}: {error}",
            model_path.display()
        )
    })
}

#[cfg(feature = "voxtral")]
fn create_split_session_info(
    paths: &VoxtralSplitPaths,
    providers: &[HardwareBackend],
) -> Result<VoxtralOnnxSessionInfo, String> {
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for (label, path) in [
        ("audio_encoder", &paths.audio_encoder_path),
        ("embed_tokens", &paths.embed_tokens_path),
        ("decoder", &paths.decoder_path),
    ] {
        let session = create_session(path, providers)?;
        inputs.extend(
            session
                .inputs()
                .iter()
                .map(|input| format!("{label}:{}", input.name())),
        );
        outputs.extend(
            session
                .outputs()
                .iter()
                .map(|output| format!("{label}:{}", output.name())),
        );
    }

    Ok(VoxtralOnnxSessionInfo {
        model_path: paths.decoder_path.clone(),
        providers: providers.to_vec(),
        inputs,
        outputs,
    })
}

#[cfg(feature = "voxtral")]
fn create_session_info(
    model_path: &Path,
    providers: &[HardwareBackend],
) -> Result<VoxtralOnnxSessionInfo, String> {
    let session = create_session(model_path, providers)?;
    let inputs = session
        .inputs()
        .iter()
        .map(|input| input.name().to_string())
        .collect();
    let outputs = session
        .outputs()
        .iter()
        .map(|output| output.name().to_string())
        .collect();

    Ok(VoxtralOnnxSessionInfo {
        model_path: model_path.to_path_buf(),
        providers: providers.to_vec(),
        inputs,
        outputs,
    })
}

#[cfg(feature = "voxtral")]
fn encode_audio_embeddings(
    audio: &PreprocessedAudio,
    session: &mut ort::session::Session,
) -> Result<(Vec<f32>, usize, usize), String> {
    let chunk_samples = 30 * audio.sample_rate as usize;
    let total_chunks = audio.samples.len().div_ceil(chunk_samples).max(1);
    let mut all_embeds = Vec::new();
    let mut total_frames = 0usize;
    let mut embed_dim = 0usize;

    for chunk_index in 0..total_chunks {
        let start = chunk_index * chunk_samples;
        let end = (start + chunk_samples).min(audio.samples.len());
        let chunk = &audio.samples[start..end];
        let mel = log_mel_features(chunk, audio.sample_rate);
        let input_name = session
            .inputs()
            .first()
            .map(|input| input.name().to_string())
            .ok_or_else(|| "Voxtral audio encoder has no inputs".to_string())?;
        let input_tensor = Tensor::from_array(([1usize, 128, 3000], mel.into_boxed_slice()))
            .map_err(|error| format!("failed to create Voxtral mel tensor: {error}"))?;
        let outputs = session
            .run(ort::inputs![input_name => input_tensor])
            .map_err(|error| format!("Voxtral audio encoder inference failed: {error}"))?;
        if outputs.len() == 0 {
            return Err("Voxtral audio encoder returned no outputs".to_string());
        }
        let output = outputs
            .get("pooler_output")
            .unwrap_or_else(|| &outputs[outputs.len().saturating_sub(1)]);
        let (shape, data) = output
            .try_extract_tensor::<f32>()
            .map_err(|error| format!("failed to read Voxtral audio embeddings: {error}"))?;
        let dims = shape.iter().map(|dim| *dim as usize).collect::<Vec<_>>();
        let (frames, dim) = match dims.as_slice() {
            [1, frames, dim] => (*frames, *dim),
            [frames, dim] => (*frames, *dim),
            other => {
                return Err(format!(
                    "unexpected Voxtral audio embedding shape: {other:?}"
                ))
            }
        };
        total_frames += frames;
        embed_dim = dim;
        all_embeds.extend_from_slice(data);
    }

    Ok((all_embeds, total_frames, embed_dim))
}

#[cfg(feature = "voxtral")]
fn embed_input_ids(
    session: &mut ort::session::Session,
    input_ids: &[i64],
) -> Result<Vec<f32>, String> {
    let input_name = session
        .inputs()
        .first()
        .map(|input| input.name().to_string())
        .unwrap_or_else(|| "input_ids".to_string());
    let tensor = Tensor::from_array((
        [1usize, input_ids.len()],
        input_ids.to_vec().into_boxed_slice(),
    ))
    .map_err(|error| format!("failed to create Voxtral input_ids tensor: {error}"))?;
    let outputs = session
        .run(ort::inputs![input_name => tensor])
        .map_err(|error| format!("Voxtral token embedding inference failed: {error}"))?;
    let (_shape, data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|error| format!("failed to read Voxtral token embeddings: {error}"))?;
    Ok(data.to_vec())
}

#[cfg(feature = "voxtral")]
fn splice_audio_embeddings(
    inputs_embeds: &mut [f32],
    audio_embeds: &[f32],
    audio_frames: usize,
    embed_dim: usize,
) -> Result<(), String> {
    let splice_start = 3usize * embed_dim;
    let splice_len = audio_frames * embed_dim;
    if inputs_embeds.len() < splice_start + splice_len {
        return Err("Voxtral prompt embedding buffer is shorter than audio span".to_string());
    }
    if audio_embeds.len() < splice_len {
        return Err("Voxtral audio embedding buffer is shorter than expected".to_string());
    }
    inputs_embeds[splice_start..splice_start + splice_len]
        .copy_from_slice(&audio_embeds[..splice_len]);
    Ok(())
}

#[cfg(feature = "voxtral")]
fn generate_tokens(
    decoder: &mut ort::session::Session,
    embed_tokens: &mut ort::session::Session,
    initial_inputs_embeds: Vec<f32>,
    initial_sequence_len: usize,
    max_tokens: usize,
) -> Result<Vec<i64>, String> {
    let num_layers = decoder
        .inputs()
        .iter()
        .filter(|input| input.name().ends_with(".key"))
        .count();
    if num_layers == 0 {
        return Err("Voxtral decoder exposes no past_key_values inputs".to_string());
    }

    let embed_dim = initial_inputs_embeds
        .len()
        .checked_div(initial_sequence_len)
        .ok_or_else(|| "invalid Voxtral prompt embedding length".to_string())?;
    let mut generated = Vec::new();
    let mut past_cache: Option<Vec<(Vec<usize>, Vec<f32>)>> = None;
    // ONNX Runtime cannot create a raw tensor with a zero-length dimension, so
    // the first call uses a masked dummy cache entry while logical token
    // positions still start at zero.
    let mut physical_past_len = 1usize;
    let mut logical_past_len = 0usize;

    for step in 0..max_tokens {
        let is_prefill = step == 0;
        let current_step_len = if is_prefill { initial_sequence_len } else { 1 };
        let current_total_len = physical_past_len + current_step_len;
        let inputs_embeds = if is_prefill {
            initial_inputs_embeds.clone()
        } else {
            let last_token = *generated
                .last()
                .ok_or_else(|| "missing previous Voxtral token".to_string())?;
            embed_input_ids(embed_tokens, &[last_token])?
        };

        let mut inputs = ort::inputs![
            "inputs_embeds" => Tensor::from_array((
                [1usize, current_step_len, embed_dim],
                inputs_embeds.into_boxed_slice()
            ))
            .map_err(|error| format!("failed to create Voxtral decoder embeddings tensor: {error}"))?,
            "attention_mask" => Tensor::from_array((
                [1usize, current_total_len],
                build_voxtral_attention_mask(current_total_len, true).into_boxed_slice()
            ))
            .map_err(|error| format!("failed to create Voxtral attention mask tensor: {error}"))?,
            "position_ids" => Tensor::from_array((
                [1usize, current_step_len],
                if is_prefill {
                    (logical_past_len as i64..(logical_past_len + current_step_len) as i64)
                        .collect::<Vec<_>>()
                        .into_boxed_slice()
                } else {
                    vec![logical_past_len as i64].into_boxed_slice()
                }
            ))
            .map_err(|error| format!("failed to create Voxtral position ids tensor: {error}"))?,
        ];

        for layer in 0..num_layers {
            let initial_key;
            let initial_value;
            let (key, value) = if let Some(cache) = past_cache.as_ref() {
                let key = cache
                    .get(layer * 2)
                    .ok_or_else(|| format!("missing Voxtral key cache for layer {layer}"))?;
                let value = cache
                    .get(layer * 2 + 1)
                    .ok_or_else(|| format!("missing Voxtral value cache for layer {layer}"))?;
                (key, value)
            } else {
                initial_key = (vec![1usize, 8, 1, 128], vec![0.0_f32; 8 * 128]);
                initial_value = (vec![1usize, 8, 1, 128], vec![0.0_f32; 8 * 128]);
                (&initial_key, &initial_value)
            };
            inputs.push((
                format!("past_key_values.{layer}.key").into(),
                Tensor::from_array((key.0.clone(), key.1.clone().into_boxed_slice()))
                    .map_err(|error| format!("failed to create Voxtral key cache tensor: {error}"))?
                    .into(),
            ));
            inputs.push((
                format!("past_key_values.{layer}.value").into(),
                Tensor::from_array((value.0.clone(), value.1.clone().into_boxed_slice()))
                    .map_err(|error| {
                        format!("failed to create Voxtral value cache tensor: {error}")
                    })?
                    .into(),
            ));
        }

        let outputs = decoder
            .run(inputs)
            .map_err(|error| format!("Voxtral decoder inference failed: {error}"))?;
        let (logit_shape, logits) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|error| format!("failed to read Voxtral logits: {error}"))?;
        let dims = logit_shape
            .iter()
            .map(|dim| *dim as usize)
            .collect::<Vec<_>>();
        let next_token = argmax_last_token(&dims, logits)?;
        if next_token == VOXTRAL_EOS_TOKEN_ID {
            break;
        }
        generated.push(next_token);

        let mut new_cache = Vec::with_capacity(num_layers * 2);
        for cache_index in 0..(num_layers * 2) {
            let (shape, data) = outputs[cache_index + 1]
                .try_extract_tensor::<f32>()
                .map_err(|error| {
                    format!("failed to read Voxtral cache output {cache_index}: {error}")
                })?;
            new_cache.push((
                shape.iter().map(|dim| *dim as usize).collect::<Vec<_>>(),
                data.to_vec(),
            ));
        }
        past_cache = Some(new_cache);
        physical_past_len = current_total_len;
        logical_past_len += current_step_len;
    }

    Ok(generated)
}

#[cfg(feature = "voxtral")]
fn build_voxtral_attention_mask(total_len: usize, has_dummy_past: bool) -> Vec<i64> {
    let mut mask = vec![1_i64; total_len];
    if has_dummy_past && !mask.is_empty() {
        mask[0] = 0;
    }
    mask
}

#[cfg(feature = "voxtral")]
fn argmax_last_token(shape: &[usize], logits: &[f32]) -> Result<i64, String> {
    let [batch, sequence, vocab] = shape else {
        return Err(format!("unexpected Voxtral logits shape: {shape:?}"));
    };
    if *batch != 1 || *sequence == 0 || *vocab == 0 {
        return Err(format!("invalid Voxtral logits shape: {shape:?}"));
    }
    let start = (sequence - 1) * vocab;
    let end = start + vocab;
    let (index, _) = logits
        .get(start..end)
        .ok_or_else(|| "Voxtral logits buffer is shorter than expected".to_string())?
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .ok_or_else(|| "Voxtral logits are empty".to_string())?;
    Ok(index as i64)
}

#[cfg(feature = "voxtral")]
fn log_mel_features(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    const N_FFT: usize = 400;
    const HOP_LENGTH: usize = 160;
    const N_MELS: usize = 128;
    const N_FRAMES: usize = 3000;

    let target_samples = sample_rate as usize * 30;
    let mut padded = vec![0.0_f32; target_samples.max(samples.len())];
    padded[..samples.len()].copy_from_slice(samples);
    padded.truncate(target_samples);

    let filters = mel_filter_bank(sample_rate, N_FFT, N_MELS);
    let window = (0..N_FFT)
        .map(|n| 0.5_f32 - 0.5_f32 * ((2.0 * std::f32::consts::PI * n as f32) / N_FFT as f32).cos())
        .collect::<Vec<_>>();
    let mut planner = rustfft::FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut output = vec![0.0_f32; N_MELS * N_FRAMES];
    let mut spectrum = vec![rustfft::num_complex::Complex::new(0.0_f32, 0.0_f32); N_FFT];
    let bins = N_FFT / 2 + 1;

    for frame in 0..N_FRAMES {
        let start = frame * HOP_LENGTH;
        for index in 0..N_FFT {
            let sample = padded.get(start + index).copied().unwrap_or(0.0);
            spectrum[index] = rustfft::num_complex::Complex::new(sample * window[index], 0.0);
        }
        fft.process(&mut spectrum);
        let power = spectrum
            .iter()
            .take(bins)
            .map(|value| value.norm_sqr())
            .collect::<Vec<_>>();
        for mel in 0..N_MELS {
            let energy = filters[mel]
                .iter()
                .zip(power.iter())
                .map(|(weight, power)| weight * power)
                .sum::<f32>()
                .max(1e-10);
            output[mel * N_FRAMES + frame] = energy.log10();
        }
    }

    let max_value = output.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for value in &mut output {
        *value = value.max(max_value - 8.0);
        *value = (*value + 4.0) / 4.0;
    }

    output
}

#[cfg(feature = "voxtral")]
fn mel_filter_bank(sample_rate: u32, n_fft: usize, n_mels: usize) -> Vec<Vec<f32>> {
    let bins = n_fft / 2 + 1;
    let min_mel = hz_to_mel(0.0);
    let max_mel = hz_to_mel(8000.0_f32.min(sample_rate as f32 / 2.0));
    let mel_points = (0..n_mels + 2)
        .map(|index| min_mel + (max_mel - min_mel) * index as f32 / (n_mels + 1) as f32)
        .map(mel_to_hz)
        .collect::<Vec<_>>();
    let fft_freqs = (0..bins)
        .map(|bin| bin as f32 * sample_rate as f32 / n_fft as f32)
        .collect::<Vec<_>>();
    let mut filters = vec![vec![0.0_f32; bins]; n_mels];

    for mel in 0..n_mels {
        let lower = mel_points[mel];
        let center = mel_points[mel + 1];
        let upper = mel_points[mel + 2];
        for (bin, freq) in fft_freqs.iter().enumerate() {
            filters[mel][bin] = if *freq >= lower && *freq <= center {
                (*freq - lower) / (center - lower).max(f32::EPSILON)
            } else if *freq > center && *freq <= upper {
                (upper - *freq) / (upper - center).max(f32::EPSILON)
            } else {
                0.0
            };
        }
    }

    filters
}

#[cfg(feature = "voxtral")]
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

#[cfg(feature = "voxtral")]
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10_f32.powf(mel / 2595.0) - 1.0)
}

#[cfg(all(
    feature = "voxtral",
    any(feature = "voxtral-cuda", feature = "voxtral-directml")
))]
fn build_execution_providers(
    providers: &[HardwareBackend],
) -> Vec<ort::execution_providers::ExecutionProviderDispatch> {
    let mut eps = Vec::new();
    for provider in providers {
        match provider {
            #[cfg(feature = "voxtral-cuda")]
            HardwareBackend::Cuda => {
                eps.push(ort::execution_providers::CUDAExecutionProvider::default().build())
            }
            #[cfg(feature = "voxtral-directml")]
            HardwareBackend::DirectMl => {
                eps.push(ort::execution_providers::DirectMLExecutionProvider::default().build())
            }
            _ => {}
        }
    }
    eps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_plan_keeps_cpu_fallback() {
        let providers = voxtral_provider_plan(&[HardwareBackend::DirectMl]);

        assert_eq!(
            providers,
            vec![HardwareBackend::DirectMl, HardwareBackend::Cpu]
        );
    }

    #[cfg(feature = "voxtral")]
    #[test]
    fn log_mel_features_have_voxtral_shape() {
        let samples = vec![0.0_f32; 16_000];
        let features = log_mel_features(&samples, 16_000);

        assert_eq!(features.len(), 128 * 3000);
        assert!(features.iter().all(|value| value.is_finite()));
    }

    #[cfg(feature = "voxtral")]
    #[test]
    fn transcription_prompt_uses_voxtral_transcribe_mode() {
        let prompt = build_voxtral_transcription_prompt_from_instruction_tokens(
            3,
            &[9909, 1058, 2922, 1032, VOXTRAL_TRANSCRIBE_TOKEN_ID],
        );

        assert_eq!(
            prompt,
            vec![
                VOXTRAL_BOS_TOKEN_ID,
                VOXTRAL_INST_START_TOKEN_ID,
                VOXTRAL_BEGIN_AUDIO_TOKEN_ID,
                VOXTRAL_AUDIO_TOKEN_ID,
                VOXTRAL_AUDIO_TOKEN_ID,
                VOXTRAL_AUDIO_TOKEN_ID,
                9909,
                1058,
                2922,
                1032,
                VOXTRAL_TRANSCRIBE_TOKEN_ID,
                VOXTRAL_INST_END_TOKEN_ID,
            ]
        );
    }

    #[cfg(feature = "voxtral")]
    #[test]
    fn transcription_prompt_allows_language_auto_detect() {
        let prompt = build_voxtral_transcription_prompt_from_instruction_tokens(
            1,
            &[VOXTRAL_TRANSCRIBE_TOKEN_ID],
        );

        assert_eq!(
            prompt,
            vec![
                VOXTRAL_BOS_TOKEN_ID,
                VOXTRAL_INST_START_TOKEN_ID,
                VOXTRAL_BEGIN_AUDIO_TOKEN_ID,
                VOXTRAL_AUDIO_TOKEN_ID,
                VOXTRAL_TRANSCRIBE_TOKEN_ID,
                VOXTRAL_INST_END_TOKEN_ID,
            ]
        );
    }

    #[cfg(feature = "voxtral")]
    #[test]
    fn dummy_cache_attention_mask_hides_seed_token() {
        assert_eq!(build_voxtral_attention_mask(4, true), vec![0, 1, 1, 1]);
        assert_eq!(build_voxtral_attention_mask(4, false), vec![1, 1, 1, 1]);
    }

    #[cfg(feature = "voxtral")]
    #[test]
    fn transcription_language_guard_rejects_japanese_for_voxtral() {
        assert!(validate_voxtral_transcription_language(Some("auto")).is_ok());
        assert!(validate_voxtral_transcription_language(Some("en")).is_ok());

        let error = validate_voxtral_transcription_language(Some("ja")).unwrap_err();
        assert!(error.contains("does not support language 'ja'"));
        assert!(error.contains("Use Qwen3-ASR for Japanese transcription"));
    }

    #[cfg(feature = "voxtral")]
    #[test]
    fn translation_instruction_keeps_original_transcript_context() {
        let instruction = translation_instruction(
            Some("en"),
            "ja",
            "Original transcript",
            Some("Previous context"),
        );

        assert!(instruction.contains("The source speech language is English."));
        assert!(instruction.contains("Previous transcript context: Previous context"));
        assert!(instruction.contains("Original transcript: Original transcript"));
        assert!(instruction.contains("Translate the original transcript into Japanese"));
    }
}
