use super::audio::PreprocessedAudio;
use crate::accelerator::HardwareBackend;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[cfg(feature = "voxtral")]
use ort::value::Tensor;
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
        std::env::var("AMCP_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                std::env::current_dir()
                    .unwrap_or_else(|_| PathBuf::from("."))
                    .join("models")
            })
            .join("voxtral")
    })
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
    let (transcript_text, transcript_tokens) = run_voxtral_generation(
        &tokenizer,
        &mut embed_tokens,
        &mut decoder,
        &audio_embeds,
        audio_frames,
        embed_dim,
        "Transcribe the audio. Return only the transcription.",
        max_tokens,
    )?;
    let target_language = normalize_target_language(target_language);
    let mut translation_tokens = 0usize;
    let translated_text = if let Some(target_language) = target_language.as_deref() {
        let instruction = translation_instruction(language, target_language, previous_text);
        let (translated, tokens) = run_voxtral_generation(
            &tokenizer,
            &mut embed_tokens,
            &mut decoder,
            &audio_embeds,
            audio_frames,
            embed_dim,
            &instruction,
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
    instruction: &str,
    max_tokens: usize,
) -> Result<(String, usize), String> {
    let instruction = tokenizer
        .encode(instruction, false)
        .map_err(|error| format!("failed to encode Voxtral prompt: {error}"))?;
    let instruction_ids = instruction.get_ids().iter().map(|id| *id as i64);
    let mut prompt_tokens = vec![1_i64, 3, 25];
    prompt_tokens.extend(std::iter::repeat(24_i64).take(audio_frames));
    prompt_tokens.extend(instruction_ids);
    prompt_tokens.push(4);

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
        "{source_hint}{context_hint}Translate the audio into {target}. Return only the translated text.",
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
        let (shape, data) = outputs[0]
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
    // ONNX Runtime requires all tensor dimensions to be >= 1, so the initial
    // cache uses a single zero token instead of the logical zero-length cache.
    let mut current_past_len = 1usize;

    for step in 0..max_tokens {
        let is_prefill = step == 0;
        let current_step_len = if is_prefill { initial_sequence_len } else { 1 };
        let current_total_len = current_past_len + current_step_len;
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
                vec![1_i64; current_total_len].into_boxed_slice()
            ))
            .map_err(|error| format!("failed to create Voxtral attention mask tensor: {error}"))?,
            "position_ids" => Tensor::from_array((
                [1usize, current_step_len],
                if is_prefill {
                    (current_past_len as i64..(current_past_len + current_step_len) as i64)
                        .collect::<Vec<_>>()
                        .into_boxed_slice()
                } else {
                    vec![current_past_len as i64].into_boxed_slice()
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
        if next_token == 2 {
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
        current_past_len = current_total_len;
    }

    Ok(generated)
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
}
