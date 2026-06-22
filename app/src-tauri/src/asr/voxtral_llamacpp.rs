use super::audio::PreprocessedAudio;
#[cfg(any(
    feature = "voxtral-llamacpp-native",
    feature = "voxtral-llamacpp-realtime-patched"
))]
use super::audio::TARGET_SAMPLE_RATE;
use crate::accelerator::HardwareBackend;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const DEFAULT_LLAMA_MODEL_ID: &str = "acceldium/Voxtral-Mini-4B-Realtime-2602_GGUF";
const DEFAULT_LLAMA_REPO_ID: &str = DEFAULT_LLAMA_MODEL_ID;
const DEFAULT_LLAMA_MODEL_FILE: &str = "voxtral-realtime-4b-text-q8_0.gguf";
const DEFAULT_LLAMA_MMPROJ_FILE: &str = "voxtral-realtime-4b-mmproj-f16.gguf";
const DEFAULT_CONTEXT_SIZE: u32 = 4096;
const DEFAULT_BATCH_SIZE: u32 = 512;
const DEFAULT_MAX_TOKENS: usize = 512;
const DEFAULT_GPU_LAYERS: u32 = 999;
const HF_SNAPSHOT_SEARCH_DEPTH: usize = 4;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VoxtralLlamaCppConfig {
    pub model_path: Option<PathBuf>,
    pub mmproj_path: Option<PathBuf>,
    pub model_id: String,
    pub repo_id: String,
    pub model_file: Option<String>,
    pub mmproj_file: Option<String>,
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
        cfg!(any(
            feature = "voxtral-llamacpp-vulkan",
            feature = "voxtral-realtime-vulkan"
        )) && runtime_vulkan_available()
            && providers
                .iter()
                .any(|backend| *backend != HardwareBackend::Cpu)
    });
    let configured_repo_id = env_string("AMCP_VOXTRAL_LLAMA_REPO_ID");
    let repo_id = configured_repo_id
        .clone()
        .unwrap_or_else(|| DEFAULT_LLAMA_REPO_ID.to_string());
    let model_file = env_string("AMCP_VOXTRAL_LLAMA_MODEL_FILE").or_else(|| {
        configured_repo_id
            .is_none()
            .then(|| DEFAULT_LLAMA_MODEL_FILE.into())
    });
    let mmproj_file = env_string("AMCP_VOXTRAL_LLAMA_MMPROJ_FILE").or_else(|| {
        configured_repo_id
            .is_none()
            .then(|| DEFAULT_LLAMA_MMPROJ_FILE.into())
    });
    VoxtralLlamaCppConfig {
        model_path: env_path("AMCP_VOXTRAL_LLAMA_MODEL_PATH").or_else(|| {
            resolve_hf_cached_gguf(&repo_id, model_file.as_deref(), LlamaCppGgufKind::TextModel)
        }),
        mmproj_path: env_path("AMCP_VOXTRAL_LLAMA_MMPROJ_PATH").or_else(|| {
            resolve_hf_cached_gguf(&repo_id, mmproj_file.as_deref(), LlamaCppGgufKind::Mmproj)
        }),
        model_id: env_string("AMCP_VOXTRAL_LLAMA_MODEL_ID")
            .unwrap_or_else(|| DEFAULT_LLAMA_MODEL_ID.to_string()),
        repo_id,
        model_file,
        mmproj_file,
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

pub fn should_route_to_llamacpp(_language: Option<&str>, _target_language: Option<&str>) -> bool {
    true
}

pub fn should_route_to_llamacpp_with_runtime(
    _language: Option<&str>,
    _target_language: Option<&str>,
    _runtime: Option<&str>,
) -> bool {
    true
}

pub fn llamacpp_provider_plan(available_backends: &[HardwareBackend]) -> Vec<HardwareBackend> {
    let mut providers = Vec::new();
    let include_vulkan = cfg!(any(
        feature = "voxtral-llamacpp-vulkan",
        feature = "voxtral-realtime-vulkan"
    )) && runtime_vulkan_available();
    for backend in [HardwareBackend::Vulkan, HardwareBackend::Cpu] {
        if backend == HardwareBackend::Vulkan && !include_vulkan {
            continue;
        }
        if available_backends.contains(&backend) || backend == HardwareBackend::Cpu {
            providers.push(backend);
        }
    }
    providers
}

#[cfg(any(
    feature = "voxtral-llamacpp-native",
    feature = "voxtral-llamacpp-realtime-patched"
))]
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

    let transcript_text = run_llamacpp_transcription(audio, &config)?;
    let target_language = normalize_target_language(target_language);
    let translated_text = if let Some(target_language) = target_language.as_deref() {
        let instruction =
            translation_prompt(language, target_language, &transcript_text, previous_text);
        let translated = run_llamacpp_text_generation(&config, &instruction)?;
        if translated.is_empty() || is_low_quality_translation(&translated, target_language) {
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

#[cfg(not(any(
    feature = "voxtral-llamacpp-native",
    feature = "voxtral-llamacpp-realtime-patched"
)))]
pub fn transcribe_voxtral_audio(
    _audio: &PreprocessedAudio,
    _language: Option<&str>,
    _target_language: Option<&str>,
    _previous_text: Option<&str>,
    _available_backends: &[HardwareBackend],
) -> Result<Option<VoxtralLlamaCppTranscription>, String> {
    Err(
        "Voxtral llama.cpp runtime is selected, but the voxtral feature is not enabled with a native llama.cpp backend."
            .to_string(),
    )
}

#[cfg(feature = "voxtral-llamacpp-realtime-patched")]
fn run_llamacpp_transcription(
    audio: &PreprocessedAudio,
    config: &VoxtralLlamaCppConfig,
) -> Result<String, String> {
    use std::ffi::{CStr, CString};
    use std::os::raw::{c_char, c_float, c_int};

    extern "C" {
        fn amcp_voxtral_realtime_transcribe(
            model_path: *const c_char,
            mmproj_path: *const c_char,
            pcm_samples: *const c_float,
            n_samples: usize,
            context_size: u32,
            batch_size: u32,
            max_tokens: u32,
            n_gpu_layers: c_int,
            use_gpu: bool,
            print_timings: bool,
            out_text: *mut *mut c_char,
            out_error: *mut *mut c_char,
        ) -> c_int;
        fn amcp_voxtral_realtime_free(value: *mut c_char);
    }

    let model_path = config
        .model_path
        .as_ref()
        .ok_or_else(|| "AMCP_VOXTRAL_LLAMA_MODEL_PATH is not configured.".to_string())?;
    let mmproj_path = config
        .mmproj_path
        .as_ref()
        .ok_or_else(|| "AMCP_VOXTRAL_LLAMA_MMPROJ_PATH is not configured.".to_string())?;
    let model_path = CString::new(model_path.to_string_lossy().as_bytes())
        .map_err(|_| "Voxtral Realtime model path contains an embedded NUL byte.".to_string())?;
    let mmproj_path = CString::new(mmproj_path.to_string_lossy().as_bytes())
        .map_err(|_| "Voxtral Realtime mmproj path contains an embedded NUL byte.".to_string())?;

    let mut out_text: *mut c_char = std::ptr::null_mut();
    let mut out_error: *mut c_char = std::ptr::null_mut();
    let result = unsafe {
        amcp_voxtral_realtime_transcribe(
            model_path.as_ptr(),
            mmproj_path.as_ptr(),
            audio.samples.as_ptr(),
            audio.samples.len(),
            config.context_size,
            config.batch_size,
            u32::try_from(config.max_tokens).unwrap_or(u32::MAX),
            i32::try_from(config.n_gpu_layers).unwrap_or(i32::MAX),
            config.use_gpu,
            config.print_timings,
            &mut out_text,
            &mut out_error,
        )
    };

    let error = if out_error.is_null() {
        None
    } else {
        let error = unsafe { CStr::from_ptr(out_error) }
            .to_string_lossy()
            .into_owned();
        unsafe { amcp_voxtral_realtime_free(out_error) };
        Some(error)
    };

    if result != 0 {
        if !out_text.is_null() {
            unsafe { amcp_voxtral_realtime_free(out_text) };
        }
        return Err(error.unwrap_or_else(|| {
            format!("Voxtral Realtime patched llama.cpp bridge failed with code {result}.")
        }));
    }
    if out_text.is_null() {
        return Err("Voxtral Realtime patched llama.cpp bridge returned no text.".to_string());
    }

    let text = unsafe { CStr::from_ptr(out_text) }
        .to_string_lossy()
        .into_owned();
    unsafe { amcp_voxtral_realtime_free(out_text) };
    Ok(clean_model_output(&text))
}

#[cfg(feature = "voxtral-llamacpp-realtime-patched")]
fn run_llamacpp_text_generation(
    config: &VoxtralLlamaCppConfig,
    instruction: &str,
) -> Result<String, String> {
    use std::ffi::{CStr, CString};
    use std::os::raw::{c_char, c_int};

    extern "C" {
        fn amcp_voxtral_realtime_generate_text(
            model_path: *const c_char,
            prompt: *const c_char,
            context_size: u32,
            batch_size: u32,
            max_tokens: u32,
            n_gpu_layers: c_int,
            out_text: *mut *mut c_char,
            out_error: *mut *mut c_char,
        ) -> c_int;
        fn amcp_voxtral_realtime_free(value: *mut c_char);
    }

    let model_path = config
        .model_path
        .as_ref()
        .ok_or_else(|| "AMCP_VOXTRAL_LLAMA_MODEL_PATH is not configured.".to_string())?;
    let model_path = CString::new(model_path.to_string_lossy().as_bytes())
        .map_err(|_| "Voxtral Realtime model path contains an embedded NUL byte.".to_string())?;
    let prompt = CString::new(build_text_generation_prompt(instruction))
        .map_err(|_| "Voxtral Realtime text prompt contains an embedded NUL byte.".to_string())?;

    let mut out_text: *mut c_char = std::ptr::null_mut();
    let mut out_error: *mut c_char = std::ptr::null_mut();
    let result = unsafe {
        amcp_voxtral_realtime_generate_text(
            model_path.as_ptr(),
            prompt.as_ptr(),
            config.context_size,
            config.batch_size,
            u32::try_from(config.max_tokens).unwrap_or(u32::MAX),
            i32::try_from(config.n_gpu_layers).unwrap_or(i32::MAX),
            &mut out_text,
            &mut out_error,
        )
    };

    let error = if out_error.is_null() {
        None
    } else {
        let error = unsafe { CStr::from_ptr(out_error) }
            .to_string_lossy()
            .into_owned();
        unsafe { amcp_voxtral_realtime_free(out_error) };
        Some(error)
    };

    if result != 0 {
        if !out_text.is_null() {
            unsafe { amcp_voxtral_realtime_free(out_text) };
        }
        return Err(error.unwrap_or_else(|| {
            format!("Voxtral Realtime patched llama.cpp text generation failed with code {result}.")
        }));
    }
    if out_text.is_null() {
        return Err(
            "Voxtral Realtime patched llama.cpp text generation returned no text.".to_string(),
        );
    }

    let text = unsafe { CStr::from_ptr(out_text) }
        .to_string_lossy()
        .into_owned();
    unsafe { amcp_voxtral_realtime_free(out_text) };
    Ok(clean_model_output(&text))
}

#[cfg(all(
    feature = "voxtral-llamacpp-native",
    not(feature = "voxtral-llamacpp-realtime-patched")
))]
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

#[cfg(all(
    feature = "voxtral-llamacpp-native",
    not(feature = "voxtral-llamacpp-realtime-patched")
))]
fn build_llamacpp_prompt(marker: &str, instruction: &str) -> String {
    format!("[INST] {marker}\n{} [/INST]", instruction.trim())
}

#[cfg(feature = "voxtral-llamacpp-realtime-patched")]
fn build_text_generation_prompt(instruction: &str) -> String {
    instruction.trim().to_string()
}

#[cfg(any(
    feature = "voxtral-llamacpp-native",
    feature = "voxtral-llamacpp-realtime-patched",
    test
))]
fn missing_config_error(
    config: &VoxtralLlamaCppConfig,
    validation: &VoxtralLlamaCppValidation,
) -> String {
    format!(
        "Voxtral llama.cpp runtime is selected, but GGUF/mmproj files are not configured or missing. \
Set AMCP_VOXTRAL_LLAMA_MODEL_PATH and AMCP_VOXTRAL_LLAMA_MMPROJ_PATH, or cache them in Hugging Face Hub under AMCP_VOXTRAL_LLAMA_REPO_ID with optional AMCP_VOXTRAL_LLAMA_MODEL_FILE / AMCP_VOXTRAL_LLAMA_MMPROJ_FILE. \
model_path={:?} model_exists={} mmproj_path={:?} mmproj_exists={}",
        config.model_path, validation.model_exists, config.mmproj_path, validation.mmproj_exists
    )
}

#[cfg(any(
    feature = "voxtral-llamacpp-native",
    feature = "voxtral-llamacpp-realtime-patched",
    test
))]
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
    format!("{source}{previous}Original transcript: {transcript_text}\nTranslate the transcript into {target}. Return only the translated text.\nTranslation:")
}

#[cfg(any(
    feature = "voxtral-llamacpp-native",
    feature = "voxtral-llamacpp-realtime-patched"
))]
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

#[cfg(any(
    feature = "voxtral-llamacpp-native",
    feature = "voxtral-llamacpp-realtime-patched",
    test
))]
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

fn is_llamacpp_runtime(runtime: &str) -> bool {
    matches!(
        runtime.trim().to_ascii_lowercase().as_str(),
        "llamacpp" | "llama.cpp" | "llama-cpp" | "llama_cpp"
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LlamaCppGgufKind {
    TextModel,
    Mmproj,
}

fn resolve_hf_cached_gguf(
    repo_id: &str,
    file_name: Option<&str>,
    kind: LlamaCppGgufKind,
) -> Option<PathBuf> {
    let snapshot = existing_hf_snapshot_dir(repo_id)?;
    if let Some(file_name) = file_name {
        return find_hf_cached_file(&snapshot, HF_SNAPSHOT_SEARCH_DEPTH, &|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.eq_ignore_ascii_case(file_name))
        })
        .into_iter()
        .next();
    }

    let candidates = find_hf_cached_file(&snapshot, HF_SNAPSHOT_SEARCH_DEPTH, &|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| is_llamacpp_gguf_name(name, kind))
    });
    if candidates.len() == 1 {
        candidates.into_iter().next()
    } else {
        None
    }
}

fn find_hf_cached_file(
    root: &std::path::Path,
    max_depth: usize,
    predicate: &dyn Fn(&std::path::Path) -> bool,
) -> Vec<PathBuf> {
    let mut matches = Vec::new();
    collect_hf_cached_files(root, max_depth, predicate, &mut matches);
    matches.sort();
    matches
}

fn collect_hf_cached_files(
    root: &std::path::Path,
    max_depth: usize,
    predicate: &dyn Fn(&std::path::Path) -> bool,
    matches: &mut Vec<PathBuf>,
) {
    if max_depth == 0 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if path.is_file() {
            if predicate(&path) {
                matches.push(path);
            }
        } else if path.is_dir() {
            collect_hf_cached_files(&path, max_depth - 1, predicate, matches);
        }
    }
}

fn is_llamacpp_gguf_name(name: &str, kind: LlamaCppGgufKind) -> bool {
    let lower = name.to_ascii_lowercase();
    if !lower.ends_with(".gguf") {
        return false;
    }
    let is_mmproj = lower.contains("mmproj") || lower.contains("mm-project");
    match kind {
        LlamaCppGgufKind::TextModel => !is_mmproj,
        LlamaCppGgufKind::Mmproj => is_mmproj,
    }
}

fn existing_hf_snapshot_dir(repo_id: &str) -> Option<PathBuf> {
    let repo_dir = default_hf_repo_cache_dir(repo_id);
    let revision = std::fs::read_to_string(repo_dir.join("refs").join("main")).ok()?;
    let revision = revision.trim();
    if revision.is_empty() {
        return None;
    }
    let snapshot = repo_dir.join("snapshots").join(revision);
    snapshot.is_dir().then_some(snapshot)
}

fn default_hf_repo_cache_dir(repo_id: &str) -> PathBuf {
    default_hf_hub_cache_dir().join(format!("models--{}", repo_id.replace('/', "--")))
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

#[cfg(any(
    feature = "voxtral-llamacpp-native",
    feature = "voxtral-llamacpp-realtime-patched"
))]
fn clean_model_output(text: &str) -> String {
    text.replace("[/INST]", "")
        .replace("[INST]", "")
        .trim()
        .to_string()
}

#[cfg(any(
    feature = "voxtral-llamacpp-native",
    feature = "voxtral-llamacpp-realtime-patched"
))]
fn is_low_quality_translation(text: &str, target_language: &str) -> bool {
    let text = text.trim();
    if text.is_empty() || text.contains('\u{fffd}') {
        return true;
    }
    match target_language {
        "zh" => {
            if !text.chars().any(is_cjk_unified_ideograph) {
                return true;
            }
        }
        "ko" => {
            if !text.chars().any(is_hangul_syllable) {
                return true;
            }
        }
        _ => {}
    }
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 8 {
        return false;
    }
    let repeated_small_words = words
        .windows(4)
        .filter(|window| {
            let first = window[0].to_ascii_lowercase();
            first.len() <= 3
                && window
                    .iter()
                    .all(|word| word.eq_ignore_ascii_case(first.as_str()))
        })
        .count();
    repeated_small_words >= 2
}

#[cfg(any(
    feature = "voxtral-llamacpp-native",
    feature = "voxtral-llamacpp-realtime-patched"
))]
fn is_cjk_unified_ideograph(ch: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&ch)
}

#[cfg(any(
    feature = "voxtral-llamacpp-native",
    feature = "voxtral-llamacpp-realtime-patched"
))]
fn is_hangul_syllable(ch: char) -> bool {
    ('\u{ac00}'..='\u{d7af}').contains(&ch)
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

fn runtime_vulkan_available() -> bool {
    env_bool("AMCP_VOXTRAL_PATCHED_LLAMA_LINK_VULKAN").unwrap_or(false)
        || std::env::current_exe()
            .ok()
            .and_then(|path| path.parent().map(|dir| dir.join("ggml-vulkan.dll")))
            .is_some_and(|path| path.is_file())
        || env_path("AMCP_VOXTRAL_PATCHED_LLAMA_BIN_DIR")
            .map(|dir| dir.join("ggml-vulkan.dll").is_file())
            .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn routes_all_voxtral_requests_to_llamacpp() {
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
        assert!(should_route_to_llamacpp_with_runtime(
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
            repo_id: DEFAULT_LLAMA_REPO_ID.to_string(),
            model_file: None,
            mmproj_file: None,
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
    fn resolves_gguf_and_mmproj_from_hf_cache() {
        let _guard = env_lock().lock().unwrap();
        let cache_root =
            std::env::temp_dir().join(format!("amcp-voxtral-hf-cache-{}", std::process::id()));
        let repo_dir = cache_root.join("models--example--voxtral-gguf");
        let snapshot_dir = repo_dir.join("snapshots").join("abc123");
        let _ = std::fs::remove_dir_all(&cache_root);
        std::fs::create_dir_all(repo_dir.join("refs")).unwrap();
        std::fs::create_dir_all(&snapshot_dir).unwrap();
        std::fs::write(repo_dir.join("refs").join("main"), "abc123").unwrap();
        let model_path = snapshot_dir.join("voxtral-q4_k_m.gguf");
        let mmproj_path = snapshot_dir.join("mmproj-voxtral.gguf");
        std::fs::write(&model_path, b"model").unwrap();
        std::fs::write(&mmproj_path, b"mmproj").unwrap();

        std::env::set_var("HF_HUB_CACHE", &cache_root);
        std::env::remove_var("AMCP_VOXTRAL_LLAMA_MODEL_PATH");
        std::env::remove_var("AMCP_VOXTRAL_LLAMA_MMPROJ_PATH");
        std::env::remove_var("AMCP_VOXTRAL_LLAMA_MODEL_FILE");
        std::env::remove_var("AMCP_VOXTRAL_LLAMA_MMPROJ_FILE");
        std::env::set_var("AMCP_VOXTRAL_LLAMA_REPO_ID", "example/voxtral-gguf");
        let config = configure_voxtral_llamacpp(&[HardwareBackend::Cpu]);
        std::env::remove_var("HF_HUB_CACHE");
        std::env::remove_var("AMCP_VOXTRAL_LLAMA_REPO_ID");
        let _ = std::fs::remove_dir_all(&cache_root);

        assert_eq!(config.model_path.as_deref(), Some(model_path.as_path()));
        assert_eq!(config.mmproj_path.as_deref(), Some(mmproj_path.as_path()));
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
