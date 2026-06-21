use super::audio::PreprocessedAudio;
use super::TranscriptionOptions;
use crate::accelerator::HardwareBackend;
use crate::asr::qwen_ffi;
use crate::asr::voxtral_onnx;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendTranscription {
    pub text: String,
    pub language: Option<String>,
    pub chunks: Vec<serde_json::Value>,
}

pub fn try_transcribe_real(
    audio: &PreprocessedAudio,
    options: &TranscriptionOptions,
    available_backends: &[HardwareBackend],
) -> Result<Option<BackendTranscription>, String> {
    if options.model_id.starts_with("whisper") {
        return try_transcribe_whisper(audio, options);
    }
    if options.model_id.starts_with("qwen3-asr") {
        return try_transcribe_qwen(audio, options, available_backends);
    }
    if options.model_id.starts_with("voxtral") {
        return try_transcribe_voxtral(audio, available_backends);
    }

    Ok(None)
}

pub fn transcribe_placeholder(audio: &PreprocessedAudio, options: &TranscriptionOptions) -> String {
    let family_hint = if options.model_id.starts_with("qwen3-asr") {
        "Qwen3-ASR"
    } else if options.model_id.starts_with("voxtral") {
        "Voxtral"
    } else {
        "Whisper"
    };

    format!(
        "Recognized {duration:.2}s / {samples} samples with {family_hint} ({model}).",
        duration = audio.duration_seconds,
        samples = audio.samples.len(),
        model = options.model_id
    )
}

fn try_transcribe_qwen(
    audio: &PreprocessedAudio,
    options: &TranscriptionOptions,
    available_backends: &[HardwareBackend],
) -> Result<Option<BackendTranscription>, String> {
    let Some(result) = qwen_ffi::transcribe_qwen_audio(
        &audio.samples,
        options.language.as_deref(),
        options.previous_text.as_deref(),
        available_backends,
    )?
    else {
        return Ok(None);
    };

    Ok(Some(BackendTranscription {
        text: result.text,
        language: options.language.clone(),
        chunks: vec![serde_json::json!({
            "backend": "qwen-c",
            "model_dir": result.model_dir,
            "library_path": result.library_path,
            "sample_rate": audio.sample_rate,
            "duration_seconds": audio.duration_seconds,
        })],
    }))
}

fn try_transcribe_voxtral(
    audio: &PreprocessedAudio,
    available_backends: &[HardwareBackend],
) -> Result<Option<BackendTranscription>, String> {
    let Some(result) = voxtral_onnx::transcribe_voxtral_audio(audio, available_backends)? else {
        return Ok(None);
    };

    Ok(Some(BackendTranscription {
        text: result.text,
        language: None,
        chunks: vec![serde_json::json!({
            "backend": "voxtral-onnx",
            "audio_encoder_path": result.audio_encoder_path,
            "decoder_path": result.decoder_path,
            "audio_frames": result.audio_frames,
            "generated_tokens": result.generated_tokens,
            "sample_rate": audio.sample_rate,
            "duration_seconds": audio.duration_seconds,
        })],
    }))
}

#[cfg(feature = "whisper")]
fn try_transcribe_whisper(
    audio: &PreprocessedAudio,
    options: &TranscriptionOptions,
) -> Result<Option<BackendTranscription>, String> {
    let Some(model_path) = resolve_whisper_model_path(&options.model_id)? else {
        return Ok(None);
    };

    let ctx = whisper_rs::WhisperContext::new_with_params(
        &model_path,
        whisper_rs::WhisperContextParameters::default(),
    )
    .map_err(|error| format!("failed to load Whisper model at {model_path}: {error}"))?;
    let mut state = ctx
        .create_state()
        .map_err(|error| format!("failed to create Whisper state: {error}"))?;

    let beam_size = options.beam_size.unwrap_or(1).max(1) as i32;
    let strategy = if beam_size > 1 {
        whisper_rs::SamplingStrategy::BeamSearch {
            beam_size,
            patience: -1.0,
        }
    } else {
        whisper_rs::SamplingStrategy::Greedy { best_of: 1 }
    };
    let mut params = whisper_rs::FullParams::new(strategy);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_token_timestamps(options.return_timestamps);
    params.set_n_threads(
        std::thread::available_parallelism()
            .map(|threads| threads.get().min(8) as i32)
            .unwrap_or(4),
    );

    if let Some(language) = options
        .language
        .as_deref()
        .filter(|language| !matches!(*language, "" | "auto"))
    {
        params.set_language(Some(language));
    }

    state
        .full(params, &audio.samples)
        .map_err(|error| format!("Whisper transcription failed: {error}"))?;

    let mut chunks = Vec::new();
    let mut parts = Vec::new();
    for segment in state.as_iter() {
        let text = segment.to_string().trim().to_string();
        if text.is_empty() {
            continue;
        }
        chunks.push(serde_json::json!({
            "start": segment.start_timestamp() as f64 / 100.0,
            "end": segment.end_timestamp() as f64 / 100.0,
            "text": text,
        }));
        parts.push(text);
    }

    Ok(Some(BackendTranscription {
        text: parts.join(" ").trim().to_string(),
        language: options.language.clone(),
        chunks,
    }))
}

#[cfg(not(feature = "whisper"))]
fn try_transcribe_whisper(
    _audio: &PreprocessedAudio,
    _options: &TranscriptionOptions,
) -> Result<Option<BackendTranscription>, String> {
    Ok(None)
}

#[cfg(feature = "whisper")]
fn resolve_whisper_model_path(model_id: &str) -> Result<Option<String>, String> {
    let suffix = model_id
        .trim_start_matches("whisper-")
        .replace('-', "_")
        .to_ascii_uppercase();
    let specific_key = format!("AMCP_WHISPER_{suffix}_MODEL_PATH");

    if let Some(path) = std::env::var(&specific_key)
        .ok()
        .or_else(|| std::env::var("AMCP_WHISPER_MODEL_PATH").ok())
        .filter(|path| !path.trim().is_empty())
    {
        return Ok(Some(path));
    }

    let Some((file_name, url)) = whisper_model_download(model_id) else {
        return Ok(None);
    };
    let model_dir = std::env::var("AMCP_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::current_dir()
                .unwrap_or_else(|_| std::path::PathBuf::from("."))
                .join("models")
                .join("whisper")
        });
    let model_path = model_dir.join(file_name);
    if model_path.is_file() {
        return Ok(Some(model_path.to_string_lossy().to_string()));
    }

    std::fs::create_dir_all(&model_dir)
        .map_err(|error| format!("failed to create model cache {model_dir:?}: {error}"))?;
    download_file(url, &model_path)?;
    Ok(Some(model_path.to_string_lossy().to_string()))
}

#[cfg(feature = "whisper")]
fn whisper_model_download(model_id: &str) -> Option<(&'static str, &'static str)> {
    match model_id {
        "whisper-tiny" => Some((
            "ggml-tiny.bin",
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
        )),
        "whisper-small" => Some((
            "ggml-small.bin",
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
        )),
        "whisper-medium" => Some((
            "ggml-medium.bin",
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
        )),
        "whisper-large-v3-turbo" => Some((
            "ggml-large-v3-turbo.bin",
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
        )),
        _ => None,
    }
}

#[cfg(feature = "whisper")]
fn download_file(url: &str, path: &std::path::Path) -> Result<(), String> {
    let tmp_path = path.with_extension("download");
    let mut response = reqwest::blocking::get(url)
        .map_err(|error| format!("failed to download model from {url}: {error}"))?
        .error_for_status()
        .map_err(|error| format!("model download failed for {url}: {error}"))?;
    let mut output = std::fs::File::create(&tmp_path)
        .map_err(|error| format!("failed to create temporary model file {tmp_path:?}: {error}"))?;
    std::io::copy(&mut response, &mut output)
        .map_err(|error| format!("failed to write model file {tmp_path:?}: {error}"))?;
    std::fs::rename(&tmp_path, path)
        .map_err(|error| format!("failed to move model into cache {path:?}: {error}"))?;
    Ok(())
}

#[cfg(feature = "whisper")]
pub mod whisper_rs_backend {
    //! Integration point for `whisper-rs` / whisper.cpp.
    //!
    //! Build flags are expected to enable the best backend per platform:
    //! CUDA on NVIDIA Windows/Linux, Metal on Apple Silicon, and Vulkan where
    //! available. The safe wrapper should implement the same trait boundary as
    //! the placeholder manager in `asr::mod`.
}

#[cfg(feature = "qwen")]
pub mod qwen_c_backend {
    //! Integration point for antirez/qwen-asr through C FFI.
    //!
    //! The build script should link Accelerate on macOS, OpenBLAS on Linux, and
    //! a bundled or user-provided BLAS on Windows, then expose a safe Rust
    //! context that honors the single-loaded-model invariant.
}

#[cfg(feature = "voxtral")]
pub mod voxtral_onnx_backend {
    //! Integration point for ONNX Runtime via `ort`.
    //!
    //! Provider priority should prefer CUDA on Windows/Linux and CoreML on
    //! Apple Silicon, with CPU execution provider as the final fallback.
}
