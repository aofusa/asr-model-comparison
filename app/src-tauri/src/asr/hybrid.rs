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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendAssetProgress {
    pub phase: String,
    pub message: String,
    pub progress: Option<u8>,
    pub bytes_downloaded: Option<u64>,
    pub total_bytes: Option<u64>,
}

pub fn prepare_real_model_assets(
    options: &TranscriptionOptions,
    available_backends: &[HardwareBackend],
) -> Result<Vec<BackendAssetProgress>, String> {
    let mut events = if options.model_id.starts_with("whisper") {
        prepare_whisper_assets(&options.model_id)?
    } else if options.model_id.starts_with("qwen3-asr") {
        prepare_qwen_assets(available_backends)?
    } else if options.model_id.starts_with("voxtral") {
        prepare_voxtral_assets(available_backends)?
    } else {
        Vec::new()
    };

    if options
        .target_language
        .as_deref()
        .is_some_and(|language| !matches!(language, "" | "none" | "auto"))
    {
        events.extend(prepare_translation_assets());
    }

    Ok(events)
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

#[cfg(feature = "whisper")]
fn prepare_whisper_assets(model_id: &str) -> Result<Vec<BackendAssetProgress>, String> {
    let mut events = Vec::new();
    let _ = resolve_whisper_model_path_with_progress(model_id, |event| events.push(event))?;
    Ok(events)
}

#[cfg(not(feature = "whisper"))]
fn prepare_whisper_assets(_model_id: &str) -> Result<Vec<BackendAssetProgress>, String> {
    Ok(Vec::new())
}

fn prepare_translation_assets() -> Vec<BackendAssetProgress> {
    let status = crate::translation::runtime_status();
    vec![BackendAssetProgress {
        phase: "translation_runner".to_string(),
        message: if status.configured {
            format!(
                "Translation runner configured: {} {}.",
                status.command.unwrap_or_default(),
                status.args.join(" ")
            )
        } else {
            status.reason
        },
        progress: Some(58),
        bytes_downloaded: None,
        total_bytes: None,
    }]
}

#[cfg(feature = "qwen")]
fn prepare_qwen_assets(
    available_backends: &[HardwareBackend],
) -> Result<Vec<BackendAssetProgress>, String> {
    let mut events = vec![BackendAssetProgress {
        phase: "qwen_paths".to_string(),
        message: "Resolving Qwen ASR library and model directory.".to_string(),
        progress: Some(48),
        bytes_downloaded: None,
        total_bytes: None,
    }];

    let Some(config) = qwen_ffi::configure_qwen_ffi(available_backends) else {
        events.push(BackendAssetProgress {
            phase: "qwen_missing".to_string(),
            message: "Qwen is not fully configured; set AMCP_QWEN_ASR_LIB or AMCP_QWEN_ASR_DIR plus AMCP_QWEN_MODEL_DIR.".to_string(),
            progress: Some(50),
            bytes_downloaded: None,
            total_bytes: None,
        });
        return Ok(events);
    };

    events.push(BackendAssetProgress {
        phase: "qwen_library".to_string(),
        message: format!(
            "Checking Qwen ASR library: {}.",
            config.library_path.display()
        ),
        progress: Some(52),
        bytes_downloaded: None,
        total_bytes: None,
    });
    events.push(BackendAssetProgress {
        phase: "qwen_model_dir".to_string(),
        message: format!(
            "Checking Qwen model directory: {}.",
            config.model_dir.display()
        ),
        progress: Some(54),
        bytes_downloaded: None,
        total_bytes: None,
    });

    match qwen_ffi::validate_qwen_ffi(&config, false) {
        Ok(_) => events.push(BackendAssetProgress {
            phase: "qwen_symbols".to_string(),
            message: "Qwen DLL loaded and required symbols were found.".to_string(),
            progress: Some(56),
            bytes_downloaded: None,
            total_bytes: None,
        }),
        Err(error) => events.push(BackendAssetProgress {
            phase: "qwen_validation_error".to_string(),
            message: error,
            progress: Some(56),
            bytes_downloaded: None,
            total_bytes: None,
        }),
    }

    Ok(events)
}

#[cfg(not(feature = "qwen"))]
fn prepare_qwen_assets(
    _available_backends: &[HardwareBackend],
) -> Result<Vec<BackendAssetProgress>, String> {
    Ok(Vec::new())
}

#[cfg(feature = "voxtral")]
fn prepare_voxtral_assets(
    available_backends: &[HardwareBackend],
) -> Result<Vec<BackendAssetProgress>, String> {
    let mut events = vec![BackendAssetProgress {
        phase: "voxtral_paths".to_string(),
        message: "Resolving Voxtral ONNX split model files.".to_string(),
        progress: Some(48),
        bytes_downloaded: None,
        total_bytes: None,
    }];

    let Some(config) = voxtral_onnx::configure_voxtral_onnx(available_backends) else {
        events.push(BackendAssetProgress {
            phase: "voxtral_missing".to_string(),
            message: "Voxtral is not configured; set AMCP_VOXTRAL_MODEL_DIR or split ONNX path variables.".to_string(),
            progress: Some(50),
            bytes_downloaded: None,
            total_bytes: None,
        });
        return Ok(events);
    };

    for (phase, path) in [
        ("voxtral_audio_encoder", config.audio_encoder_path.as_ref()),
        ("voxtral_embed_tokens", config.embed_tokens_path.as_ref()),
        ("voxtral_decoder", config.decoder_path.as_ref()),
        ("voxtral_tokenizer", config.tokenizer_path.as_ref()),
    ] {
        events.push(BackendAssetProgress {
            phase: phase.to_string(),
            message: path
                .map(|path| format!("Checking {phase}: {}.", path.display()))
                .unwrap_or_else(|| format!("{phase} is not configured.")),
            progress: Some(52),
            bytes_downloaded: None,
            total_bytes: None,
        });
    }

    events.push(BackendAssetProgress {
        phase: "voxtral_session_metadata".to_string(),
        message: "Loading Voxtral ONNX sessions to verify input/output metadata.".to_string(),
        progress: Some(56),
        bytes_downloaded: None,
        total_bytes: None,
    });
    match voxtral_onnx::validate_voxtral_session(&config) {
        Ok(info) => events.push(BackendAssetProgress {
            phase: "voxtral_session_ready".to_string(),
            message: format!(
                "Voxtral ONNX sessions validated: {} inputs, {} outputs.",
                info.inputs.len(),
                info.outputs.len()
            ),
            progress: Some(58),
            bytes_downloaded: None,
            total_bytes: None,
        }),
        Err(error) => events.push(BackendAssetProgress {
            phase: "voxtral_validation_error".to_string(),
            message: error,
            progress: Some(58),
            bytes_downloaded: None,
            total_bytes: None,
        }),
    }

    Ok(events)
}

#[cfg(not(feature = "voxtral"))]
fn prepare_voxtral_assets(
    _available_backends: &[HardwareBackend],
) -> Result<Vec<BackendAssetProgress>, String> {
    Ok(Vec::new())
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
    resolve_whisper_model_path_with_progress(model_id, |_| {})
}

#[cfg(feature = "whisper")]
fn resolve_whisper_model_path_with_progress(
    model_id: &str,
    mut progress: impl FnMut(BackendAssetProgress),
) -> Result<Option<String>, String> {
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
    download_file(url, &model_path, &mut progress)?;
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
fn download_file(
    url: &str,
    path: &std::path::Path,
    progress: &mut impl FnMut(BackendAssetProgress),
) -> Result<(), String> {
    let url = url.to_string();
    let path = path.to_path_buf();
    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn({
        let url = url.clone();
        let path = path.clone();
        move || download_file_blocking(&url, &path, tx)
    });

    for event in rx {
        progress(event);
    }

    handle
        .join()
        .map_err(|_| format!("model download thread panicked for {url}"))?
}

#[cfg(feature = "whisper")]
fn download_file_blocking(
    url: &str,
    path: &std::path::Path,
    progress: std::sync::mpsc::Sender<BackendAssetProgress>,
) -> Result<(), String> {
    use std::io::Read;

    let tmp_path = path.with_extension("download");
    let mut response = reqwest::blocking::get(url)
        .map_err(|error| format!("failed to download model from {url}: {error}"))?
        .error_for_status()
        .map_err(|error| format!("model download failed for {url}: {error}"))?;
    let total_bytes = response.content_length();
    let mut output = std::fs::File::create(&tmp_path)
        .map_err(|error| format!("failed to create temporary model file {tmp_path:?}: {error}"))?;
    let mut downloaded = 0_u64;
    let mut last_reported = 0_u8;
    let mut buffer = [0_u8; 256 * 1024];
    let _ = progress.send(BackendAssetProgress {
        phase: "downloading".to_string(),
        message: format!("Downloading model from {url}."),
        progress: Some(0),
        bytes_downloaded: Some(0),
        total_bytes,
    });

    loop {
        let read = response
            .read(&mut buffer)
            .map_err(|error| format!("failed to read model download stream: {error}"))?;
        if read == 0 {
            break;
        }
        std::io::Write::write_all(&mut output, &buffer[..read])
            .map_err(|error| format!("failed to write model file {tmp_path:?}: {error}"))?;
        downloaded += read as u64;
        let percent = total_bytes
            .map(|total| ((downloaded.saturating_mul(100)) / total.max(1)).min(100) as u8)
            .unwrap_or(0);
        if percent >= last_reported.saturating_add(5) || total_bytes.is_none() {
            last_reported = percent;
            let _ = progress.send(BackendAssetProgress {
                phase: "downloading".to_string(),
                message: format!(
                    "Downloaded {}{}.",
                    human_bytes(downloaded),
                    total_bytes
                        .map(|total| format!(" / {}", human_bytes(total)))
                        .unwrap_or_default()
                ),
                progress: total_bytes.map(|_| percent),
                bytes_downloaded: Some(downloaded),
                total_bytes,
            });
        }
    }
    std::fs::rename(&tmp_path, path)
        .map_err(|error| format!("failed to move model into cache {path:?}: {error}"))?;
    let _ = progress.send(BackendAssetProgress {
        phase: "downloaded".to_string(),
        message: format!("Model download complete: {}.", path.display()),
        progress: Some(100),
        bytes_downloaded: Some(downloaded),
        total_bytes: total_bytes.or(Some(downloaded)),
    });
    Ok(())
}

#[cfg(feature = "whisper")]
fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut value = bytes as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} {}", bytes, UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
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
