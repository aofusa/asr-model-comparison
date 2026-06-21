use super::audio::PreprocessedAudio;
use super::TranscriptionOptions;

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
