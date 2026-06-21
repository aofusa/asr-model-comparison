use crate::accelerator::{HardwareBackend, ModelFamily};
use crate::asr::qwen_candle;
use crate::asr::voxtral_onnx;
use crate::models::{available_models, family_for_model};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeBackendKind {
    WhisperRs,
    QwenNative,
    VoxtralOnnx,
    Placeholder,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeBackendStatus {
    pub model_id: String,
    pub family: ModelFamily,
    pub backend: RuntimeBackendKind,
    pub real_inference_available: bool,
    pub configured: bool,
    pub selected_accelerators: Vec<HardwareBackend>,
    pub artifacts: Vec<RuntimeArtifactStatus>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeArtifactStatus {
    pub name: String,
    pub kind: RuntimeArtifactKind,
    pub path: Option<String>,
    pub env_var: Option<String>,
    pub required: bool,
    pub exists: bool,
    pub note: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeArtifactKind {
    File,
    Directory,
    Command,
    Download,
}

pub fn runtime_backend_statuses(
    available_backends: &[HardwareBackend],
) -> Vec<RuntimeBackendStatus> {
    available_models()
        .into_iter()
        .map(|model| runtime_backend_status(model.id, available_backends))
        .collect()
}

pub fn runtime_backend_status(
    model_id: &str,
    available_backends: &[HardwareBackend],
) -> RuntimeBackendStatus {
    let family = family_for_model(model_id).unwrap_or(ModelFamily::Whisper);
    match family {
        ModelFamily::Whisper => whisper_status(model_id, family, available_backends),
        ModelFamily::Qwen3 => qwen_status(model_id, family, available_backends),
        ModelFamily::Voxtral => voxtral_status(model_id, family, available_backends),
    }
}

pub fn runtime_backend_kind(
    model_id: &str,
    available_backends: &[HardwareBackend],
) -> RuntimeBackendKind {
    runtime_backend_status(model_id, available_backends).backend
}

fn whisper_status(
    model_id: &str,
    family: ModelFamily,
    available_backends: &[HardwareBackend],
) -> RuntimeBackendStatus {
    let configured = cfg!(feature = "whisper");
    let artifacts = whisper_artifacts(model_id);
    let real_inference_available = configured;
    RuntimeBackendStatus {
        model_id: model_id.to_string(),
        family,
        backend: if configured {
            RuntimeBackendKind::WhisperRs
        } else {
            RuntimeBackendKind::Placeholder
        },
        real_inference_available,
        configured,
        selected_accelerators: filter_backends(
            available_backends,
            &[
                HardwareBackend::Cuda,
                HardwareBackend::Vulkan,
                HardwareBackend::Metal,
                HardwareBackend::Cpu,
            ],
        ),
        artifacts,
        reason: if configured {
            "whisper-rs is compiled in; model path or auto-download is resolved at load time."
                .to_string()
        } else {
            "whisper-rs feature is not enabled; using placeholder inference.".to_string()
        },
    }
}

fn qwen_status(
    model_id: &str,
    family: ModelFamily,
    available_backends: &[HardwareBackend],
) -> RuntimeBackendStatus {
    let configured = cfg!(feature = "qwen");
    let candle_config = qwen_candle::configure_qwen_candle(model_id, available_backends);
    let artifacts = qwen_artifacts(&candle_config);
    let model_files_available = qwen_model_files_available(&candle_config.model_dir);
    let real_inference_available =
        configured && (model_files_available || candle_config.auto_download);
    RuntimeBackendStatus {
        model_id: model_id.to_string(),
        family,
        backend: if real_inference_available {
            RuntimeBackendKind::QwenNative
        } else {
            RuntimeBackendKind::Placeholder
        },
        real_inference_available,
        configured,
        selected_accelerators: candle_config.backends,
        artifacts,
        reason: if real_inference_available {
            "qwen feature is enabled and Qwen Candle model files are present or can be downloaded."
                .to_string()
        } else if configured {
            "qwen feature is enabled, but Qwen Candle model files are missing and auto-download is disabled."
                .to_string()
        } else {
            "qwen feature is not enabled; using placeholder inference.".to_string()
        },
    }
}

fn voxtral_status(
    model_id: &str,
    family: ModelFamily,
    available_backends: &[HardwareBackend],
) -> RuntimeBackendStatus {
    let configured = cfg!(feature = "voxtral");
    let onnx_config = voxtral_onnx::configure_voxtral_onnx(available_backends);
    let artifacts = voxtral_artifacts(onnx_config.as_ref());
    let split_files_available = onnx_config
        .as_ref()
        .map(|config| {
            [
                config.audio_encoder_path.as_ref(),
                config.embed_tokens_path.as_ref(),
                config.decoder_path.as_ref(),
                config.tokenizer_path.as_ref(),
            ]
            .into_iter()
            .all(|path| path.map(|path| path.is_file()).unwrap_or(false))
        })
        .unwrap_or(false);
    let real_inference_available = configured && split_files_available;
    RuntimeBackendStatus {
        model_id: model_id.to_string(),
        family,
        backend: if real_inference_available {
            RuntimeBackendKind::VoxtralOnnx
        } else {
            RuntimeBackendKind::Placeholder
        },
        real_inference_available,
        configured,
        selected_accelerators: onnx_config
            .as_ref()
            .map(|config| config.providers.clone())
            .unwrap_or_else(|| {
                filter_backends(
                    available_backends,
                    &[
                        HardwareBackend::Cuda,
                        HardwareBackend::DirectMl,
                        HardwareBackend::CoreMl,
                        HardwareBackend::NnApi,
                        HardwareBackend::Vulkan,
                        HardwareBackend::Cpu,
                    ],
                )
            }),
        artifacts,
        reason: if real_inference_available {
            "voxtral feature is enabled and Voxtral split ONNX/tokenizer files are configured."
                .to_string()
        } else if configured {
            "voxtral feature is enabled, but audio_encoder.onnx, embed_tokens.onnx, decoder_model_merged.onnx, and tokenizer.json are not fully configured."
                .to_string()
        } else {
            "voxtral feature is not enabled; using placeholder inference.".to_string()
        },
    }
}

fn whisper_artifacts(model_id: &str) -> Vec<RuntimeArtifactStatus> {
    let suffix = model_id
        .trim_start_matches("whisper-")
        .replace('-', "_")
        .to_ascii_uppercase();
    let specific_key = format!("AMCP_WHISPER_{suffix}_MODEL_PATH");
    if let Some(path) = std::env::var(&specific_key)
        .ok()
        .filter(|path| !path.trim().is_empty())
        .or_else(|| std::env::var("AMCP_WHISPER_MODEL_PATH").ok())
        .filter(|path| !path.trim().is_empty())
    {
        let exists = std::path::Path::new(&path).is_file();
        return vec![RuntimeArtifactStatus {
            name: "whisper_model".to_string(),
            kind: RuntimeArtifactKind::File,
            path: Some(path),
            env_var: Some(specific_key),
            required: false,
            exists,
            note: Some("explicit Whisper model path".to_string()),
        }];
    }

    let model_dir = std::env::var("AMCP_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::current_dir()
                .unwrap_or_else(|_| std::path::PathBuf::from("."))
                .join("models")
                .join("whisper")
        });
    let Some((file_name, url)) = whisper_model_download(model_id) else {
        return vec![RuntimeArtifactStatus {
            name: "whisper_model".to_string(),
            kind: RuntimeArtifactKind::Download,
            path: None,
            env_var: Some("AMCP_WHISPER_MODEL_PATH".to_string()),
            required: false,
            exists: false,
            note: Some("no built-in download URL is known for this Whisper model".to_string()),
        }];
    };
    let model_path = model_dir.join(file_name);
    vec![RuntimeArtifactStatus {
        name: "whisper_model".to_string(),
        kind: RuntimeArtifactKind::Download,
        path: Some(model_path.to_string_lossy().to_string()),
        env_var: Some("AMCP_MODEL_DIR".to_string()),
        required: false,
        exists: model_path.is_file(),
        note: Some(format!("auto-download source: {url}")),
    }]
}

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

fn qwen_artifacts(config: &qwen_candle::QwenCandleConfig) -> Vec<RuntimeArtifactStatus> {
    vec![
        dir_artifact(
            "qwen_model_dir",
            &config.model_dir,
            Some("AMCP_QWEN_MODEL_DIR"),
            false,
        ),
        file_artifact(
            "qwen_config",
            &config.model_dir.join("config.json"),
            Some("AMCP_QWEN_MODEL_DIR"),
            false,
        ),
        file_artifact(
            "qwen_tokenizer",
            &config.model_dir.join("tokenizer.json"),
            Some("AMCP_QWEN_MODEL_DIR"),
            false,
        ),
        RuntimeArtifactStatus {
            name: "qwen_weights".to_string(),
            kind: RuntimeArtifactKind::File,
            path: Some(
                config
                    .model_dir
                    .join("model.safetensors")
                    .to_string_lossy()
                    .to_string(),
            ),
            env_var: Some("AMCP_QWEN_MODEL_DIR".to_string()),
            required: false,
            exists: qwen_model_weights_available(&config.model_dir),
            note: Some(
                "single model.safetensors, sharded index, or any *.safetensors file".to_string(),
            ),
        },
    ]
}

fn qwen_model_files_available(model_dir: &std::path::Path) -> bool {
    model_dir.join("config.json").is_file()
        && model_dir.join("tokenizer.json").is_file()
        && qwen_model_weights_available(model_dir)
}

fn qwen_model_weights_available(model_dir: &std::path::Path) -> bool {
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

fn voxtral_artifacts(
    config: Option<&voxtral_onnx::VoxtralOnnxConfig>,
) -> Vec<RuntimeArtifactStatus> {
    let Some(config) = config else {
        return vec![missing_env(
            "voxtral_model_dir",
            RuntimeArtifactKind::Directory,
            "AMCP_VOXTRAL_MODEL_DIR",
        )];
    };
    vec![
        optional_file_artifact(
            "voxtral_audio_encoder",
            config.audio_encoder_path.as_ref(),
            Some("AMCP_VOXTRAL_AUDIO_ENCODER_PATH"),
            true,
        ),
        optional_file_artifact(
            "voxtral_embed_tokens",
            config.embed_tokens_path.as_ref(),
            Some("AMCP_VOXTRAL_EMBED_TOKENS_PATH"),
            true,
        ),
        optional_file_artifact(
            "voxtral_decoder",
            config.decoder_path.as_ref(),
            Some("AMCP_VOXTRAL_DECODER_PATH"),
            true,
        ),
        optional_file_artifact(
            "voxtral_tokenizer",
            config.tokenizer_path.as_ref(),
            Some("AMCP_VOXTRAL_TOKENIZER_PATH"),
            true,
        ),
    ]
}

fn file_artifact(
    name: &str,
    path: &std::path::Path,
    env_var: Option<&str>,
    required: bool,
) -> RuntimeArtifactStatus {
    RuntimeArtifactStatus {
        name: name.to_string(),
        kind: RuntimeArtifactKind::File,
        path: Some(path.to_string_lossy().to_string()),
        env_var: env_var.map(ToString::to_string),
        required,
        exists: path.is_file(),
        note: None,
    }
}

fn dir_artifact(
    name: &str,
    path: &std::path::Path,
    env_var: Option<&str>,
    required: bool,
) -> RuntimeArtifactStatus {
    RuntimeArtifactStatus {
        name: name.to_string(),
        kind: RuntimeArtifactKind::Directory,
        path: Some(path.to_string_lossy().to_string()),
        env_var: env_var.map(ToString::to_string),
        required,
        exists: path.is_dir(),
        note: None,
    }
}

fn optional_file_artifact(
    name: &str,
    path: Option<&std::path::PathBuf>,
    env_var: Option<&str>,
    required: bool,
) -> RuntimeArtifactStatus {
    match path {
        Some(path) => file_artifact(name, path, env_var, required),
        None => RuntimeArtifactStatus {
            name: name.to_string(),
            kind: RuntimeArtifactKind::File,
            path: None,
            env_var: env_var.map(ToString::to_string),
            required,
            exists: false,
            note: Some("path is not configured".to_string()),
        },
    }
}

fn missing_env(name: &str, kind: RuntimeArtifactKind, env_var: &str) -> RuntimeArtifactStatus {
    RuntimeArtifactStatus {
        name: name.to_string(),
        kind,
        path: None,
        env_var: Some(env_var.to_string()),
        required: true,
        exists: false,
        note: Some("environment variable is not configured".to_string()),
    }
}

fn filter_backends(
    available_backends: &[HardwareBackend],
    preferred: &[HardwareBackend],
) -> Vec<HardwareBackend> {
    let mut filtered: Vec<_> = preferred
        .iter()
        .copied()
        .filter(|backend| available_backends.contains(backend))
        .collect();
    if !filtered.contains(&HardwareBackend::Cpu) {
        filtered.push(HardwareBackend::Cpu);
    }
    filtered
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_placeholder_for_unconfigured_qwen() {
        let status = runtime_backend_status(
            "qwen3-asr-0.6b",
            &[HardwareBackend::Cuda, HardwareBackend::Cpu],
        );

        assert_eq!(status.family, ModelFamily::Qwen3);
        assert_eq!(status.backend, RuntimeBackendKind::Placeholder);
        assert!(status.selected_accelerators.contains(&HardwareBackend::Cpu));
        assert!(status
            .artifacts
            .iter()
            .any(|artifact| artifact.env_var.as_deref() == Some("AMCP_QWEN_MODEL_DIR")));
        assert!(status
            .artifacts
            .iter()
            .any(|artifact| artifact.name == "qwen_weights"));
    }

    #[test]
    fn reports_whisper_runtime_boundary() {
        let status = runtime_backend_status(
            "whisper-tiny",
            &[HardwareBackend::Vulkan, HardwareBackend::Cpu],
        );

        assert_eq!(status.family, ModelFamily::Whisper);
        assert!(status
            .selected_accelerators
            .contains(&HardwareBackend::Vulkan));
        assert!(status.selected_accelerators.contains(&HardwareBackend::Cpu));
        assert!(status
            .artifacts
            .iter()
            .any(|artifact| artifact.name == "whisper_model"));
    }

    #[test]
    fn reports_voxtral_artifact_requirements() {
        let status = runtime_backend_status("voxtral-mini-4b", &[HardwareBackend::Cpu]);

        assert_eq!(status.family, ModelFamily::Voxtral);
        assert!(status
            .artifacts
            .iter()
            .any(|artifact| artifact.name == "voxtral_audio_encoder"
                || artifact.env_var.as_deref() == Some("AMCP_VOXTRAL_MODEL_DIR")));
        assert!(status.artifacts.iter().any(|artifact| artifact.required));
    }
}
