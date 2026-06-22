use crate::accelerator::{HardwareBackend, ModelFamily};
use crate::asr::qwen_candle;
use crate::asr::voxtral_llamacpp;
use crate::models::{available_models, family_for_model};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeBackendKind {
    WhisperRs,
    QwenNative,
    VoxtralLlamaCpp,
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
    let mut selected_accelerators = Vec::new();
    if cfg!(feature = "whisper-cuda") {
        selected_accelerators.push(HardwareBackend::Cuda);
    }
    if cfg!(feature = "whisper-metal") {
        selected_accelerators.push(HardwareBackend::Metal);
    }
    if cfg!(feature = "whisper-vulkan") {
        selected_accelerators.push(HardwareBackend::Vulkan);
    }
    selected_accelerators.push(HardwareBackend::Cpu);
    let selected_accelerators = filter_backends(available_backends, &selected_accelerators);
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
        selected_accelerators,
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
    voxtral_llamacpp_status(model_id, family, available_backends)
}

fn voxtral_llamacpp_status(
    model_id: &str,
    family: ModelFamily,
    available_backends: &[HardwareBackend],
) -> RuntimeBackendStatus {
    let configured = cfg!(any(
        feature = "voxtral-llamacpp-native",
        feature = "voxtral-llamacpp-realtime-patched"
    ));
    let config = voxtral_llamacpp::configure_voxtral_llamacpp(available_backends);
    let validation = voxtral_llamacpp::validate_voxtral_llamacpp_config(&config);
    let real_inference_available = configured && validation.configured;
    RuntimeBackendStatus {
        model_id: model_id.to_string(),
        family,
        backend: if real_inference_available {
            RuntimeBackendKind::VoxtralLlamaCpp
        } else {
            RuntimeBackendKind::Placeholder
        },
        real_inference_available,
        configured,
        selected_accelerators: config.providers.clone(),
        artifacts: voxtral_llamacpp_artifacts(&config),
        reason: if real_inference_available {
            format!(
                "voxtral-llamacpp feature is enabled and Voxtral GGUF/mmproj files are configured for {}. Metal acceleration is {}. Vulkan acceleration is {}. Patched Voxtral Realtime bridge is {}.",
                config.model_id,
                if cfg!(feature = "voxtral-realtime-metal") {
                    "compiled in"
                } else {
                    "not compiled in"
                },
                if cfg!(any(
                    feature = "voxtral-llamacpp-vulkan",
                    feature = "voxtral-realtime-vulkan"
                )) {
                    "compiled in"
                } else {
                    "not compiled in"
                },
                if cfg!(feature = "voxtral-llamacpp-realtime-patched") {
                    "compiled in"
                } else {
                    "not compiled in"
                }
            )
        } else if configured && cfg!(feature = "voxtral-llamacpp-realtime-patched") {
            "voxtral-llamacpp-realtime-patched feature is enabled, but AMCP_VOXTRAL_LLAMA_MODEL_PATH and AMCP_VOXTRAL_LLAMA_MMPROJ_PATH are not fully configured."
                .to_string()
        } else if configured
            && cfg!(any(
                feature = "voxtral-llamacpp-vulkan",
                feature = "voxtral-realtime-vulkan"
            ))
        {
            "voxtral-llamacpp-vulkan feature is enabled, but AMCP_VOXTRAL_LLAMA_MODEL_PATH and AMCP_VOXTRAL_LLAMA_MMPROJ_PATH are not fully configured."
                .to_string()
        } else if configured {
            "voxtral-llamacpp feature is enabled, but AMCP_VOXTRAL_LLAMA_MODEL_PATH and AMCP_VOXTRAL_LLAMA_MMPROJ_PATH are not fully configured. Enable voxtral-llamacpp-vulkan for Vulkan acceleration."
                .to_string()
        } else {
            "voxtral-llamacpp-native feature is not enabled; using placeholder inference."
                .to_string()
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
    let model_path = std::env::var("AMCP_MODEL_DIR")
        .ok()
        .filter(|path| !path.trim().is_empty())
        .map(std::path::PathBuf::from)
        .map(|model_dir| model_dir.join(file_name))
        .unwrap_or_else(|| hf_cached_file_path("ggerganov/whisper.cpp", file_name));
    vec![RuntimeArtifactStatus {
        name: "whisper_model".to_string(),
        kind: RuntimeArtifactKind::Download,
        path: Some(model_path.to_string_lossy().to_string()),
        env_var: Some("AMCP_WHISPER_MODEL_PATH".to_string()),
        required: false,
        exists: model_path.is_file(),
        note: Some(format!("Hugging Face Hub cache source: {url}")),
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
            "qwen_tokenizer_source",
            &config.model_dir.join("tokenizer_config.json"),
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
        && qwen_tokenizer_available(model_dir)
        && qwen_model_weights_available(model_dir)
}

fn qwen_tokenizer_available(model_dir: &std::path::Path) -> bool {
    model_dir.join("tokenizer.json").is_file()
        || (model_dir.join("tokenizer_config.json").is_file()
            && model_dir.join("vocab.json").is_file()
            && model_dir.join("merges.txt").is_file())
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

fn hf_cached_file_path(repo_id: &str, file_name: &str) -> std::path::PathBuf {
    if let Some(snapshot) = existing_hf_snapshot_dir(repo_id) {
        return snapshot.join(file_name);
    }
    default_hf_repo_cache_dir(repo_id).join(file_name)
}

fn existing_hf_snapshot_dir(repo_id: &str) -> Option<std::path::PathBuf> {
    let repo_dir = default_hf_repo_cache_dir(repo_id);
    let revision = std::fs::read_to_string(repo_dir.join("refs").join("main")).ok()?;
    let revision = revision.trim();
    if revision.is_empty() {
        return None;
    }
    let snapshot = repo_dir.join("snapshots").join(revision);
    snapshot.is_dir().then_some(snapshot)
}

fn default_hf_repo_cache_dir(repo_id: &str) -> std::path::PathBuf {
    default_hf_hub_cache_dir().join(format!("models--{}", repo_id.replace('/', "--")))
}

fn default_hf_hub_cache_dir() -> std::path::PathBuf {
    if let Ok(cache) = std::env::var("HF_HUB_CACHE") {
        let cache = cache.trim();
        if !cache.is_empty() {
            return std::path::PathBuf::from(cache);
        }
    }
    let hf_home = std::env::var("HF_HOME")
        .ok()
        .filter(|path| !path.trim().is_empty())
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var("XDG_CACHE_HOME")
                .ok()
                .filter(|path| !path.trim().is_empty())
                .map(std::path::PathBuf::from)
                .unwrap_or_else(default_user_cache_dir)
                .join("huggingface")
        });
    hf_home.join("hub")
}

fn default_user_cache_dir() -> std::path::PathBuf {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .join(".cache")
}

fn voxtral_llamacpp_artifacts(
    config: &voxtral_llamacpp::VoxtralLlamaCppConfig,
) -> Vec<RuntimeArtifactStatus> {
    vec![
        RuntimeArtifactStatus {
            name: "voxtral_runtime".to_string(),
            kind: RuntimeArtifactKind::Command,
            path: voxtral_llamacpp::voxtral_runtime_preference(),
            env_var: Some("AMCP_VOXTRAL_RUNTIME".to_string()),
            required: false,
            exists: voxtral_llamacpp::is_llamacpp_requested(),
            note: Some("set to 'llamacpp' to force the embedded llama.cpp runtime".to_string()),
        },
        optional_file_artifact(
            "voxtral_llama_model",
            config.model_path.as_ref(),
            Some("AMCP_VOXTRAL_LLAMA_MODEL_PATH"),
            true,
        ),
        optional_file_artifact(
            "voxtral_llama_mmproj",
            config.mmproj_path.as_ref(),
            Some("AMCP_VOXTRAL_LLAMA_MMPROJ_PATH"),
            true,
        ),
        RuntimeArtifactStatus {
            name: "voxtral_llama_hf_repo".to_string(),
            kind: RuntimeArtifactKind::Download,
            path: Some(config.repo_id.clone()),
            env_var: Some("AMCP_VOXTRAL_LLAMA_REPO_ID".to_string()),
            required: false,
            exists: config.model_path.is_some() || config.mmproj_path.is_some(),
            note: Some(format!(
                "uses the shared Hugging Face cache when explicit paths are unset; model_file={:?} mmproj_file={:?}",
                config.model_file, config.mmproj_file
            )),
        },
        RuntimeArtifactStatus {
            name: "voxtral_llama_metal".to_string(),
            kind: RuntimeArtifactKind::Command,
            path: Some(format!(
                "feature=voxtral-realtime-metal use_gpu={} n_gpu_layers={}",
                config.use_gpu, config.n_gpu_layers
            )),
            env_var: Some("AMCP_VOXTRAL_PATCHED_LLAMA_LINK_METAL".to_string()),
            required: false,
            exists: cfg!(feature = "voxtral-realtime-metal"),
            note: Some(
                "patched llama.cpp Metal build requires libggml-metal.dylib or embedded Metal support at build time"
                    .to_string(),
            ),
        },
        RuntimeArtifactStatus {
            name: "voxtral_llama_vulkan".to_string(),
            kind: RuntimeArtifactKind::Command,
            path: Some(format!(
                "feature=voxtral-llamacpp-vulkan use_gpu={} n_gpu_layers={}",
                config.use_gpu, config.n_gpu_layers
            )),
            env_var: Some("AMCP_VOXTRAL_LLAMA_GPU_LAYERS".to_string()),
            required: false,
            exists: cfg!(any(
                feature = "voxtral-llamacpp-vulkan",
                feature = "voxtral-realtime-vulkan"
            )),
            note: Some("llama-cpp-4 Vulkan feature requires Vulkan SDK at build time".to_string()),
        },
        RuntimeArtifactStatus {
            name: "voxtral_llama_realtime_patched".to_string(),
            kind: RuntimeArtifactKind::Command,
            path: Some("feature=voxtral-llamacpp-realtime-patched".to_string()),
            env_var: Some("AMCP_VOXTRAL_PATCHED_LLAMA_DIR".to_string()),
            required: false,
            exists: cfg!(feature = "voxtral-llamacpp-realtime-patched"),
            note: Some(
                "links a patched llama.cpp build that exports Voxtral Realtime dual-stream mtmd helpers"
                    .to_string(),
            ),
        },
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
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

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
        if cfg!(feature = "whisper-vulkan") {
            assert!(status
                .selected_accelerators
                .contains(&HardwareBackend::Vulkan));
        }
        assert!(status.selected_accelerators.contains(&HardwareBackend::Cpu));
        assert!(status
            .artifacts
            .iter()
            .any(|artifact| artifact.name == "whisper_model"));
    }

    #[test]
    fn reports_voxtral_artifact_requirements() {
        let _guard = env_lock().lock().unwrap();
        std::env::remove_var("AMCP_VOXTRAL_RUNTIME");
        std::env::remove_var("AMCP_VOXTRAL_LLAMA_MODEL_PATH");
        std::env::remove_var("AMCP_VOXTRAL_LLAMA_MMPROJ_PATH");
        let status = runtime_backend_status("voxtral-mini-4b", &[HardwareBackend::Cpu]);

        assert_eq!(status.family, ModelFamily::Voxtral);
        assert!(status
            .artifacts
            .iter()
            .any(
                |artifact| artifact.env_var.as_deref() == Some("AMCP_VOXTRAL_LLAMA_MODEL_PATH")
                    || artifact.env_var.as_deref() == Some("AMCP_VOXTRAL_LLAMA_REPO_ID")
            ));
        assert!(status.artifacts.iter().any(|artifact| artifact.required));
    }

    #[test]
    fn reports_voxtral_llamacpp_runtime_when_requested() {
        let _guard = env_lock().lock().unwrap();
        std::env::set_var("AMCP_VOXTRAL_RUNTIME", "llamacpp");
        std::env::set_var("AMCP_VOXTRAL_LLAMA_MODEL_PATH", "missing-model.gguf");
        std::env::set_var("AMCP_VOXTRAL_LLAMA_MMPROJ_PATH", "missing-mmproj.gguf");
        std::env::set_var("AMCP_VOXTRAL_PATCHED_LLAMA_LINK_VULKAN", "1");
        let status = runtime_backend_status("voxtral-mini-4b", &[HardwareBackend::Vulkan]);
        std::env::remove_var("AMCP_VOXTRAL_RUNTIME");
        std::env::remove_var("AMCP_VOXTRAL_LLAMA_MODEL_PATH");
        std::env::remove_var("AMCP_VOXTRAL_LLAMA_MMPROJ_PATH");
        std::env::remove_var("AMCP_VOXTRAL_PATCHED_LLAMA_LINK_VULKAN");

        assert_eq!(status.backend, RuntimeBackendKind::Placeholder);
        if cfg!(any(
            feature = "voxtral-llamacpp-vulkan",
            feature = "voxtral-realtime-vulkan"
        )) {
            assert!(status
                .selected_accelerators
                .contains(&HardwareBackend::Vulkan));
        }
        assert!(status
            .artifacts
            .iter()
            .any(|artifact| artifact.env_var.as_deref() == Some("AMCP_VOXTRAL_LLAMA_MMPROJ_PATH")));
    }
}
