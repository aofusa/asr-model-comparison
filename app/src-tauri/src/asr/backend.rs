use crate::accelerator::{HardwareBackend, ModelFamily};
use crate::asr::qwen_ffi;
use crate::asr::voxtral_onnx;
use crate::models::{available_models, family_for_model};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeBackendKind {
    WhisperRs,
    QwenC,
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
    pub reason: String,
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
    let ffi_config = qwen_ffi::configure_qwen_ffi(available_backends);
    let ffi_paths_available = ffi_config
        .as_ref()
        .map(|config| config.library_path.is_file() && config.model_dir.is_dir())
        .unwrap_or(false);
    let real_inference_available = configured && ffi_paths_available;
    RuntimeBackendStatus {
        model_id: model_id.to_string(),
        family,
        backend: if real_inference_available {
            RuntimeBackendKind::QwenC
        } else {
            RuntimeBackendKind::Placeholder
        },
        real_inference_available,
        configured,
        selected_accelerators: ffi_config
            .as_ref()
            .map(|config| config.backends.clone())
            .unwrap_or_else(|| {
                filter_backends(
                    available_backends,
                    &[
                        HardwareBackend::Cuda,
                        HardwareBackend::DirectMl,
                        HardwareBackend::Vulkan,
                        HardwareBackend::Wgpu,
                        HardwareBackend::OpenVino,
                        HardwareBackend::Blas,
                        HardwareBackend::Cpu,
                    ],
                )
            }),
        reason: if real_inference_available {
            "qwen feature is enabled and Qwen ASR library/model paths exist.".to_string()
        } else if configured {
            "qwen feature is enabled, but AMCP_QWEN_ASR_LIB or AMCP_QWEN_ASR_DIR plus AMCP_QWEN_MODEL_DIR does not point to existing paths."
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
    let model_file_available = onnx_config
        .as_ref()
        .map(|config| config.model_path.is_file())
        .unwrap_or(false);
    let real_inference_available = configured && model_file_available;
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
        reason: if real_inference_available {
            "voxtral feature is enabled and a Voxtral ONNX model file is configured.".to_string()
        } else if configured {
            "voxtral feature is enabled, but AMCP_VOXTRAL_ONNX_MODEL_PATH or AMCP_VOXTRAL_MODEL_DIR/model.onnx does not point to an existing model file."
                .to_string()
        } else {
            "voxtral feature is not enabled; using placeholder inference.".to_string()
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

    #[test]
    fn reports_placeholder_for_unconfigured_qwen() {
        let status = runtime_backend_status(
            "qwen3-asr-0.6b",
            &[HardwareBackend::Cuda, HardwareBackend::Cpu],
        );

        assert_eq!(status.family, ModelFamily::Qwen3);
        assert_eq!(status.backend, RuntimeBackendKind::Placeholder);
        assert!(status.selected_accelerators.contains(&HardwareBackend::Cpu));
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
    }
}
