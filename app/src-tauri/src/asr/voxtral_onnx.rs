use crate::accelerator::HardwareBackend;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[cfg(feature = "voxtral")]
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VoxtralOnnxConfig {
    pub model_path: PathBuf,
    pub providers: Vec<HardwareBackend>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VoxtralOnnxSessionInfo {
    pub model_path: PathBuf,
    pub providers: Vec<HardwareBackend>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

pub fn resolve_voxtral_model_path() -> Option<PathBuf> {
    std::env::var("AMCP_VOXTRAL_ONNX_MODEL_PATH")
        .ok()
        .filter(|path| !path.trim().is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("AMCP_VOXTRAL_MODEL_DIR")
                .ok()
                .filter(|path| !path.trim().is_empty())
                .map(|dir| PathBuf::from(dir).join("model.onnx"))
        })
}

pub fn configure_voxtral_onnx(available_backends: &[HardwareBackend]) -> Option<VoxtralOnnxConfig> {
    let model_path = resolve_voxtral_model_path()?;
    Some(VoxtralOnnxConfig {
        model_path,
        providers: voxtral_provider_plan(available_backends),
    })
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
fn create_session_info(
    model_path: &Path,
    providers: &[HardwareBackend],
) -> Result<VoxtralOnnxSessionInfo, String> {
    let mut builder = ort::session::Session::builder()
        .map_err(|error| format!("failed to create ONNX Runtime session builder: {error}"))?;

    #[cfg(any(feature = "voxtral-cuda", feature = "voxtral-directml"))]
    {
        let execution_providers = build_execution_providers(providers);
        if !execution_providers.is_empty() {
            builder = builder
                .with_execution_providers(execution_providers)
                .map_err(|error| format!("failed to register ONNX execution providers: {error}"))?;
        }
    }

    let session = builder
        .commit_from_file(model_path)
        .map_err(|error| format!("failed to load Voxtral ONNX model: {error}"))?;
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
}
