use crate::accelerator::HardwareBackend;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[cfg(feature = "qwen")]
use std::ffi::{c_char, c_void, CString};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QwenFfiConfig {
    pub library_path: PathBuf,
    pub model_dir: PathBuf,
    pub backends: Vec<HardwareBackend>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QwenFfiValidation {
    pub library_path: PathBuf,
    pub model_dir: PathBuf,
    pub load_symbol_found: bool,
    pub free_symbol_found: bool,
    pub model_load_checked: bool,
}

pub fn configure_qwen_ffi(available_backends: &[HardwareBackend]) -> Option<QwenFfiConfig> {
    let library_path = std::env::var("AMCP_QWEN_ASR_LIB")
        .ok()
        .filter(|path| !path.trim().is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("AMCP_QWEN_ASR_DIR")
                .ok()
                .filter(|path| !path.trim().is_empty())
                .map(|dir| {
                    let file_name = if cfg!(windows) {
                        "qwen_asr.dll"
                    } else if cfg!(target_os = "macos") {
                        "libqwen_asr.dylib"
                    } else {
                        "libqwen_asr.so"
                    };
                    PathBuf::from(dir).join(file_name)
                })
        })?;
    let model_dir = std::env::var("AMCP_QWEN_MODEL_DIR")
        .ok()
        .filter(|path| !path.trim().is_empty())
        .map(PathBuf::from)?;

    Some(QwenFfiConfig {
        library_path,
        model_dir,
        backends: qwen_backend_plan(available_backends),
    })
}

pub fn qwen_backend_plan(available_backends: &[HardwareBackend]) -> Vec<HardwareBackend> {
    let mut backends = Vec::new();
    for backend in [
        HardwareBackend::Cuda,
        HardwareBackend::DirectMl,
        HardwareBackend::Vulkan,
        HardwareBackend::Wgpu,
        HardwareBackend::OpenVino,
        HardwareBackend::Blas,
        HardwareBackend::Cpu,
    ] {
        if available_backends.contains(&backend) || backend == HardwareBackend::Cpu {
            backends.push(backend);
        }
    }
    backends
}

#[cfg(feature = "qwen")]
pub fn validate_qwen_ffi(
    config: &QwenFfiConfig,
    load_model: bool,
) -> Result<QwenFfiValidation, String> {
    if !config.library_path.is_file() {
        return Err(format!(
            "Qwen ASR library does not exist: {}",
            config.library_path.display()
        ));
    }
    if !config.model_dir.is_dir() {
        return Err(format!(
            "Qwen model directory does not exist: {}",
            config.model_dir.display()
        ));
    }

    let library = unsafe { libloading::Library::new(&config.library_path) }
        .map_err(|error| format!("failed to load Qwen ASR library: {error}"))?;

    unsafe {
        let qwen_load: libloading::Symbol<'_, unsafe extern "C" fn(*const c_char) -> *mut c_void> =
            library
                .get(b"qwen_load\0")
                .map_err(|error| format!("missing qwen_load symbol: {error}"))?;
        let _qwen_free: libloading::Symbol<'_, unsafe extern "C" fn(*mut c_void)> = library
            .get(b"qwen_free\0")
            .map_err(|error| format!("missing qwen_free symbol: {error}"))?;

        if load_model {
            let model_dir = CString::new(config.model_dir.to_string_lossy().as_bytes())
                .map_err(|_| "Qwen model directory contains an interior NUL byte".to_string())?;
            let ctx = qwen_load(model_dir.as_ptr());
            if ctx.is_null() {
                return Err("qwen_load returned null".to_string());
            }
            _qwen_free(ctx);
        }
    }

    Ok(QwenFfiValidation {
        library_path: config.library_path.clone(),
        model_dir: config.model_dir.clone(),
        load_symbol_found: true,
        free_symbol_found: true,
        model_load_checked: load_model,
    })
}

#[cfg(not(feature = "qwen"))]
pub fn validate_qwen_ffi(
    _config: &QwenFfiConfig,
    _load_model: bool,
) -> Result<QwenFfiValidation, String> {
    Err("qwen feature is not enabled".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_plan_keeps_cpu_fallback() {
        let backends = qwen_backend_plan(&[HardwareBackend::Blas]);

        assert_eq!(backends, vec![HardwareBackend::Blas, HardwareBackend::Cpu]);
    }
}
