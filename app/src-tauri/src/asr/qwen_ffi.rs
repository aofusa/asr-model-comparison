use crate::accelerator::HardwareBackend;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[cfg(feature = "qwen")]
use std::ffi::{c_char, c_void, CStr, CString};

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
    pub transcribe_audio_symbol_found: bool,
    pub free_symbol_found: bool,
    pub model_load_checked: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QwenFfiTranscription {
    pub text: String,
    pub model_dir: PathBuf,
    pub library_path: PathBuf,
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
        let _qwen_transcribe_audio: libloading::Symbol<
            '_,
            unsafe extern "C" fn(*mut c_void, *const libc::c_float, i32) -> *mut c_char,
        > = library
            .get(b"qwen_transcribe_audio\0")
            .map_err(|error| format!("missing qwen_transcribe_audio symbol: {error}"))?;
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
        transcribe_audio_symbol_found: true,
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

#[cfg(feature = "qwen")]
pub fn transcribe_qwen_audio(
    samples: &[f32],
    language: Option<&str>,
    previous_text: Option<&str>,
    available_backends: &[HardwareBackend],
) -> Result<Option<QwenFfiTranscription>, String> {
    if samples.is_empty() {
        return Ok(Some(QwenFfiTranscription {
            text: String::new(),
            model_dir: PathBuf::new(),
            library_path: PathBuf::new(),
        }));
    }

    let Some(config) = configure_qwen_ffi(available_backends) else {
        return Ok(None);
    };
    if !config.library_path.is_file() || !config.model_dir.is_dir() {
        return Ok(None);
    }

    let library = unsafe { libloading::Library::new(&config.library_path) }
        .map_err(|error| format!("failed to load Qwen ASR library: {error}"))?;

    unsafe {
        let qwen_load: libloading::Symbol<'_, unsafe extern "C" fn(*const c_char) -> *mut c_void> =
            library
                .get(b"qwen_load\0")
                .map_err(|error| format!("missing qwen_load symbol: {error}"))?;
        let qwen_transcribe_audio: libloading::Symbol<
            '_,
            unsafe extern "C" fn(*mut c_void, *const libc::c_float, i32) -> *mut c_char,
        > = library
            .get(b"qwen_transcribe_audio\0")
            .map_err(|error| format!("missing qwen_transcribe_audio symbol: {error}"))?;
        let qwen_free: libloading::Symbol<'_, unsafe extern "C" fn(*mut c_void)> = library
            .get(b"qwen_free\0")
            .map_err(|error| format!("missing qwen_free symbol: {error}"))?;
        let qwen_set_force_language: Option<
            libloading::Symbol<'_, unsafe extern "C" fn(*mut c_void, *const c_char)>,
        > = library.get(b"qwen_set_force_language\0").ok();
        let qwen_set_prompt: Option<
            libloading::Symbol<'_, unsafe extern "C" fn(*mut c_void, *const c_char)>,
        > = library.get(b"qwen_set_prompt\0").ok();

        let model_dir = CString::new(config.model_dir.to_string_lossy().as_bytes())
            .map_err(|_| "Qwen model directory contains an interior NUL byte".to_string())?;
        let language = qwen_language_name(language)
            .map(CString::new)
            .transpose()
            .map_err(|_| "Qwen language contains an interior NUL byte".to_string())?;
        let prompt = qwen_prompt(previous_text)
            .map(CString::new)
            .transpose()
            .map_err(|_| "Qwen prompt contains an interior NUL byte".to_string())?;
        let ctx = qwen_load(model_dir.as_ptr());
        if ctx.is_null() {
            return Err("qwen_load returned null".to_string());
        }

        if let (Some(set_language), Some(language)) = (qwen_set_force_language, language.as_ref()) {
            set_language(ctx, language.as_ptr());
        }

        if let (Some(set_prompt), Some(prompt)) = (qwen_set_prompt, prompt.as_ref()) {
            set_prompt(ctx, prompt.as_ptr());
        }

        let raw_text = qwen_transcribe_audio(ctx, samples.as_ptr(), samples.len() as i32);
        qwen_free(ctx);
        if raw_text.is_null() {
            return Err("qwen_transcribe_audio returned null".to_string());
        }

        let text = CStr::from_ptr(raw_text)
            .to_string_lossy()
            .trim()
            .to_string();
        libc::free(raw_text.cast::<c_void>());

        Ok(Some(QwenFfiTranscription {
            text,
            model_dir: config.model_dir,
            library_path: config.library_path,
        }))
    }
}

#[cfg(not(feature = "qwen"))]
pub fn transcribe_qwen_audio(
    _samples: &[f32],
    _language: Option<&str>,
    _previous_text: Option<&str>,
    _available_backends: &[HardwareBackend],
) -> Result<Option<QwenFfiTranscription>, String> {
    Ok(None)
}

#[cfg(any(test, feature = "qwen"))]
fn qwen_language_name(language: Option<&str>) -> Option<&'static str> {
    match language?.trim().to_ascii_lowercase().as_str() {
        "" | "auto" | "none" => None,
        "ja" | "japanese" => Some("Japanese"),
        "en" | "english" => Some("English"),
        "zh" | "chinese" => Some("Chinese"),
        "ko" | "korean" => Some("Korean"),
        "fr" | "french" => Some("French"),
        "de" | "german" => Some("German"),
        "es" | "spanish" => Some("Spanish"),
        "it" | "italian" => Some("Italian"),
        "pt" | "portuguese" => Some("Portuguese"),
        "ru" | "russian" => Some("Russian"),
        "ar" | "arabic" => Some("Arabic"),
        "hi" | "hindi" => Some("Hindi"),
        "vi" | "vietnamese" => Some("Vietnamese"),
        "th" | "thai" => Some("Thai"),
        "id" | "indonesian" => Some("Indonesian"),
        "tr" | "turkish" => Some("Turkish"),
        "nl" | "dutch" => Some("Dutch"),
        "pl" | "polish" => Some("Polish"),
        "sv" | "swedish" => Some("Swedish"),
        _ => None,
    }
}

#[cfg(feature = "qwen")]
fn qwen_prompt(previous_text: Option<&str>) -> Option<String> {
    let previous_text = previous_text?.trim();
    if previous_text.is_empty() {
        return None;
    }

    Some(format!("Previous transcript context: {previous_text}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_plan_keeps_cpu_fallback() {
        let backends = qwen_backend_plan(&[HardwareBackend::Blas]);

        assert_eq!(backends, vec![HardwareBackend::Blas, HardwareBackend::Cpu]);
    }

    #[test]
    fn maps_language_codes_to_qwen_names() {
        assert_eq!(qwen_language_name(Some("ja")), Some("Japanese"));
        assert_eq!(qwen_language_name(Some("auto")), None);
    }
}
