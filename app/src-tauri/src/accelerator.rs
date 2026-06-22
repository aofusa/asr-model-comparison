use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcceleratorPreference {
    Auto,
    Gpu,
    Cpu,
}

impl Default for AcceleratorPreference {
    fn default() -> Self {
        Self::Auto
    }
}

impl std::str::FromStr for AcceleratorPreference {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "gpu" => Ok(Self::Gpu),
            "cpu" => Ok(Self::Cpu),
            other => Err(format!("unknown accelerator preference: {other}")),
        }
    }
}

impl std::str::FromStr for HardwareBackend {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "cuda" => Ok(Self::Cuda),
            "directml" | "direct_ml" | "dml" => Ok(Self::DirectMl),
            "metal" => Ok(Self::Metal),
            "coreml" | "core_ml" => Ok(Self::CoreMl),
            "vulkan" => Ok(Self::Vulkan),
            "wgpu" => Ok(Self::Wgpu),
            "openvino" | "open_vino" => Ok(Self::OpenVino),
            "nnapi" | "nn_api" => Ok(Self::NnApi),
            "blas" => Ok(Self::Blas),
            "cpu" => Ok(Self::Cpu),
            other => Err(format!("unknown hardware backend: {other}")),
        }
    }
}

impl fmt::Display for AcceleratorPreference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::Auto => "auto",
            Self::Gpu => "gpu",
            Self::Cpu => "cpu",
        };
        f.write_str(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFamily {
    Whisper,
    Qwen3,
    Voxtral,
}

impl fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::Whisper => "whisper",
            Self::Qwen3 => "qwen3",
            Self::Voxtral => "voxtral",
        };
        f.write_str(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HardwareBackend {
    Cuda,
    DirectMl,
    Metal,
    CoreMl,
    Vulkan,
    Wgpu,
    OpenVino,
    NnApi,
    Blas,
    Cpu,
}

impl HardwareBackend {
    pub fn is_accelerated(self) -> bool {
        matches!(
            self,
            Self::Cuda
                | Self::DirectMl
                | Self::Metal
                | Self::CoreMl
                | Self::Vulkan
                | Self::Wgpu
                | Self::OpenVino
                | Self::NnApi
        )
    }
}

impl fmt::Display for HardwareBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::Cuda => "cuda",
            Self::DirectMl => "directml",
            Self::Metal => "metal",
            Self::CoreMl => "coreml",
            Self::Vulkan => "vulkan",
            Self::Wgpu => "wgpu",
            Self::OpenVino => "openvino",
            Self::NnApi => "nnapi",
            Self::Blas => "blas",
            Self::Cpu => "cpu",
        };
        f.write_str(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcceleratorSelection {
    pub preference: AcceleratorPreference,
    pub selected: HardwareBackend,
    pub attempted: Vec<HardwareBackend>,
    pub fallback_used: bool,
    pub reason: String,
}

pub fn select_accelerator(
    family: ModelFamily,
    preference: AcceleratorPreference,
    available: &[HardwareBackend],
) -> AcceleratorSelection {
    let requested_plan = prioritized_plan(family, preference);
    let requested_preferred = requested_plan[0];
    let plan = runtime_supported_plan(family, requested_plan);
    let selected = plan
        .iter()
        .copied()
        .find(|candidate| available.contains(candidate))
        .unwrap_or(HardwareBackend::Cpu);

    let fallback_used = preference != AcceleratorPreference::Cpu && selected != requested_preferred;
    let reason = if preference == AcceleratorPreference::Cpu {
        "CPU was explicitly requested.".to_string()
    } else if fallback_used {
        format!(
            "{} was selected after preferred backends were unavailable.",
            selected
        )
    } else {
        format!(
            "{} is the preferred backend for {} on this platform.",
            selected, family
        )
    };

    AcceleratorSelection {
        preference,
        selected,
        attempted: plan,
        fallback_used,
        reason,
    }
}

pub fn detect_available_backends() -> Vec<HardwareBackend> {
    detect_available_backends_with(
        std::env::consts::OS,
        |key| std::env::var(key).ok(),
        command_exists,
        |path| std::path::Path::new(path).is_file(),
    )
}

fn detect_available_backends_with(
    os: &str,
    env_get: impl Fn(&str) -> Option<String>,
    command_exists: impl Fn(&str) -> bool,
    path_exists: impl Fn(&str) -> bool,
) -> Vec<HardwareBackend> {
    if let Some(configured) = env_get("AMCP_AVAILABLE_BACKENDS") {
        let mut parsed: Vec<_> = configured
            .split(',')
            .filter_map(|value| value.parse::<HardwareBackend>().ok())
            .collect();
        push_unique(&mut parsed, HardwareBackend::Cpu);
        return parsed;
    }

    let mut available = Vec::new();
    match os {
        "windows" => {
            if env_get("CUDA_PATH").is_some()
                || env_get("CUDA_HOME").is_some()
                || command_exists("nvidia-smi")
                || command_exists("nvcc")
            {
                push_unique(&mut available, HardwareBackend::Cuda);
            }
            if env_get("VK_SDK").is_some() || command_exists("vulkaninfo") {
                push_unique(&mut available, HardwareBackend::Vulkan);
            }
            if env_get("AMCP_ENABLE_DIRECTML").as_deref() == Some("1")
                || env_get("DIRECTML_PATH").is_some()
                || env_get("WINDIR")
                    .map(|windir| format!("{windir}\\System32\\DirectML.dll"))
                    .map(|path| path_exists(&path))
                    .unwrap_or(false)
            {
                push_unique(&mut available, HardwareBackend::DirectMl);
            }
            push_unique(&mut available, HardwareBackend::Wgpu);
            if env_get("OPENVINO_DIR").is_some()
                || env_get("INTEL_OPENVINO_DIR").is_some()
                || command_exists("benchmark_app")
            {
                push_unique(&mut available, HardwareBackend::OpenVino);
            }
            if env_get("OPENBLAS_DIR").is_some()
                || env_get("MKLROOT").is_some()
                || env_get("AMCP_ENABLE_BLAS").as_deref() == Some("1")
            {
                push_unique(&mut available, HardwareBackend::Blas);
            }
        }
        "linux" => {
            if env_get("CUDA_PATH").is_some()
                || env_get("CUDA_HOME").is_some()
                || command_exists("nvidia-smi")
                || command_exists("nvcc")
            {
                push_unique(&mut available, HardwareBackend::Cuda);
            }
            if env_get("VK_SDK").is_some() || command_exists("vulkaninfo") {
                push_unique(&mut available, HardwareBackend::Vulkan);
            }
            push_unique(&mut available, HardwareBackend::Wgpu);
            if env_get("OPENVINO_DIR").is_some()
                || env_get("INTEL_OPENVINO_DIR").is_some()
                || command_exists("benchmark_app")
            {
                push_unique(&mut available, HardwareBackend::OpenVino);
            }
            if env_get("OPENBLAS_DIR").is_some()
                || env_get("MKLROOT").is_some()
                || env_get("AMCP_ENABLE_BLAS").as_deref() == Some("1")
            {
                push_unique(&mut available, HardwareBackend::Blas);
            }
        }
        "macos" => {
            push_unique(&mut available, HardwareBackend::Metal);
            push_unique(&mut available, HardwareBackend::CoreMl);
            push_unique(&mut available, HardwareBackend::Wgpu);
            push_unique(&mut available, HardwareBackend::Blas);
        }
        "ios" => {
            push_unique(&mut available, HardwareBackend::Metal);
            push_unique(&mut available, HardwareBackend::CoreMl);
            push_unique(&mut available, HardwareBackend::Blas);
        }
        "android" => {
            push_unique(&mut available, HardwareBackend::NnApi);
            push_unique(&mut available, HardwareBackend::Vulkan);
            push_unique(&mut available, HardwareBackend::Wgpu);
        }
        _ => {}
    }

    push_unique(&mut available, HardwareBackend::Cpu);
    available
}

fn push_unique(backends: &mut Vec<HardwareBackend>, backend: HardwareBackend) {
    if !backends.contains(&backend) {
        backends.push(backend);
    }
}

fn command_exists(command: &str) -> bool {
    command_exists_in_path(
        command,
        std::env::var_os("PATH"),
        std::env::var_os("PATHEXT"),
    )
}

fn command_exists_in_path(
    command: &str,
    path: Option<std::ffi::OsString>,
    path_ext: Option<std::ffi::OsString>,
) -> bool {
    let command_path = std::path::Path::new(command);
    if command_path.components().count() > 1 {
        return command_path.is_file();
    }

    let Some(path) = path else {
        return false;
    };

    let extensions = executable_extensions(command, path_ext);
    std::env::split_paths(&path).any(|dir| {
        extensions
            .iter()
            .any(|extension| dir.join(format!("{command}{extension}")).is_file())
    })
}

fn executable_extensions(command: &str, path_ext: Option<std::ffi::OsString>) -> Vec<String> {
    if std::path::Path::new(command).extension().is_some() {
        return vec![String::new()];
    }

    if cfg!(windows) {
        let configured = path_ext
            .and_then(|value| value.into_string().ok())
            .unwrap_or_else(|| ".COM;.EXE;.BAT;.CMD".to_string());
        configured
            .split(';')
            .filter_map(|extension| {
                let trimmed = extension.trim();
                (!trimmed.is_empty()).then(|| {
                    if trimmed.starts_with('.') {
                        trimmed.to_string()
                    } else {
                        format!(".{trimmed}")
                    }
                })
            })
            .collect()
    } else {
        vec![String::new()]
    }
}

fn prioritized_plan(
    family: ModelFamily,
    preference: AcceleratorPreference,
) -> Vec<HardwareBackend> {
    prioritized_plan_for_os(std::env::consts::OS, family, preference)
}

fn prioritized_plan_for_os(
    os: &str,
    family: ModelFamily,
    preference: AcceleratorPreference,
) -> Vec<HardwareBackend> {
    if preference == AcceleratorPreference::Cpu {
        return vec![HardwareBackend::Cpu];
    }

    let mut plan = match (os, family) {
        ("macos", ModelFamily::Whisper) => vec![
            HardwareBackend::Metal,
            HardwareBackend::Vulkan,
            HardwareBackend::Cpu,
        ],
        ("macos", ModelFamily::Qwen3) => vec![HardwareBackend::Metal, HardwareBackend::Cpu],
        ("macos", ModelFamily::Voxtral) => vec![HardwareBackend::CoreMl, HardwareBackend::Cpu],
        ("windows", ModelFamily::Whisper) => vec![
            HardwareBackend::Cuda,
            HardwareBackend::Vulkan,
            HardwareBackend::Cpu,
        ],
        ("windows", ModelFamily::Qwen3) => vec![HardwareBackend::Cuda, HardwareBackend::Cpu],
        ("windows", ModelFamily::Voxtral) => vec![
            HardwareBackend::DirectMl,
            HardwareBackend::Cuda,
            HardwareBackend::Cpu,
        ],
        ("linux", ModelFamily::Whisper) => vec![
            HardwareBackend::Cuda,
            HardwareBackend::Vulkan,
            HardwareBackend::Cpu,
        ],
        ("linux", ModelFamily::Qwen3) => vec![HardwareBackend::Cuda, HardwareBackend::Cpu],
        ("linux", ModelFamily::Voxtral) => vec![HardwareBackend::Cuda, HardwareBackend::Cpu],
        ("ios", ModelFamily::Whisper) => vec![HardwareBackend::Metal, HardwareBackend::Cpu],
        ("ios", ModelFamily::Qwen3) => vec![HardwareBackend::Cpu],
        ("ios", ModelFamily::Voxtral) => vec![HardwareBackend::CoreMl, HardwareBackend::Cpu],
        ("android", ModelFamily::Whisper) => vec![HardwareBackend::Vulkan, HardwareBackend::Cpu],
        ("android", ModelFamily::Qwen3) => vec![HardwareBackend::Cpu],
        ("android", ModelFamily::Voxtral) => vec![
            HardwareBackend::NnApi,
            HardwareBackend::Vulkan,
            HardwareBackend::Cpu,
        ],
        (_, ModelFamily::Whisper) => vec![HardwareBackend::Vulkan, HardwareBackend::Cpu],
        (_, ModelFamily::Qwen3) => vec![HardwareBackend::Cpu],
        (_, ModelFamily::Voxtral) => vec![HardwareBackend::Cpu],
    };

    if preference == AcceleratorPreference::Gpu {
        plan.retain(|backend| backend.is_accelerated() || *backend == HardwareBackend::Cpu);
        if !plan.contains(&HardwareBackend::Cpu) {
            plan.push(HardwareBackend::Cpu);
        }
    }

    plan
}

fn runtime_supported_plan(family: ModelFamily, plan: Vec<HardwareBackend>) -> Vec<HardwareBackend> {
    let supported = runtime_supported_backends(family);
    let mut filtered = plan
        .into_iter()
        .filter(|backend| supported.contains(backend))
        .collect::<Vec<_>>();
    if !filtered.contains(&HardwareBackend::Cpu) {
        filtered.push(HardwareBackend::Cpu);
    }
    filtered
}

fn runtime_supported_backends(family: ModelFamily) -> Vec<HardwareBackend> {
    let mut supported = Vec::new();
    match family {
        ModelFamily::Whisper => {
            if cfg!(feature = "whisper-cuda") {
                supported.push(HardwareBackend::Cuda);
            }
            if cfg!(feature = "whisper-vulkan") {
                supported.push(HardwareBackend::Vulkan);
            }
            if cfg!(any(target_os = "macos", target_os = "ios")) {
                supported.push(HardwareBackend::Metal);
            }
        }
        ModelFamily::Qwen3 => {
            if cfg!(feature = "qwen-cuda") {
                supported.push(HardwareBackend::Cuda);
            }
        }
        ModelFamily::Voxtral => {
            if cfg!(feature = "voxtral-llamacpp-vulkan") {
                supported.push(HardwareBackend::Vulkan);
            }
        }
    }
    supported.push(HardwareBackend::Cpu);
    supported
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_cpu_skips_gpu_backends() {
        let selected = select_accelerator(
            ModelFamily::Whisper,
            AcceleratorPreference::Cpu,
            &[HardwareBackend::Cuda, HardwareBackend::Cpu],
        );

        assert_eq!(selected.selected, HardwareBackend::Cpu);
        assert_eq!(selected.attempted, vec![HardwareBackend::Cpu]);
        assert!(!selected.fallback_used);
    }

    #[test]
    fn gpu_preference_falls_back_to_cpu_safely() {
        let selected = select_accelerator(ModelFamily::Voxtral, AcceleratorPreference::Gpu, &[]);

        assert_eq!(selected.selected, HardwareBackend::Cpu);
        assert!(selected.fallback_used);
        assert!(selected.attempted.contains(&HardwareBackend::Cpu));
    }

    #[cfg(not(any(feature = "whisper-cuda", feature = "whisper-vulkan")))]
    #[test]
    fn whisper_uses_cpu_when_gpu_runtime_features_are_not_compiled() {
        let selected = select_accelerator(
            ModelFamily::Whisper,
            AcceleratorPreference::Auto,
            &[HardwareBackend::Vulkan, HardwareBackend::Cpu],
        );

        assert_eq!(selected.selected, HardwareBackend::Cpu);
        assert_eq!(selected.attempted, vec![HardwareBackend::Cpu]);
    }

    #[test]
    fn ios_prefers_metal_and_coreml() {
        assert_eq!(
            prioritized_plan_for_os("ios", ModelFamily::Whisper, AcceleratorPreference::Auto),
            vec![HardwareBackend::Metal, HardwareBackend::Cpu]
        );
        assert_eq!(
            prioritized_plan_for_os("ios", ModelFamily::Voxtral, AcceleratorPreference::Auto),
            vec![HardwareBackend::CoreMl, HardwareBackend::Cpu]
        );
    }

    #[test]
    fn android_prefers_vulkan_or_nnapi() {
        assert_eq!(
            prioritized_plan_for_os("android", ModelFamily::Whisper, AcceleratorPreference::Auto),
            vec![HardwareBackend::Vulkan, HardwareBackend::Cpu]
        );
        assert_eq!(
            prioritized_plan_for_os("android", ModelFamily::Voxtral, AcceleratorPreference::Auto),
            vec![
                HardwareBackend::NnApi,
                HardwareBackend::Vulkan,
                HardwareBackend::Cpu
            ]
        );
    }

    #[test]
    fn qwen_uses_candle_supported_accelerators() {
        assert_eq!(
            prioritized_plan_for_os("windows", ModelFamily::Qwen3, AcceleratorPreference::Auto),
            vec![HardwareBackend::Cuda, HardwareBackend::Cpu]
        );
        assert_eq!(
            prioritized_plan_for_os("android", ModelFamily::Qwen3, AcceleratorPreference::Auto),
            vec![HardwareBackend::Cpu]
        );
    }

    #[test]
    fn parses_configured_backend_override() {
        let detected = detect_available_backends_with(
            "windows",
            |key| (key == "AMCP_AVAILABLE_BACKENDS").then(|| "vulkan,cpu".to_string()),
            |_| false,
            |_| false,
        );

        assert_eq!(
            detected,
            vec![HardwareBackend::Vulkan, HardwareBackend::Cpu]
        );
    }

    #[test]
    fn windows_detection_finds_cuda_vulkan_directml_and_cpu() {
        let detected = detect_available_backends_with(
            "windows",
            |key| match key {
                "CUDA_PATH" => Some("C:\\CUDA".to_string()),
                "VK_SDK" => Some("C:\\VulkanSDK".to_string()),
                "WINDIR" => Some("C:\\Windows".to_string()),
                _ => None,
            },
            |_| false,
            |path| path.ends_with("DirectML.dll"),
        );

        assert!(detected.contains(&HardwareBackend::Cuda));
        assert!(detected.contains(&HardwareBackend::Vulkan));
        assert!(detected.contains(&HardwareBackend::DirectMl));
        assert!(detected.contains(&HardwareBackend::Wgpu));
        assert!(detected.contains(&HardwareBackend::Cpu));
    }

    #[test]
    fn command_lookup_does_not_require_external_where_command() {
        let temp_root =
            std::env::temp_dir().join(format!("amcp-command-lookup-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp_root);
        std::fs::create_dir_all(&temp_root).unwrap();

        let executable = if cfg!(windows) {
            temp_root.join("vulkaninfo.EXE")
        } else {
            temp_root.join("vulkaninfo")
        };
        std::fs::write(&executable, b"").unwrap();

        let found = command_exists_in_path(
            "vulkaninfo",
            Some(temp_root.as_os_str().to_os_string()),
            Some(".EXE;.CMD".into()),
        );

        let _ = std::fs::remove_dir_all(&temp_root);
        assert!(found);
    }
}
