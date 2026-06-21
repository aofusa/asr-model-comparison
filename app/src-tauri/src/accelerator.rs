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
    Metal,
    CoreMl,
    Vulkan,
    NnApi,
    Blas,
    Cpu,
}

impl HardwareBackend {
    pub fn is_accelerated(self) -> bool {
        matches!(
            self,
            Self::Cuda | Self::Metal | Self::CoreMl | Self::Vulkan | Self::NnApi
        )
    }
}

impl fmt::Display for HardwareBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::CoreMl => "coreml",
            Self::Vulkan => "vulkan",
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
    let plan = prioritized_plan(family, preference);
    let selected = plan
        .iter()
        .copied()
        .find(|candidate| available.contains(candidate))
        .unwrap_or(HardwareBackend::Cpu);

    let fallback_used = selected != plan[0];
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
        ("macos", ModelFamily::Qwen3) => vec![HardwareBackend::Blas, HardwareBackend::Cpu],
        ("macos", ModelFamily::Voxtral) => vec![HardwareBackend::CoreMl, HardwareBackend::Cpu],
        ("windows", ModelFamily::Whisper) => vec![
            HardwareBackend::Cuda,
            HardwareBackend::Vulkan,
            HardwareBackend::Cpu,
        ],
        ("windows", ModelFamily::Qwen3) => vec![HardwareBackend::Blas, HardwareBackend::Cpu],
        ("windows", ModelFamily::Voxtral) => vec![HardwareBackend::Cuda, HardwareBackend::Cpu],
        ("linux", ModelFamily::Whisper) => vec![
            HardwareBackend::Cuda,
            HardwareBackend::Vulkan,
            HardwareBackend::Cpu,
        ],
        ("linux", ModelFamily::Qwen3) => vec![HardwareBackend::Blas, HardwareBackend::Cpu],
        ("linux", ModelFamily::Voxtral) => vec![HardwareBackend::Cuda, HardwareBackend::Cpu],
        ("ios", ModelFamily::Whisper) => vec![HardwareBackend::Metal, HardwareBackend::Cpu],
        ("ios", ModelFamily::Qwen3) => vec![HardwareBackend::Blas, HardwareBackend::Cpu],
        ("ios", ModelFamily::Voxtral) => vec![HardwareBackend::CoreMl, HardwareBackend::Cpu],
        ("android", ModelFamily::Whisper) => vec![HardwareBackend::Vulkan, HardwareBackend::Cpu],
        ("android", ModelFamily::Qwen3) => vec![HardwareBackend::Blas, HardwareBackend::Cpu],
        ("android", ModelFamily::Voxtral) => vec![
            HardwareBackend::NnApi,
            HardwareBackend::Vulkan,
            HardwareBackend::Cpu,
        ],
        (_, ModelFamily::Whisper) => vec![HardwareBackend::Vulkan, HardwareBackend::Cpu],
        (_, ModelFamily::Qwen3) => vec![HardwareBackend::Blas, HardwareBackend::Cpu],
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
}
