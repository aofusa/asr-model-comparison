use crate::accelerator::ModelFamily;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelInfo {
    pub id: &'static str,
    pub name: &'static str,
    pub family: ModelFamily,
    pub size: &'static str,
    pub parameters: Option<&'static str>,
    pub description: Option<&'static str>,
    pub recommended_vram_gb: Option<f32>,
    pub languages: Vec<&'static str>,
}

pub fn available_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            id: "whisper-tiny",
            name: "Whisper Tiny",
            family: ModelFamily::Whisper,
            size: "tiny",
            parameters: Some("39M"),
            description: Some("最速の軽量モデル。精度は低めだがリソース消費が非常に少ない。"),
            recommended_vram_gb: Some(2.0),
            languages: vec!["ja", "en", "zh"],
        },
        ModelInfo {
            id: "whisper-small",
            name: "Whisper Small",
            family: ModelFamily::Whisper,
            size: "small",
            parameters: Some("244M"),
            description: Some("バランスの取れた軽量モデル。"),
            recommended_vram_gb: Some(4.0),
            languages: vec!["ja", "en", "zh"],
        },
        ModelInfo {
            id: "whisper-medium",
            name: "Whisper Medium",
            family: ModelFamily::Whisper,
            size: "medium",
            parameters: Some("769M"),
            description: Some("精度と速度のバランスが良い標準モデル。"),
            recommended_vram_gb: Some(7.0),
            languages: vec!["ja", "en", "zh"],
        },
        ModelInfo {
            id: "whisper-large-v3-turbo",
            name: "Whisper Large-v3 Turbo",
            family: ModelFamily::Whisper,
            size: "large-v3-turbo",
            parameters: Some("~1.6B"),
            description: Some("高速化された高精度モデル。"),
            recommended_vram_gb: Some(9.0),
            languages: vec!["ja", "en", "zh"],
        },
        ModelInfo {
            id: "qwen3-asr-0.6b",
            name: "Qwen3-ASR 0.6B",
            family: ModelFamily::Qwen3,
            size: "0.6B",
            parameters: Some("0.6B"),
            description: Some("Qwen3シリーズの軽量高性能モデル。日本語に強い。"),
            recommended_vram_gb: Some(5.0),
            languages: vec!["ja", "en", "zh"],
        },
        ModelInfo {
            id: "qwen3-asr-1.7b",
            name: "Qwen3-ASR 1.7B",
            family: ModelFamily::Qwen3,
            size: "1.7B",
            parameters: Some("1.7B"),
            description: Some("Qwen3シリーズのバランスモデル。精度と速度の優秀なトレードオフ。"),
            recommended_vram_gb: Some(10.0),
            languages: vec!["ja", "en", "zh"],
        },
        ModelInfo {
            id: "voxtral-mini-4b",
            name: "Voxtral Mini 4B Realtime",
            family: ModelFamily::Voxtral,
            size: "Mini 4B",
            parameters: Some("4B"),
            description: Some("Mistral製の高精度Realtime向けモデル。"),
            recommended_vram_gb: Some(16.0),
            languages: vec!["ja", "en", "zh"],
        },
    ]
}

pub fn family_for_model(model_id: &str) -> Option<ModelFamily> {
    available_models()
        .into_iter()
        .find(|model| model.id == model_id)
        .map(|model| model.family)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_python_model_ids_compatible() {
        let ids: Vec<_> = available_models()
            .into_iter()
            .map(|model| model.id)
            .collect();

        assert!(ids.contains(&"whisper-tiny"));
        assert!(ids.contains(&"qwen3-asr-0.6b"));
        assert!(ids.contains(&"voxtral-mini-4b"));
    }
}
