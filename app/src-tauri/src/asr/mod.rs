use crate::accelerator::{
    select_accelerator, AcceleratorPreference, AcceleratorSelection, HardwareBackend,
};
use crate::models::family_for_model;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;
use tokio::sync::Mutex;

pub mod audio;
pub mod hybrid;

#[derive(Debug, Error)]
pub enum AsrError {
    #[error("unsupported model: {0}")]
    UnsupportedModel(String),
    #[error("audio payload is empty")]
    EmptyAudio,
    #[error("{0}")]
    Audio(#[from] audio::AudioError),
    #[error("backend failed: {0}")]
    Backend(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionOptions {
    pub model_id: String,
    pub language: Option<String>,
    pub target_language: Option<String>,
    pub beam_size: Option<u8>,
    pub temperature: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub return_timestamps: bool,
    pub previous_text: Option<String>,
    pub accelerator: AcceleratorPreference,
}

impl Default for TranscriptionOptions {
    fn default() -> Self {
        Self {
            model_id: "whisper-tiny".to_string(),
            language: Some("auto".to_string()),
            target_language: None,
            beam_size: Some(6),
            temperature: Some(0.0),
            repetition_penalty: Some(1.15),
            return_timestamps: true,
            previous_text: None,
            accelerator: AcceleratorPreference::Auto,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    pub model_id: String,
    pub text: String,
    pub transcript_text: String,
    pub translated_text: Option<String>,
    pub processing_time_seconds: f64,
    pub language: Option<String>,
    pub target_language: Option<String>,
    pub chunks: Vec<serde_json::Value>,
    pub had_speech: bool,
    pub audio_duration_seconds: f64,
    pub input_sample_rate: u32,
    pub input_channels: u16,
    pub input_rms: f32,
    pub input_peak: f32,
    pub accelerator: AcceleratorSelection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProgress {
    pub r#type: &'static str,
    pub model_id: String,
    pub phase: String,
    pub message: String,
    pub progress: Option<u8>,
    pub elapsed_seconds: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagerStatus {
    pub loaded_model_id: Option<String>,
    pub loaded_backend: Option<HardwareBackend>,
    pub service: &'static str,
}

pub type SharedModelManager = Arc<HybridModelManager>;

#[derive(Debug)]
pub struct HybridModelManager {
    inner: Mutex<ManagerInner>,
    available_backends: Vec<HardwareBackend>,
}

#[derive(Debug, Default)]
struct ManagerInner {
    loaded_model_id: Option<String>,
    loaded_backend: Option<HardwareBackend>,
}

impl HybridModelManager {
    pub fn new(available_backends: Vec<HardwareBackend>) -> Self {
        Self {
            inner: Mutex::new(ManagerInner::default()),
            available_backends,
        }
    }

    pub async fn status(&self) -> ManagerStatus {
        let inner = self.inner.lock().await;
        ManagerStatus {
            loaded_model_id: inner.loaded_model_id.clone(),
            loaded_backend: inner.loaded_backend,
            service: "amcp-rust-backend",
        }
    }

    pub async fn prepare_model(
        &self,
        options: &TranscriptionOptions,
    ) -> Result<(AcceleratorSelection, Vec<ModelProgress>), AsrError> {
        let family = family_for_model(&options.model_id)
            .ok_or_else(|| AsrError::UnsupportedModel(options.model_id.clone()))?;
        let accelerator = select_accelerator(family, options.accelerator, &self.available_backends);
        let started = Instant::now();
        let mut inner = self.inner.lock().await;
        let mut progress = Vec::new();

        if inner.loaded_model_id.as_deref() != Some(options.model_id.as_str())
            || inner.loaded_backend != Some(accelerator.selected)
        {
            if inner.loaded_model_id.is_some() {
                progress.push(ModelProgress {
                    r#type: "model_progress",
                    model_id: options.model_id.clone(),
                    phase: "unloading".to_string(),
                    message: "Unloading previous model to keep single-model memory usage."
                        .to_string(),
                    progress: Some(20),
                    elapsed_seconds: Some(started.elapsed().as_secs_f64()),
                });
            }

            progress.push(ModelProgress {
                r#type: "model_progress",
                model_id: options.model_id.clone(),
                phase: "loading".to_string(),
                message: format!("Preparing {} backend via {}.", family, accelerator.selected),
                progress: Some(60),
                elapsed_seconds: Some(started.elapsed().as_secs_f64()),
            });

            inner.loaded_model_id = Some(options.model_id.clone());
            inner.loaded_backend = Some(accelerator.selected);
        }

        progress.push(ModelProgress {
            r#type: "model_progress",
            model_id: options.model_id.clone(),
            phase: "ready".to_string(),
            message: accelerator.reason.clone(),
            progress: Some(100),
            elapsed_seconds: Some(started.elapsed().as_secs_f64()),
        });

        Ok((accelerator, progress))
    }

    pub async fn transcribe(
        &self,
        audio: &[u8],
        options: TranscriptionOptions,
    ) -> Result<TranscriptionResult, AsrError> {
        if audio.is_empty() {
            return Err(AsrError::EmptyAudio);
        }

        let started = Instant::now();
        let (accelerator, _) = self.prepare_model(&options).await?;
        let preprocessed = audio::load_and_preprocess_wav(audio)?;
        let previous = options.previous_text.as_deref().unwrap_or("").trim();
        let backend_result = if preprocessed.had_speech {
            hybrid::try_transcribe_real(&preprocessed, &options).map_err(AsrError::Backend)?
        } else {
            None
        };
        let text = backend_result
            .as_ref()
            .map(|result| result.text.clone())
            .unwrap_or_else(|| {
                if preprocessed.had_speech {
                    hybrid::transcribe_placeholder(&preprocessed, &options)
                } else {
                    String::new()
                }
            });
        let transcript_text = merge_context(previous, &text);

        Ok(TranscriptionResult {
            model_id: options.model_id,
            text: transcript_text.clone(),
            transcript_text,
            translated_text: None,
            processing_time_seconds: started.elapsed().as_secs_f64(),
            language: backend_result
                .as_ref()
                .and_then(|result| result.language.clone())
                .or(options.language),
            target_language: options
                .target_language
                .filter(|value| !matches!(value.as_str(), "" | "none" | "auto")),
            chunks: backend_result
                .as_ref()
                .map(|result| result.chunks.clone())
                .filter(|chunks| !chunks.is_empty())
                .unwrap_or_else(|| {
                    vec![serde_json::json!({
                        "sample_rate": preprocessed.sample_rate,
                        "original_sample_rate": preprocessed.original_sample_rate,
                        "channels": preprocessed.channels,
                        "duration_seconds": preprocessed.duration_seconds,
                        "rms": preprocessed.rms,
                        "peak": preprocessed.peak,
                        "had_speech": preprocessed.had_speech,
                    })]
                }),
            had_speech: preprocessed.had_speech,
            audio_duration_seconds: preprocessed.duration_seconds,
            input_sample_rate: preprocessed.original_sample_rate,
            input_channels: preprocessed.channels,
            input_rms: preprocessed.rms,
            input_peak: preprocessed.peak,
            accelerator,
        })
    }
}

pub fn merge_context(previous_text: &str, new_text: &str) -> String {
    let previous_text = previous_text.trim();
    let new_text = new_text.trim();

    if new_text.is_empty() {
        return previous_text.to_string();
    }
    if previous_text.is_empty() {
        return new_text.to_string();
    }
    if new_text == previous_text || previous_text.ends_with(new_text) {
        return previous_text.to_string();
    }
    if new_text.starts_with(previous_text) {
        return new_text.to_string();
    }

    format!("{previous_text} {new_text}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn manager_keeps_only_one_model_loaded() {
        let manager = HybridModelManager::new(vec![HardwareBackend::Cpu]);

        let mut options = TranscriptionOptions {
            model_id: "whisper-tiny".to_string(),
            accelerator: AcceleratorPreference::Gpu,
            ..Default::default()
        };
        manager
            .transcribe(&test_wav(&[10_000, -10_000]), options.clone())
            .await
            .unwrap();

        options.model_id = "qwen3-asr-0.6b".to_string();
        manager
            .transcribe(&test_wav(&[10_000, -10_000]), options)
            .await
            .unwrap();

        let status = manager.status().await;
        assert_eq!(status.loaded_model_id.as_deref(), Some("qwen3-asr-0.6b"));
        assert_eq!(status.loaded_backend, Some(HardwareBackend::Cpu));
    }

    #[test]
    fn context_merge_avoids_duplicate_growth() {
        assert_eq!(merge_context("hello world", "world"), "hello world");
        assert_eq!(merge_context("hello", "hello world"), "hello world");
        assert_eq!(merge_context("hello", "world"), "hello world");
    }

    #[tokio::test]
    async fn silent_audio_returns_no_speech_without_text() {
        let manager = HybridModelManager::new(vec![HardwareBackend::Cpu]);

        let result = manager
            .transcribe(&test_wav(&[0, 1, -1, 0]), TranscriptionOptions::default())
            .await
            .unwrap();

        assert!(!result.had_speech);
        assert_eq!(result.text, "");
        assert!(result.input_rms < 0.006);
    }

    fn test_wav(samples: &[i16]) -> Vec<u8> {
        let data_size = samples.len() * 2;
        let mut wav = Vec::with_capacity(44 + data_size);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_size as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16_u32.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&16_000_u32.to_le_bytes());
        wav.extend_from_slice(&(16_000_u32 * 2).to_le_bytes());
        wav.extend_from_slice(&2_u16.to_le_bytes());
        wav.extend_from_slice(&16_u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data_size as u32).to_le_bytes());
        for sample in samples {
            wav.extend_from_slice(&sample.to_le_bytes());
        }
        wav
    }
}
