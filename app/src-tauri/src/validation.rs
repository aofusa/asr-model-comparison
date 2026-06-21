use crate::asr::{HybridModelManager, TranscriptionOptions};
use crate::config::ValidateArgs;
use crate::server::default_available_backends;
use serde::Serialize;
use std::time::Instant;

#[derive(Debug, Serialize)]
pub struct ValidationReport {
    pub model_id: String,
    pub audio_path: String,
    pub text: String,
    pub transcript_text: String,
    pub translated_text: Option<String>,
    pub language: Option<String>,
    pub target_language: Option<String>,
    pub runtime_backend: String,
    pub accelerator: String,
    pub audio_duration_seconds: f64,
    pub processing_time_seconds: f64,
    pub wall_time_seconds: f64,
    pub realtime_factor: Option<f64>,
    pub expected_text: Option<String>,
    pub character_error_rate: Option<f64>,
}

pub async fn run(args: ValidateArgs) -> anyhow::Result<()> {
    let manager = HybridModelManager::new(default_available_backends());
    if args.diagnostics_only {
        let status = manager.status().await;
        if args.json {
            println!("{}", serde_json::to_string_pretty(&status)?);
        } else {
            println!("Service: {}", status.service);
            println!("Available backends: {:?}", status.available_backends);
            println!("Translation: {}", status.translation.reason);
            for backend in status.runtime_backends {
                println!(
                    "{}: {:?} real={} ({})",
                    backend.model_id,
                    backend.backend,
                    backend.real_inference_available,
                    backend.reason
                );
                for artifact in backend.artifacts {
                    println!(
                        "  - {} {:?} exists={} path={:?} env={:?}",
                        artifact.name,
                        artifact.kind,
                        artifact.exists,
                        artifact.path,
                        artifact.env_var
                    );
                }
            }
        }
        return Ok(());
    }

    let audio_path = args
        .audio
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--audio is required unless --diagnostics-only is set"))?;
    let audio = std::fs::read(&audio_path)?;
    let options = TranscriptionOptions {
        model_id: args.model_id.clone(),
        language: Some(args.language.clone()),
        target_language: args.target_language.clone(),
        accelerator: args.accelerator.into(),
        ..Default::default()
    };

    let started = Instant::now();
    let result = manager.transcribe(&audio, options).await?;
    let wall_time_seconds = started.elapsed().as_secs_f64();
    let realtime_factor = if result.audio_duration_seconds > 0.0 {
        Some(wall_time_seconds / result.audio_duration_seconds)
    } else {
        None
    };
    let character_error_rate = args
        .expected_text
        .as_deref()
        .map(|expected| character_error_rate(expected, &result.transcript_text));

    let report = ValidationReport {
        model_id: result.model_id,
        audio_path: audio_path.to_string_lossy().to_string(),
        text: result.text,
        transcript_text: result.transcript_text,
        translated_text: result.translated_text,
        language: result.language,
        target_language: result.target_language,
        runtime_backend: format!("{:?}", result.runtime_backend),
        accelerator: result.accelerator.selected.to_string(),
        audio_duration_seconds: result.audio_duration_seconds,
        processing_time_seconds: result.processing_time_seconds,
        wall_time_seconds,
        realtime_factor,
        expected_text: args.expected_text,
        character_error_rate,
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("Model: {}", report.model_id);
        println!("Backend: {}", report.runtime_backend);
        println!("Accelerator: {}", report.accelerator);
        println!("Audio duration: {:.3}s", report.audio_duration_seconds);
        println!("Wall time: {:.3}s", report.wall_time_seconds);
        if let Some(realtime_factor) = report.realtime_factor {
            println!("Realtime factor: {:.3}x", realtime_factor);
        }
        if let Some(cer) = report.character_error_rate {
            println!("Character error rate: {:.4}", cer);
        }
        println!("Transcript: {}", report.transcript_text);
        if let Some(translated) = report.translated_text.as_deref() {
            println!("Translation: {translated}");
        }
    }

    Ok(())
}

fn character_error_rate(expected: &str, actual: &str) -> f64 {
    let expected = expected.chars().collect::<Vec<_>>();
    let actual = actual.chars().collect::<Vec<_>>();
    if expected.is_empty() {
        return if actual.is_empty() { 0.0 } else { 1.0 };
    }
    levenshtein_distance(&expected, &actual) as f64 / expected.len() as f64
}

fn levenshtein_distance(expected: &[char], actual: &[char]) -> usize {
    let mut previous = (0..=actual.len()).collect::<Vec<_>>();
    let mut current = vec![0usize; actual.len() + 1];

    for (i, expected_char) in expected.iter().enumerate() {
        current[0] = i + 1;
        for (j, actual_char) in actual.iter().enumerate() {
            let substitution_cost = usize::from(expected_char != actual_char);
            current[j + 1] = (previous[j + 1] + 1)
                .min(current[j] + 1)
                .min(previous[j] + substitution_cost);
        }
        std::mem::swap(&mut previous, &mut current);
    }

    previous[actual.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn character_error_rate_counts_insertions_deletions_and_substitutions() {
        assert_eq!(character_error_rate("abc", "abc"), 0.0);
        assert!((character_error_rate("abc", "axc") - (1.0 / 3.0)).abs() < f64::EPSILON);
        assert!((character_error_rate("abc", "ab") - (1.0 / 3.0)).abs() < f64::EPSILON);
    }
}
