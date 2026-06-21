use serde::{Deserialize, Serialize};
use std::io::Cursor;
use std::process::Command;
use thiserror::Error;

pub const TARGET_SAMPLE_RATE: u32 = 16_000;
const SPEECH_RMS_THRESHOLD: f32 = 0.006;
const SPEECH_PEAK_THRESHOLD: f32 = 0.03;

#[derive(Debug, Error)]
pub enum AudioError {
    #[error("audio payload is empty")]
    Empty,
    #[error("unsupported audio format: {0}")]
    Unsupported(String),
    #[error("invalid wav data: {0}")]
    InvalidWav(String),
    #[error("decode failed: {0}")]
    Decode(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub original_sample_rate: u32,
    pub channels: u16,
    pub duration_seconds: f64,
    pub rms: f32,
    pub peak: f32,
    pub had_speech: bool,
}

pub fn load_and_preprocess_wav(audio: &[u8]) -> Result<PreprocessedAudio, AudioError> {
    if audio.is_empty() {
        return Err(AudioError::Empty);
    }

    let decoded = decode_audio(audio)?;
    let mono = downmix_to_mono(&decoded.samples, decoded.channels);
    let samples = if decoded.sample_rate == TARGET_SAMPLE_RATE {
        mono
    } else {
        resample_linear(&mono, decoded.sample_rate, TARGET_SAMPLE_RATE)
    };
    let (rms, peak) = pcm_stats(&samples);
    let duration_seconds = if TARGET_SAMPLE_RATE == 0 {
        0.0
    } else {
        samples.len() as f64 / TARGET_SAMPLE_RATE as f64
    };

    Ok(PreprocessedAudio {
        samples,
        sample_rate: TARGET_SAMPLE_RATE,
        original_sample_rate: decoded.sample_rate,
        channels: decoded.channels,
        duration_seconds,
        rms,
        peak,
        had_speech: rms >= SPEECH_RMS_THRESHOLD || peak >= SPEECH_PEAK_THRESHOLD,
    })
}

#[derive(Debug)]
struct DecodedAudio {
    samples: Vec<f32>,
    sample_rate: u32,
    channels: u16,
}

fn decode_audio(audio: &[u8]) -> Result<DecodedAudio, AudioError> {
    if audio.len() >= 12 && &audio[0..4] == b"RIFF" && &audio[8..12] == b"WAVE" {
        return decode_wav(audio);
    }

    decode_with_symphonia(audio).or_else(|symphonia_error| {
        decode_with_ffmpeg(audio).map_err(|ffmpeg_error| {
            AudioError::Unsupported(format!(
                "symphonia failed ({symphonia_error}); ffmpeg fallback failed ({ffmpeg_error})"
            ))
        })
    })
}

fn decode_wav(audio: &[u8]) -> Result<DecodedAudio, AudioError> {
    if audio.len() < 44 {
        return Err(AudioError::InvalidWav("file is too short".to_string()));
    }
    if &audio[0..4] != b"RIFF" || &audio[8..12] != b"WAVE" {
        return Err(AudioError::Unsupported(
            "only RIFF/WAVE PCM input is currently supported".to_string(),
        ));
    }

    let mut offset = 12;
    let mut channels = None;
    let mut sample_rate = None;
    let mut bits_per_sample = None;
    let mut audio_format = None;
    let mut data: Option<&[u8]> = None;

    while offset + 8 <= audio.len() {
        let chunk_id = &audio[offset..offset + 4];
        let chunk_size = u32::from_le_bytes(
            audio[offset + 4..offset + 8]
                .try_into()
                .expect("slice length checked"),
        ) as usize;
        offset += 8;

        if offset + chunk_size > audio.len() {
            return Err(AudioError::InvalidWav(
                "chunk exceeds file length".to_string(),
            ));
        }

        match chunk_id {
            b"fmt " => {
                if chunk_size < 16 {
                    return Err(AudioError::InvalidWav("fmt chunk is too small".to_string()));
                }
                audio_format = Some(u16::from_le_bytes(
                    audio[offset..offset + 2].try_into().expect("fmt checked"),
                ));
                channels = Some(u16::from_le_bytes(
                    audio[offset + 2..offset + 4]
                        .try_into()
                        .expect("fmt checked"),
                ));
                sample_rate = Some(u32::from_le_bytes(
                    audio[offset + 4..offset + 8]
                        .try_into()
                        .expect("fmt checked"),
                ));
                bits_per_sample = Some(u16::from_le_bytes(
                    audio[offset + 14..offset + 16]
                        .try_into()
                        .expect("fmt checked"),
                ));
            }
            b"data" => data = Some(&audio[offset..offset + chunk_size]),
            _ => {}
        }

        offset += chunk_size + (chunk_size % 2);
    }

    let audio_format =
        audio_format.ok_or_else(|| AudioError::InvalidWav("missing fmt chunk".to_string()))?;
    let channels =
        channels.ok_or_else(|| AudioError::InvalidWav("missing channel count".to_string()))?;
    let sample_rate =
        sample_rate.ok_or_else(|| AudioError::InvalidWav("missing sample rate".to_string()))?;
    let bits_per_sample = bits_per_sample
        .ok_or_else(|| AudioError::InvalidWav("missing bits per sample".to_string()))?;
    let data = data.ok_or_else(|| AudioError::InvalidWav("missing data chunk".to_string()))?;

    if channels == 0 {
        return Err(AudioError::InvalidWav("channel count is zero".to_string()));
    }
    if sample_rate == 0 {
        return Err(AudioError::InvalidWav("sample rate is zero".to_string()));
    }

    let samples = match (audio_format, bits_per_sample) {
        (1, 16) => decode_pcm16(data),
        (3, 32) => decode_float32(data),
        (format, bits) => {
            return Err(AudioError::Unsupported(format!(
                "wav format {format} with {bits} bits is not supported"
            )))
        }
    };

    Ok(DecodedAudio {
        samples,
        sample_rate,
        channels,
    })
}

fn decode_with_symphonia(audio: &[u8]) -> Result<DecodedAudio, AudioError> {
    use symphonia::core::codecs::audio::{AudioDecoderOptions, CODEC_ID_NULL_AUDIO};
    use symphonia::core::errors::Error as SymphoniaError;
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;

    let cursor = Cursor::new(audio.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
    let mut format = symphonia::default::get_probe()
        .probe(
            &Hint::new(),
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|error| AudioError::Decode(error.to_string()))?;
    let track = format
        .tracks()
        .iter()
        .find(|track| {
            track
                .codec_params
                .as_ref()
                .and_then(|params| params.audio())
                .map(|params| params.codec != CODEC_ID_NULL_AUDIO)
                .unwrap_or(false)
        })
        .ok_or_else(|| AudioError::Decode("no supported audio track found".to_string()))?;
    let track_id = track.id;
    let codec_params = track
        .codec_params
        .as_ref()
        .and_then(|params| params.audio())
        .ok_or_else(|| AudioError::Decode("audio track has no codec params".to_string()))?
        .clone();
    let sample_rate = codec_params
        .sample_rate
        .ok_or_else(|| AudioError::Decode("audio track has no sample rate".to_string()))?;
    let channels = codec_params
        .channels
        .as_ref()
        .ok_or_else(|| AudioError::Decode("audio track has no channel layout".to_string()))?
        .count() as u16;
    let decoder_info = symphonia::default::get_codecs()
        .get_audio_decoder(codec_params.codec)
        .ok_or_else(|| {
            AudioError::Decode(format!("unsupported audio codec {:?}", codec_params.codec))
        })?;
    let mut decoder = (decoder_info.factory)(&codec_params, &AudioDecoderOptions::default())
        .map_err(|error| AudioError::Decode(error.to_string()))?;

    let mut samples = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(SymphoniaError::IoError(_)) => break,
            Err(error) => return Err(AudioError::Decode(error.to_string())),
        };
        if packet.track_id != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                decoded.copy_to_vec_interleaved::<f32>(&mut samples);
            }
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(SymphoniaError::ResetRequired) => {
                return Err(AudioError::Decode("decoder reset is required".to_string()));
            }
            Err(error) => return Err(AudioError::Decode(error.to_string())),
        }
    }

    if samples.is_empty() {
        return Err(AudioError::Decode(
            "decoded audio contains no samples".to_string(),
        ));
    }

    Ok(DecodedAudio {
        samples,
        sample_rate,
        channels,
    })
}

fn decode_with_ffmpeg(audio: &[u8]) -> Result<DecodedAudio, AudioError> {
    let stem = format!(
        "amcp-audio-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or_default()
    );
    let input_path = std::env::temp_dir().join(format!("{stem}.input"));
    let output_path = std::env::temp_dir().join(format!("{stem}.f32le"));

    std::fs::write(&input_path, audio)
        .map_err(|error| AudioError::Decode(format!("failed to write temp input: {error}")))?;

    let status = Command::new("ffmpeg")
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-y")
        .arg("-i")
        .arg(&input_path)
        .arg("-f")
        .arg("f32le")
        .arg("-acodec")
        .arg("pcm_f32le")
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg(TARGET_SAMPLE_RATE.to_string())
        .arg(&output_path)
        .status()
        .map_err(|error| AudioError::Decode(format!("failed to run ffmpeg: {error}")))?;

    let result = if status.success() {
        let bytes = std::fs::read(&output_path).map_err(|error| {
            AudioError::Decode(format!("failed to read ffmpeg output: {error}"))
        })?;
        let samples = decode_float32(&bytes);
        if samples.is_empty() {
            Err(AudioError::Decode("ffmpeg produced no samples".to_string()))
        } else {
            Ok(DecodedAudio {
                samples,
                sample_rate: TARGET_SAMPLE_RATE,
                channels: 1,
            })
        }
    } else {
        Err(AudioError::Decode(format!("ffmpeg exited with {status}")))
    };

    let _ = std::fs::remove_file(&input_path);
    let _ = std::fs::remove_file(&output_path);

    result
}

fn decode_pcm16(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / i16::MAX as f32
        })
        .collect()
}

fn decode_float32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]).clamp(-1.0, 1.0))
        .collect()
}

fn downmix_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }

    let channel_count = channels as usize;
    samples
        .chunks_exact(channel_count)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

fn resample_linear(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if samples.is_empty() || from_rate == to_rate {
        return samples.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = ((samples.len() as f64 / ratio).round() as usize).max(1);
    let mut output = Vec::with_capacity(output_len);

    for output_index in 0..output_len {
        let source_position = output_index as f64 * ratio;
        let left = source_position.floor() as usize;
        let right = (left + 1).min(samples.len() - 1);
        let fraction = (source_position - left as f64) as f32;
        output.push(samples[left] * (1.0 - fraction) + samples[right] * fraction);
    }

    output
}

fn pcm_stats(samples: &[f32]) -> (f32, f32) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }

    let mut sum_squares = 0.0_f32;
    let mut peak = 0.0_f32;
    for sample in samples {
        let abs = sample.abs();
        peak = peak.max(abs);
        sum_squares += sample * sample;
    }

    ((sum_squares / samples.len() as f32).sqrt(), peak)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_pcm16_wav_and_detects_speech() {
        let wav = test_wav(16_000, 1, &[0, 10_000, -10_000, 0]);

        let audio = load_and_preprocess_wav(&wav).unwrap();

        assert_eq!(audio.sample_rate, TARGET_SAMPLE_RATE);
        assert_eq!(audio.original_sample_rate, 16_000);
        assert_eq!(audio.channels, 1);
        assert!(audio.had_speech);
        assert!(audio.peak > 0.2);
    }

    #[test]
    fn resamples_to_16khz_and_downmixes() {
        let wav = test_wav(8_000, 2, &[10_000, 10_000, -10_000, -10_000]);

        let audio = load_and_preprocess_wav(&wav).unwrap();

        assert_eq!(audio.sample_rate, TARGET_SAMPLE_RATE);
        assert_eq!(audio.original_sample_rate, 8_000);
        assert_eq!(audio.channels, 2);
        assert_eq!(audio.samples.len(), 4);
    }

    #[test]
    fn marks_low_energy_audio_as_silence() {
        let wav = test_wav(16_000, 1, &[0, 1, -1, 0]);

        let audio = load_and_preprocess_wav(&wav).unwrap();

        assert!(!audio.had_speech);
        assert!(audio.rms < SPEECH_RMS_THRESHOLD);
        assert!(audio.peak < SPEECH_PEAK_THRESHOLD);
    }

    pub fn test_wav(sample_rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
        let data_size = samples.len() * 2;
        let mut wav = Vec::with_capacity(44 + data_size);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_size as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16_u32.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&channels.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * channels as u32 * 2).to_le_bytes());
        wav.extend_from_slice(&(channels * 2).to_le_bytes());
        wav.extend_from_slice(&16_u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data_size as u32).to_le_bytes());
        for sample in samples {
            wav.extend_from_slice(&sample.to_le_bytes());
        }
        wav
    }
}
