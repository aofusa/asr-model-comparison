#![allow(dead_code)]

//! Qwen3-ASR native Candle implementation used by the Rust/Tauri app.
//!
//! This module is compiled only with the `qwen` feature and keeps Qwen
//! inference inside this crate instead of depending on an unofficial ASR
//! wrapper crate. The implementation uses `candle-core`, `candle-nn`,
//! `tokenizers`, `safetensors`, and `rustfft` directly.
//!
//! The initial implementation is adapted from the MIT-licensed
//! `qwen3-asr-rs` crate, then kept in-tree so AMCP controls the runtime
//! boundary and audits the Candle calls directly.

mod config;
mod decoder;
mod encoder;
mod error;
mod hub;
mod inference;
mod linear;
mod mel;

pub use error::{AsrError, Result};
pub use inference::{AsrInference, SpeechTranslationResult, TranscribeOptions, TranscribeResult};
pub use mel::load_audio_wav;
