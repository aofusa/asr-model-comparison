use crate::accelerator::AcceleratorPreference;
use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use std::net::{IpAddr, Ipv4Addr};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunMode {
    Desktop,
    Server,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub mode: RunMode,
    pub host: IpAddr,
    pub port: u16,
    pub accelerator: AcceleratorPreference,
    pub static_dir: Option<PathBuf>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            mode: RunMode::Desktop,
            host: IpAddr::V4(Ipv4Addr::LOCALHOST),
            port: 8000,
            accelerator: AcceleratorPreference::Auto,
            static_dir: None,
        }
    }
}

#[derive(Debug, Parser)]
#[command(name = "amcp")]
#[command(about = "Rust/Tauri ASR Model Comparison Platform")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Server(ServerArgs),
    Validate(ValidateArgs),
}

#[derive(Debug, Clone, Parser)]
pub struct ServerArgs {
    #[arg(long, default_value = "127.0.0.1")]
    pub host: IpAddr,
    #[arg(long, default_value_t = 8000)]
    pub port: u16,
    #[arg(long, value_enum, default_value_t = AcceleratorArg::Auto)]
    pub accelerator: AcceleratorArg,
    #[arg(long)]
    pub static_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Parser)]
pub struct ValidateArgs {
    #[arg(long)]
    pub audio: Option<PathBuf>,
    #[arg(long, default_value = "whisper-tiny")]
    pub model_id: String,
    #[arg(long, default_value = "auto")]
    pub language: String,
    #[arg(long)]
    pub target_language: Option<String>,
    #[arg(long)]
    pub expected_text: Option<String>,
    #[arg(long, value_enum, default_value_t = AcceleratorArg::Auto)]
    pub accelerator: AcceleratorArg,
    #[arg(long)]
    pub json: bool,
    #[arg(long)]
    pub diagnostics_only: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum AcceleratorArg {
    Auto,
    Gpu,
    Cpu,
}

impl From<AcceleratorArg> for AcceleratorPreference {
    fn from(value: AcceleratorArg) -> Self {
        match value {
            AcceleratorArg::Auto => Self::Auto,
            AcceleratorArg::Gpu => Self::Gpu,
            AcceleratorArg::Cpu => Self::Cpu,
        }
    }
}

impl From<ServerArgs> for AppConfig {
    fn from(args: ServerArgs) -> Self {
        Self {
            mode: RunMode::Server,
            host: args.host,
            port: args.port,
            accelerator: args.accelerator.into(),
            static_dir: args.static_dir,
        }
    }
}
