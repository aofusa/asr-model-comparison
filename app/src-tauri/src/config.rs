use crate::accelerator::AcceleratorPreference;
use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use std::ffi::OsString;
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

pub fn normalize_cli_args<I>(args: I) -> Vec<OsString>
where
    I: IntoIterator<Item = OsString>,
{
    args.into_iter()
        .map(|arg| {
            if arg == "--server" {
                OsString::from("server")
            } else {
                arg
            }
        })
        .collect()
}

#[derive(Debug, Clone, Parser)]
pub struct ServerArgs {
    #[arg(long, default_value = "0.0.0.0")]
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn normalizes_server_flag_to_server_subcommand() {
        let args = normalize_cli_args([
            OsString::from("AMCP.exe"),
            OsString::from("--server"),
            OsString::from("--port"),
            OsString::from("8787"),
        ]);
        let cli = Cli::parse_from(args);

        match cli.command {
            Some(Command::Server(server_args)) => assert_eq!(server_args.port, 8787),
            _ => panic!("expected server command"),
        }
    }

    #[test]
    fn server_mode_defaults_to_all_interfaces() {
        let args = normalize_cli_args([OsString::from("AMCP.exe"), OsString::from("--server")]);
        let cli = Cli::parse_from(args);

        match cli.command {
            Some(Command::Server(server_args)) => {
                assert_eq!(server_args.host, IpAddr::V4(Ipv4Addr::UNSPECIFIED));
            }
            _ => panic!("expected server command"),
        }
    }

    #[test]
    fn keeps_default_desktop_mode_when_no_command_is_passed() {
        let cli = Cli::parse_from(normalize_cli_args([OsString::from("AMCP.exe")]));

        assert!(cli.command.is_none());
    }
}
