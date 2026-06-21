use amcp_tauri::config::{Cli, Command};
use amcp_tauri::{server, AppConfig};
use clap::Parser;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    let config = match cli.command {
        Some(Command::Server(args)) => AppConfig::from(args),
        None => AppConfig::default(),
    };

    server::run(config).await
}
