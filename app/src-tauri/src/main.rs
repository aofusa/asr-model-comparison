use amcp_tauri::config::{Cli, Command};
use amcp_tauri::{server, validation, AppConfig};
use clap::Parser;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    match cli.command {
        Some(Command::Server(args)) => server::run(AppConfig::from(args)).await,
        Some(Command::Validate(args)) => validation::run(args).await,
        None => server::run(AppConfig::default()).await,
    }
}
