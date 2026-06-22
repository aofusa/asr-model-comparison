use amcp_tauri::config::{normalize_cli_args, Cli, Command};
use amcp_tauri::{server, validation, AppConfig};
use clap::Parser;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    amcp_tauri::logging::init_cli();

    let cli = Cli::parse_from(normalize_cli_args(std::env::args_os()));
    match cli.command {
        Some(Command::Server(args)) => {
            let config = AppConfig::from(args);
            tracing::info!(
                host = %config.host,
                port = config.port,
                accelerator = %config.accelerator,
                "starting AMCP CLI server mode"
            );
            server::run(config).await
        }
        Some(Command::Validate(args)) => {
            tracing::info!(
                model_id = %args.model_id,
                language = %args.language,
                target_language = ?args.target_language,
                "starting AMCP validation mode"
            );
            validation::run(args).await
        }
        None => {
            let config = AppConfig::default();
            tracing::info!(
                host = %config.host,
                port = config.port,
                accelerator = %config.accelerator,
                "starting AMCP CLI default server mode"
            );
            server::run(config).await
        }
    }
}
