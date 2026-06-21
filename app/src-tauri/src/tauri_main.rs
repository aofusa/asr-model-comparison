#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]
#![cfg(feature = "desktop")]

use amcp_tauri::accelerator::{AcceleratorPreference, HardwareBackend};
use amcp_tauri::asr::HybridModelManager;
use amcp_tauri::config::{normalize_cli_args, Cli, Command, ServerArgs};
use amcp_tauri::server::{default_available_backends, spawn_embedded_server};
use amcp_tauri::{server, validation, AppConfig};
use clap::Parser;
use std::ffi::OsString;
use std::sync::Arc;
use tauri::{WebviewUrl, WebviewWindowBuilder};

const EMBEDDED_SERVER_PORT: u16 = 8765;

struct DesktopState {
    manager: Arc<HybridModelManager>,
    api_base_url: String,
}

#[tauri::command]
async fn backend_status(
    state: tauri::State<'_, DesktopState>,
) -> Result<serde_json::Value, String> {
    let mut status = serde_json::json!(state.manager.status().await);
    status["api_base_url"] = serde_json::json!(state.api_base_url);
    Ok(status)
}

#[tauri::command]
async fn accelerator_plan(
    model_id: String,
    preference: Option<AcceleratorPreference>,
) -> Result<serde_json::Value, String> {
    let family = amcp_tauri::models::family_for_model(&model_id)
        .ok_or_else(|| format!("unsupported model: {model_id}"))?;
    let available: Vec<HardwareBackend> = default_available_backends();
    let selected = amcp_tauri::select_accelerator(
        family,
        preference.unwrap_or(AcceleratorPreference::Auto),
        &available,
    );
    Ok(serde_json::json!(selected))
}

fn main() {
    match desktop_command() {
        Some(Command::Server(args)) => {
            run_cli(async move { server::run(AppConfig::from(args)).await });
        }
        Some(Command::Validate(args)) => {
            run_cli(async move { validation::run(args).await });
        }
        None => run_desktop(),
    }
}

fn desktop_command() -> Option<Command> {
    let args = desktop_args_os();
    if args.iter().skip(1).any(|arg| arg == "--server") {
        let server_args = std::iter::once(
            args.first()
                .cloned()
                .unwrap_or_else(|| OsString::from("AMCP.exe")),
        )
        .chain(args.into_iter().skip(1).filter(|arg| arg != "--server"));
        return Some(Command::Server(ServerArgs::parse_from(server_args)));
    }

    Cli::parse_from(normalize_cli_args(args)).command
}

#[cfg(not(target_os = "windows"))]
fn desktop_args_os() -> Vec<OsString> {
    std::env::args_os().collect()
}

#[cfg(target_os = "windows")]
fn desktop_args_os() -> Vec<OsString> {
    use std::os::windows::ffi::OsStringExt;

    #[link(name = "kernel32")]
    extern "system" {
        fn GetCommandLineW() -> *const u16;
        fn LocalFree(hmem: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
    }

    #[link(name = "shell32")]
    extern "system" {
        fn CommandLineToArgvW(
            lp_cmd_line: *const u16,
            p_num_args: *mut i32,
        ) -> *mut *mut u16;
    }

    unsafe {
        let command_line = GetCommandLineW();
        if command_line.is_null() {
            return std::env::args_os().collect();
        }

        let mut argc = 0i32;
        let argv = CommandLineToArgvW(command_line, &mut argc);
        if argv.is_null() || argc <= 0 {
            return std::env::args_os().collect();
        }

        let args = (0..argc)
            .filter_map(|index| {
                let ptr = *argv.add(index as usize);
                if ptr.is_null() {
                    return None;
                }

                let mut len = 0usize;
                while *ptr.add(len) != 0 {
                    len += 1;
                }
                Some(OsString::from_wide(std::slice::from_raw_parts(ptr, len)))
            })
            .collect::<Vec<_>>();
        let _ = LocalFree(argv.cast());

        if args.is_empty() {
            std::env::args_os().collect()
        } else {
            args
        }
    }
}

fn run_cli<F>(future: F)
where
    F: std::future::Future<Output = anyhow::Result<()>>,
{
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    tokio::runtime::Runtime::new()
        .expect("failed to create AMCP CLI runtime")
        .block_on(future)
        .expect("AMCP CLI command failed");
}

fn run_desktop() {
    let embedded_addr = spawn_embedded_server(EMBEDDED_SERVER_PORT, AcceleratorPreference::Auto)
        .expect("failed to start embedded AMCP Rust API server");
    let api_base_url = format!("http://{embedded_addr}");

    tauri::Builder::default()
        .manage(DesktopState {
            manager: Arc::new(HybridModelManager::new(default_available_backends())),
            api_base_url,
        })
        .invoke_handler(tauri::generate_handler![backend_status, accelerator_plan])
        .setup(|app| {
            WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
                .title("ASR Model Comparison Platform")
                .inner_size(1240.0, 900.0)
                .resizable(true)
                .build()?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running AMCP desktop app");
}
