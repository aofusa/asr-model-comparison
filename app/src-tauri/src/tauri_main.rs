#![cfg(feature = "desktop")]

use amcp_tauri::accelerator::{AcceleratorPreference, HardwareBackend};
use amcp_tauri::asr::HybridModelManager;
use amcp_tauri::server::{default_available_backends, spawn_embedded_server};
use std::sync::Arc;

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
    let embedded_addr = spawn_embedded_server(EMBEDDED_SERVER_PORT, AcceleratorPreference::Auto)
        .expect("failed to start embedded AMCP Rust API server");
    let api_base_url = format!("http://{embedded_addr}");

    tauri::Builder::default()
        .manage(DesktopState {
            manager: Arc::new(HybridModelManager::new(default_available_backends())),
            api_base_url,
        })
        .invoke_handler(tauri::generate_handler![backend_status, accelerator_plan])
        .run(tauri::generate_context!())
        .expect("error while running AMCP desktop app");
}
