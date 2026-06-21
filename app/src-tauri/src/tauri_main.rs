#![cfg(feature = "desktop")]

use amcp_tauri::accelerator::{AcceleratorPreference, HardwareBackend};
use amcp_tauri::asr::HybridModelManager;
use amcp_tauri::server::default_available_backends;
use std::sync::Arc;

struct DesktopState {
    manager: Arc<HybridModelManager>,
}

#[tauri::command]
async fn backend_status(
    state: tauri::State<'_, DesktopState>,
) -> Result<serde_json::Value, String> {
    Ok(serde_json::json!(state.manager.status().await))
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
    tauri::Builder::default()
        .manage(DesktopState {
            manager: Arc::new(HybridModelManager::new(default_available_backends())),
        })
        .invoke_handler(tauri::generate_handler![backend_status, accelerator_plan])
        .run(tauri::generate_context!())
        .expect("error while running AMCP desktop app");
}
