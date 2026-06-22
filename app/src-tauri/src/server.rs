use crate::accelerator::{detect_available_backends, AcceleratorPreference, HardwareBackend};
use crate::asr::{HybridModelManager, SharedModelManager, TranscriptionOptions};
use crate::config::AppConfig;
use crate::models::available_models;
use axum::extract::ws::{Message, WebSocket};
use axum::extract::{Multipart, State, WebSocketUpgrade};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::json;
use std::net::SocketAddr;
use std::net::{IpAddr, Ipv4Addr};
use std::path::PathBuf;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;

#[derive(Clone)]
pub struct AppState {
    pub manager: SharedModelManager,
    pub accelerator: AcceleratorPreference,
    pub static_dir: Option<PathBuf>,
}

pub fn default_available_backends() -> Vec<HardwareBackend> {
    detect_available_backends()
}

pub fn router(config: AppConfig) -> Router {
    tracing::info!(
        mode = ?config.mode,
        accelerator = %config.accelerator,
        static_dir = ?config.static_dir,
        "building AMCP API router"
    );
    let state = AppState {
        manager: Arc::new(HybridModelManager::new(default_available_backends())),
        accelerator: config.accelerator,
        static_dir: config.static_dir.clone(),
    };

    let api = Router::new()
        .route("/health", get(health))
        .route("/api/models", get(models))
        .route("/api/status", get(status))
        .route("/api/transcribe", post(transcribe))
        .route("/api/ws/transcribe", get(ws_transcribe))
        .with_state(state.clone());

    let app = api
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http());

    if let Some(static_dir) = config.static_dir {
        tracing::info!(static_dir = %static_dir.display(), "serving frontend static files");
        let index_file = static_dir.join("index.html");
        app.fallback_service(ServeDir::new(static_dir).fallback(ServeFile::new(index_file)))
    } else {
        app
    }
}

pub async fn run(config: AppConfig) -> anyhow::Result<()> {
    let addr = SocketAddr::new(config.host, config.port);
    tracing::info!(
        mode = ?config.mode,
        host = %config.host,
        port = config.port,
        accelerator = %config.accelerator,
        static_dir = ?config.static_dir,
        available_backends = ?default_available_backends(),
        "starting AMCP Rust backend"
    );
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("AMCP Rust server listening on http://{addr}");
    axum::serve(listener, router(config)).await?;
    Ok(())
}

pub fn spawn_embedded_server(
    port: u16,
    accelerator: AcceleratorPreference,
) -> anyhow::Result<SocketAddr> {
    let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);
    tracing::info!(
        %addr,
        accelerator = %accelerator,
        "spawning embedded AMCP Rust backend"
    );
    std::thread::Builder::new()
        .name("amcp-embedded-server".to_string())
        .spawn(move || {
            let runtime = tokio::runtime::Runtime::new()
                .expect("failed to create embedded AMCP server runtime");
            runtime.block_on(async move {
                if let Err(error) = run(AppConfig {
                    mode: crate::config::RunMode::Desktop,
                    host: IpAddr::V4(Ipv4Addr::LOCALHOST),
                    port,
                    accelerator,
                    static_dir: None,
                })
                .await
                {
                    tracing::error!("embedded AMCP server failed: {error}");
                }
            });
        })?;
    Ok(addr)
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({ "status": "ok", "service": "amcp-rust-backend" }))
}

async fn models() -> Json<serde_json::Value> {
    Json(json!(available_models()))
}

async fn status(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(json!(state.manager.status().await))
}

async fn transcribe(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<serde_json::Value>, AppError> {
    let mut audio: Option<Vec<u8>> = None;
    let mut options = TranscriptionOptions {
        accelerator: state.accelerator,
        ..Default::default()
    };

    while let Some(field) = multipart.next_field().await.map_err(multipart_error)? {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "audio" => {
                audio = Some(field.bytes().await.map_err(multipart_error)?.to_vec());
                if let Some(audio) = audio.as_ref() {
                    tracing::info!(
                        bytes = audio.len(),
                        "received HTTP transcription audio payload"
                    );
                }
            }
            "model_id" => options.model_id = field.text().await.map_err(multipart_error)?,
            "language" => options.language = Some(field.text().await.map_err(multipart_error)?),
            "target_language" => {
                options.target_language = Some(field.text().await.map_err(multipart_error)?)
            }
            "beam_size" => {
                options.beam_size = field.text().await.map_err(multipart_error)?.parse().ok()
            }
            "accelerator" | "hardware_accelerator" => {
                options.accelerator = field
                    .text()
                    .await
                    .map_err(multipart_error)?
                    .parse()
                    .map_err(AppError::InvalidAccelerator)?;
            }
            _ => {}
        }
    }

    let audio =
        audio.ok_or_else(|| AppError::InvalidInput("No audio file provided".to_string()))?;
    tracing::info!(
        model_id = %options.model_id,
        language = ?options.language,
        target_language = ?options.target_language,
        accelerator = %options.accelerator,
        bytes = audio.len(),
        "starting HTTP transcription"
    );
    let result = state.manager.transcribe(&audio, options).await?;
    tracing::info!(
        model_id = %result.model_id,
        backend = ?result.runtime_backend,
        accelerator = %result.accelerator.selected,
        duration_seconds = result.audio_duration_seconds,
        processing_seconds = result.processing_time_seconds,
        had_speech = result.had_speech,
        transcript_chars = result.transcript_text.chars().count(),
        translated_chars = result
            .translated_text
            .as_ref()
            .map(|text| text.chars().count())
            .unwrap_or(0),
        "finished HTTP transcription"
    );
    Ok(Json(json!(result)))
}

async fn ws_transcribe(State(state): State<AppState>, ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

#[derive(Debug, Deserialize)]
struct WsConfig {
    model_id: Option<String>,
    language: Option<String>,
    target_language: Option<String>,
    beam_size: Option<u8>,
    temperature: Option<f32>,
    repetition_penalty: Option<f32>,
    previous_text: Option<String>,
    accelerator: Option<AcceleratorPreference>,
    hardware_accelerator: Option<AcceleratorPreference>,
}

async fn handle_socket(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();
    tracing::info!(
        accelerator = %state.accelerator,
        "WebSocket transcription client connected"
    );
    let mut options = TranscriptionOptions {
        accelerator: state.accelerator,
        ..Default::default()
    };
    let mut chunk_index = 0_u64;
    let mut accumulated_text = String::new();

    while let Some(Ok(message)) = receiver.next().await {
        match message {
            Message::Text(text) => {
                if let Ok(config) = serde_json::from_str::<WsConfig>(&text) {
                    tracing::info!(
                        model_id = ?config.model_id,
                        language = ?config.language,
                        target_language = ?config.target_language,
                        accelerator = ?config.hardware_accelerator.or(config.accelerator),
                        previous_text_chars = config
                            .previous_text
                            .as_ref()
                            .map(|text| text.chars().count())
                            .unwrap_or(0),
                        "received WebSocket transcription config"
                    );
                    if let Some(model_id) = config.model_id {
                        options.model_id = model_id;
                    }
                    options.language = config.language.or(options.language);
                    options.target_language = config.target_language;
                    options.beam_size = config.beam_size.or(options.beam_size);
                    options.temperature = config.temperature.or(options.temperature);
                    options.repetition_penalty =
                        config.repetition_penalty.or(options.repetition_penalty);
                    options.previous_text = config.previous_text;
                    options.accelerator = config
                        .hardware_accelerator
                        .or(config.accelerator)
                        .unwrap_or(state.accelerator);

                    match state.manager.prepare_model(&options).await {
                        Ok((_accelerator, progress)) => {
                            for event in progress {
                                tracing::info!(
                                    model_id = %event.model_id,
                                    phase = %event.phase,
                                    progress = ?event.progress,
                                    elapsed_seconds = ?event.elapsed_seconds,
                                    message = %event.message,
                                    "model preparation progress"
                                );
                                if sender
                                    .send(Message::Text(json!(event).to_string().into()))
                                    .await
                                    .is_err()
                                {
                                    return;
                                }
                            }
                            let ready = json!({
                                "type": "ready",
                                "model_id": options.model_id,
                                "accelerator": options.accelerator,
                                "runtime_backend": state
                                    .manager
                                    .status()
                                    .await
                                    .runtime_backends
                                    .into_iter()
                                    .find(|backend| backend.model_id == options.model_id)
                            });
                            if sender
                                .send(Message::Text(ready.to_string().into()))
                                .await
                                .is_err()
                            {
                                return;
                            }
                            tracing::info!(
                                model_id = %options.model_id,
                                accelerator = %options.accelerator,
                                "WebSocket transcription model ready"
                            );
                        }
                        Err(error) => {
                            tracing::error!(
                                model_id = %options.model_id,
                                error = %error,
                                "failed to prepare WebSocket transcription model"
                            );
                            let _ = sender
                                .send(Message::Text(
                                    json!({ "type": "error", "message": error.to_string() })
                                        .to_string()
                                        .into(),
                                ))
                                .await;
                            return;
                        }
                    }
                }
            }
            Message::Binary(audio) => {
                chunk_index += 1;
                tracing::info!(
                    chunk_index,
                    bytes = audio.len(),
                    model_id = %options.model_id,
                    language = ?options.language,
                    target_language = ?options.target_language,
                    previous_text_chars = accumulated_text.chars().count(),
                    "received WebSocket audio chunk"
                );
                let mut chunk_options = options.clone();
                chunk_options.previous_text = Some(accumulated_text.clone());
                match state.manager.transcribe(&audio, chunk_options).await {
                    Ok(result) => {
                        accumulated_text = result.transcript_text.clone();
                        tracing::info!(
                            chunk_index,
                            model_id = %result.model_id,
                            backend = ?result.runtime_backend,
                            accelerator = %result.accelerator.selected,
                            duration_seconds = result.audio_duration_seconds,
                            processing_seconds = result.processing_time_seconds,
                            had_speech = result.had_speech,
                            transcript_chars = result.transcript_text.chars().count(),
                            translated_chars = result
                                .translated_text
                                .as_ref()
                                .map(|text| text.chars().count())
                                .unwrap_or(0),
                            "finished WebSocket audio chunk"
                        );
                        let response = json!({
                            "type": "transcription",
                            "model_id": result.model_id,
                            "text": result.text,
                            "transcript_text": result.transcript_text,
                            "translated_text": result.translated_text,
                            "is_final": false,
                            "chunks": result.chunks,
                            "processing_time_seconds": result.processing_time_seconds,
                            "had_speech": result.had_speech,
                            "audio_duration_seconds": result.audio_duration_seconds,
                            "input_sample_rate": result.input_sample_rate,
                            "input_channels": result.input_channels,
                            "input_rms": result.input_rms,
                            "input_peak": result.input_peak,
                            "translation_engine": result.translation_engine,
                            "translation_note": result.translation_note,
                            "runtime_backend": result.runtime_backend,
                            "chunk_index": chunk_index,
                            "chunk_size_bytes": audio.len(),
                            "accumulated_text": accumulated_text,
                            "accumulated_transcript_text": accumulated_text,
                            "accumulated_translated_text": null,
                            "target_language": result.target_language,
                            "accelerator": result.accelerator,
                        });
                        if sender
                            .send(Message::Text(response.to_string().into()))
                            .await
                            .is_err()
                        {
                            return;
                        }
                    }
                    Err(error) => {
                        tracing::error!(
                            chunk_index,
                            model_id = %options.model_id,
                            error = %error,
                            "failed to process WebSocket audio chunk"
                        );
                        let _ = sender
                            .send(Message::Text(
                                json!({ "type": "error", "message": error.to_string() })
                                    .to_string()
                                    .into(),
                            ))
                            .await;
                        return;
                    }
                }
            }
            Message::Close(_) => {
                tracing::info!(
                    chunks = chunk_index,
                    "WebSocket transcription client closed"
                );
                return;
            }
            _ => {}
        }
    }
    tracing::info!(
        chunks = chunk_index,
        "WebSocket transcription client disconnected"
    );
}

#[derive(Debug)]
enum AppError {
    BadRequest(String),
    InvalidInput(String),
    InvalidAccelerator(String),
    Asr(crate::asr::AsrError),
}

fn multipart_error(error: axum::extract::multipart::MultipartError) -> AppError {
    AppError::BadRequest(error.to_string())
}

impl From<crate::asr::AsrError> for AppError {
    fn from(value: crate::asr::AsrError) -> Self {
        Self::Asr(value)
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            Self::BadRequest(message) => (StatusCode::BAD_REQUEST, message),
            Self::InvalidInput(message) | Self::InvalidAccelerator(message) => {
                (StatusCode::BAD_REQUEST, message)
            }
            Self::Asr(crate::asr::AsrError::UnsupportedModel(message)) => {
                (StatusCode::BAD_REQUEST, message)
            }
            Self::Asr(error) => (StatusCode::SERVICE_UNAVAILABLE, error.to_string()),
        };

        (status, Json(json!({ "detail": message }))).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RunMode;
    use axum::body::to_bytes;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    #[tokio::test]
    async fn models_endpoint_keeps_compatible_shape() {
        let app = router(AppConfig {
            mode: RunMode::Server,
            static_dir: None,
            ..Default::default()
        });

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let models: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(models
            .as_array()
            .unwrap()
            .iter()
            .any(|model| model["id"] == "whisper-tiny"));
    }

    #[tokio::test]
    async fn status_endpoint_reports_rust_service() {
        let app = router(AppConfig {
            mode: RunMode::Server,
            static_dir: None,
            ..Default::default()
        });

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let status: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(status["service"], "amcp-rust-backend");
    }
}
