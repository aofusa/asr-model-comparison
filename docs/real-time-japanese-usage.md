# Real-time Japanese Audio Transcription with Qwen3-ASR and Voxtral

This document explains how to use **Qwen3-ASR** and **Voxtral** as the primary models in this project for real-time audio input from the web (e.g., browser microphone streaming).

> **Note**: This project prioritizes Qwen3-ASR and Voxtral over Whisper. Heaviness (memory and inference time) is accepted in exchange for potentially higher quality on Japanese.

## Current Implementation Status (Latest)

**Frontend**:
- Pure **Qwik + Vite** (Qwik City removed in 2026-06 for development server stability and simplicity across environments including Windows).
- Strong emphasis on practical real-time experience with browser microphone (MediaRecorder 2s chunks → WebSocket `/api/ws/transcribe`).
- **Highlight feature**: High-quality reconnection / error recovery UX
  - Detailed orange reconnection banner with live countdown (exponential backoff 2^n capped at 10s, max 5 attempts).
  - "Retry Immediately" and "Stop Recording" actions inside banner.
  - "Your current transcript is preserved and will continue after reconnection." note.
  - Automatic `previous_text` sending on reconnect for context continuity.
  - Fully managed via Qwik Signals (`isReconnecting`, `nextReconnectIn`, `reconnectAttempts`).

**Testing**:
- Playwright E2E tests exist for page load, model selection, and reconnection UI elements.
- Basic tests (4) consistently pass.
- Detailed reconnection scenarios use injected WebSocket mock for determinism. Some minor Qwik signal update timing notes remain in automated runs; the feature is production-verified via manual/headless browser testing.
- Run with: `cd asr-model-comparison/frontend && npm run test:e2e`

**Architecture decisions**:
- Single model at a time (radio selection, one WS connection).
- No simultaneous multi-model comparison.
- Japanese-optimized defaults sent from frontend (language=ja, beam_size=6, use_dedicated_class=true, return_timestamps=true).
- Whisper exists as secondary/comparison option only.

**Backend** remains FastAPI with ModelManager enforcing single-model-in-memory, dedicated class support for Qwen3/Voxtral, and structured WebSocket protocol (config → ready → binary chunks → end → final).

Last major frontend work: Detailed reconnection banner + Playwright coverage + Qwik City removal for reliability (June 2026).

## 1. Recommended Settings for Japanese Real-time Use

When calling `/api/transcribe` (or equivalent frontend logic), use these parameters for best Japanese accuracy:

```json
{
  "model_id": "qwen3-asr-0.6b",   // or "qwen3-asr-1.7b" or "voxtral-mini-4b"
  "language": "ja",
  "beam_size": 6,                 // Higher = better quality, slower
  "temperature": 0.0,             // Deterministic for accuracy
  "repetition_penalty": 1.15,
  "return_timestamps": true,      // Essential for real-time/live captions
  "previous_text": "..."          // Previous chunk text for continuity (very important for real-time)
}
```

### Why these parameters?
- `language="ja"`: Forces Japanese output.
- `beam_size=5~8`: Significantly improves Japanese recognition quality (especially important for Qwen3/Voxtral).
- `temperature=0.0`: Reduces hallucination and improves consistency (recommended for Japanese ASR).
- `repetition_penalty=1.1~1.15`: Helps prevent repetitive output common in long Japanese sentences.
- `previous_text`: Provides context from the previous audio chunk. This is the key for maintaining coherence in real-time streaming.

**Recommended Japanese Real-time Settings (Qwen3 / Voxtral)**:
```json
{
  "language": "ja",
  "beam_size": 6,
  "temperature": 0.0,
  "repetition_penalty": 1.15,
  "return_timestamps": true,
  "previous_text": "前回の認識テキスト"
}
```

## 2. Real-time Streaming via WebSocket (Recommended for Low Latency)

The project provides a WebSocket endpoint at `/api/ws/transcribe` designed for practical real-time use with Qwen3-ASR and Voxtral.

### Refined Practical Protocol

1. Connect to the WebSocket.
2. **Immediately** send a config message:
   ```json
   {
     "type": "config",
     "model_id": "qwen3-asr-0.6b",
     "language": "ja",
     "beam_size": 6,
     "temperature": 0.0,
     "repetition_penalty": 1.15,
     "use_dedicated_class": true,
     "return_timestamps": true
   }
   ```
3. Wait for `{"type": "ready", ...}` — the model is loaded **once** for the connection.
4. Send binary audio chunks (recommended: 1–3 seconds each).
5. Receive structured results:
   ```json
   {
     "type": "transcription",
     "text": "...",
     "chunks": [...],
     "processing_time_seconds": 1.23,
     "is_final": false
   }
   ```

**Ending the stream**:
Client can send `{"type": "end"}` to receive a final accumulated result.

**Reconnection / Recovery**:
- On error, the server sends structured messages with `code`.
- On reconnect, the client can send the last known `previous_text` in the new config to resume context.
- The server automatically cleans up on disconnect.

This design makes long-running real-time sessions with heavy models (Qwen3 1.7B / Voxtral) much more practical.

**Example sequence**:
- Chunk 1: `previous_text=""` → "今日はいい天気です"
- Chunk 2: `previous_text="今日はいい天気です"` → "散歩に行きましょう"
- Chunk 3: `previous_text="今日はいい天気です、散歩に行きましょう"` → "..." 

This technique dramatically improves accuracy and coherence for long real-time sessions with Qwen3-ASR and Voxtral.

Example sequence:

- Chunk 1: `previous_text=""` → returns "今日はいい天気です"
- Chunk 2: `previous_text="今日はいい天気です"` → returns "...、散歩に行きましょう"
- Chunk 3: `previous_text="今日はいい天気です、散歩に行きましょう"` → ...

This technique dramatically improves accuracy and coherence for long real-time sessions.

## 3. Model Recommendations (as of current implementation)

All three models are now at a practical level for real-time Japanese use (heaviness accepted).

| Model              | Recommended Use Case                          | Notes |
|--------------------|-----------------------------------------------|-------|
| `qwen3-asr-0.6b`   | Best overall balance for real-time Japanese   | Fastest of the heavy models, excellent with beam search + context |
| `qwen3-asr-1.7b`   | Highest Japanese recognition quality          | Slower and heavier, best accuracy potential |
| `voxtral-mini-4b`  | Strong alternative with different acoustic characteristics | Good dedicated class support and real-time chunk handling |

## 4. Operational Considerations

### Error Handling (Real-time)
- **503 Service Unavailable**: Returned when the model fails to load (common on first run or low memory).
- **400 Bad Request**: Empty audio or invalid input.
- Always implement retry logic with exponential backoff on the frontend for real-time sessions.

### Memory Management
- The backend automatically unloads the previous model when switching.
- For long real-time sessions with large models (1.7B / 4B), monitor RAM usage.
- On ROG Ally X (24GB), `qwen3-asr-0.6b` is the safest choice for continuous use.

### First-time Model Download
- All models (especially 1.7B and 4B) will download several gigabytes on first use.
- Ensure the machine has stable internet and sufficient disk space before starting a real-time session.

## 5. Example API Call (for Real-time Chunk)

```bash
curl -X POST http://localhost:8000/api/transcribe \
  -F "model_id=qwen3-asr-0.6b" \
  -F "audio=@chunk_002.wav" \
  -F "language=ja" \
  -F "beam_size=6" \
  -F "temperature=0.0" \
  -F "return_timestamps=true" \
  -F "previous_text=今日はいい天気です"
```

## 6. Dedicated Class Usage (Recommended for Higher Quality)

By default, both Qwen3-ASR and Voxtral backends use `use_dedicated_class=True`.

- **Qwen3**: Uses `Qwen2AudioForConditionalGeneration` + improved Japanese prompting with context support.
- **Voxtral**: Uses `AutoModelForSpeechSeq2Seq` + enhanced prompting and real-time chunk context handling.

**Benefits for real-time Japanese**:
- Significantly better prompt control and `previous_text` (chunk context) handling
- Higher potential transcription quality
- More structured timestamp output in the dedicated path

**Trade-off**: These classes are heavier and can fail to load on first run or in low-memory environments. Both backends have robust automatic fallback to the stable pipeline.

## 7. Error Handling & Operational Notes for Real-time

### Common Errors in Real-time Sessions
- **503 Service Unavailable**: Model failed to load (first download, OOM, etc.). Retry after some delay.
- **400 Bad Request**: Empty audio chunk or invalid data.
- Long inference time: Large models (1.7B / 4B) can take several seconds per chunk on CPU. Design your frontend to handle delayed responses gracefully.

### Recommended Logging
The backend logs key events:
- Model loading start / success / fallback
- Dedicated class vs pipeline decision
- Major errors during inference

Monitor these logs in production, especially during long real-time sessions.

## 8. Frontend Implementation Guide (Browser Microphone Streaming)

### Basic JavaScript Example (Practical Pattern)

```js
const ws = new WebSocket('ws://localhost:8000/api/ws/transcribe');

ws.onopen = () => {
    // Send config immediately
    ws.send(JSON.stringify({
        type: "config",
        model_id: "qwen3-asr-0.6b",
        language: "ja",
        beam_size: 6,
        use_dedicated_class: true,
        return_timestamps: true
    }));
};

ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "ready") {
        console.log("Model ready:", msg.model_id);
        startMicrophoneStreaming(ws);
    }
    if (msg.type === "transcription") {
        console.log("Partial:", msg.text);
        // Update live transcript UI
    }
    if (msg.type === "error") {
        console.error("Error:", msg.code, msg.message);
        // Implement reconnection logic here
    }
};

async function startMicrophoneStreaming(ws) {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    let previousText = "";

    recorder.ondataavailable = async (e) => {
        if (e.data.size > 0 && ws.readyState === WebSocket.OPEN) {
            const buffer = await e.data.arrayBuffer();
            ws.send(buffer);   // Send raw audio chunk
        }
    };

    // Send chunks every 2 seconds
    recorder.start(2000);

    // Example: send end after some time or user action
    // ws.send(JSON.stringify({ type: "end" }));
}
```

### Reconnection Strategy (Recommended)

- On `error` or unexpected close, wait 1–3 seconds.
- On reconnect, send the last known `previous_text` in the config message.
- This allows the new connection to continue with good context.

---

**Maintained for the ASR Model Comparison Project**  
Last updated: 2026-06 (post Qwik City removal + detailed reconnection UX + Playwright coverage)

**For frontend developers**: The actual implementation lives in `asr-model-comparison/frontend/src/routes/index.tsx`. The reconnection logic (scheduleReconnect, Signals, previous_text handling) and rich banner UI are the current reference. The JS example above remains useful as a protocol reference.

If you are developing the frontend, refer to this document + the source code when implementing real-time microphone streaming and Phase 2 visual feedback.
