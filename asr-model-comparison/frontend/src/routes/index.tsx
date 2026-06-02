import { component$, useSignal, useStore, $, useVisibleTask$ } from '@builder.io/qwik';

export default component$(() => {
  const isRecording = useSignal(false);
  const transcript = useSignal('');
  const previousText = useSignal('');
  const selectedModel = useSignal('whisper-tiny');
  const status = useSignal('Idle');
  const reconnectAttempts = useSignal(0);
  const maxReconnectAttempts = 5;
  const isReconnecting = useSignal(false);
  const nextReconnectIn = useSignal(0);

  // Phase 2: Real-time visual feedback
  const volumeLevel = useSignal(0); // 0-100 for meter visualization

  // Phase 2: is_final visual distinction
  const finalTranscript = useSignal('');
  const partialTranscript = useSignal('');

  // Phase 2: Generation settings (sent on connect / reconnect)
  const beamSize = useSignal(6);
  const temperature = useSignal(0.0);
  const repetitionPenalty = useSignal(1.15);
  const useDedicatedClass = useSignal(true);

  // A: localStorage persistence for settings
  const SETTINGS_KEY = 'asr-settings-v1';

  const saveSettings = $(() => {
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify({
        beamSize: beamSize.value,
        temperature: temperature.value,
        repetitionPenalty: repetitionPenalty.value,
        useDedicatedClass: useDedicatedClass.value,
        selectedModel: selectedModel.value,
      }));
    } catch {}
  });

  // Load persisted settings on mount (client only)
  useVisibleTask$(() => {
    try {
      const raw = localStorage.getItem(SETTINGS_KEY);
      if (raw) {
        const saved = JSON.parse(raw);
        if (typeof saved.beamSize === 'number') beamSize.value = saved.beamSize;
        if (typeof saved.temperature === 'number') temperature.value = saved.temperature;
        if (typeof saved.repetitionPenalty === 'number') repetitionPenalty.value = saved.repetitionPenalty;
        if (typeof saved.useDedicatedClass === 'boolean') useDedicatedClass.value = saved.useDedicatedClass;
        if (typeof saved.selectedModel === 'string') selectedModel.value = saved.selectedModel;
      }
    } catch {}
  });

  let mediaRecorder: MediaRecorder | null = null;
  let ws: WebSocket | null = null;
  let reconnectTimeout: any = null;
  let countdownInterval: any = null;

  // Phase 2 audio visualizer nodes
  let audioContext: AudioContext | null = null;
  let analyser: AnalyserNode | null = null;
  let volumeRaf: number | null = null;

  const models = [
    { id: 'whisper-tiny', label: 'Whisper Tiny' },
    { id: 'whisper-small', label: 'Whisper Small' },
    { id: 'whisper-medium', label: 'Whisper Medium' },
    { id: 'whisper-large-v3-turbo', label: 'Whisper Large-v3 Turbo' },
    { id: 'qwen3-asr-0.6b', label: 'Qwen3-ASR 0.6B (Main)' },
    { id: 'qwen3-asr-1.7b', label: 'Qwen3-ASR 1.7B (High Quality)' },
    { id: 'voxtral-mini-4b', label: 'Voxtral Mini 4B' },
  ];

  const clearReconnectTimeout = $(() => {
    if (reconnectTimeout) {
      clearTimeout(reconnectTimeout);
      reconnectTimeout = null;
    }
  });

  const connectWebSocket = $((isReconnect = false) => {
    clearReconnectTimeout();

    if (ws) {
      try { ws.close(); } catch {}
      ws = null;
    }

    ws = new WebSocket('ws://localhost:8000/api/ws/transcribe');

    ws.onopen = () => {
      clearCountdown();
      reconnectAttempts.value = 0;
      isReconnecting.value = false;
      status.value = isReconnect ? 'Reconnected - Resuming context...' : 'Connected';

      const config: any = {
        type: 'config',
        model_id: selectedModel.value,
        language: 'ja',
        beam_size: beamSize.value,
        temperature: temperature.value,
        repetition_penalty: repetitionPenalty.value,
        use_dedicated_class: useDedicatedClass.value,
        return_timestamps: true,
      };

      // Critical for real-time: send accumulated context on reconnect
      if (isReconnect && previousText.value) {
        config.previous_text = previousText.value;
      }

      ws!.send(JSON.stringify(config));
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'ready') {
        status.value = `Ready - ${data.model_id}`;
      }

      if (data.type === 'transcription' && data.text) {
        const isFinal = data.is_final === true;

        if (isFinal) {
          // 確定した結果 → finalTranscript に蓄積
          finalTranscript.value = (finalTranscript.value + ' ' + data.text).trim();
          partialTranscript.value = ''; // partial をクリア
          previousText.value = finalTranscript.value;
          transcript.value = finalTranscript.value; // 後方互換
        } else {
          // 部分結果 → partialTranscript で一時表示
          partialTranscript.value = data.text;
          transcript.value = (finalTranscript.value + ' ' + data.text).trim();
        }
      }

      if (data.type === 'error') {
        status.value = `Error: ${data.message || data.code}`;
        if (isRecording.value) {
          scheduleReconnect();
        }
      }

      if (data.type === 'final') {
        status.value = 'Stream ended';
      }
    };

    ws.onerror = () => {
      status.value = 'Connection error';
      if (isRecording.value) {
        scheduleReconnect();
      }
    };

    ws.onclose = () => {
      const wasRecording = isRecording.value;
      status.value = 'Disconnected';
      ws = null;

      if (wasRecording) {
        scheduleReconnect();
      }
    };
  });

  const clearCountdown = $(() => {
    if (countdownInterval) {
      clearInterval(countdownInterval);
      countdownInterval = null;
    }
    nextReconnectIn.value = 0;
  });

  // Phase 2: Simple real-time volume meter using Web Audio API Analyser
  const startVolumeMeter = $((stream: MediaStream) => {
    try {
      if (audioContext) {
        audioContext.close();
      }
      audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const source = audioContext.createMediaStreamSource(stream);
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 64;
      analyser.minDecibels = -90;
      analyser.maxDecibels = -10;
      analyser.smoothingTimeConstant = 0.7;

      source.connect(analyser);

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      const update = () => {
        if (!analyser || !isRecording.value) {
          volumeLevel.value = 0;
          return;
        }
        analyser.getByteFrequencyData(dataArray);
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) sum += dataArray[i];
        const avg = sum / bufferLength;
        volumeLevel.value = Math.min(100, Math.round((avg / 255) * 100));
        volumeRaf = requestAnimationFrame(update);
      };
      update();
    } catch (e) {
      // Visualizer is non-critical; continue without it
      volumeLevel.value = 0;
    }
  });

  const stopVolumeMeter = $(() => {
    if (volumeRaf) {
      cancelAnimationFrame(volumeRaf);
      volumeRaf = null;
    }
    if (audioContext) {
      audioContext.close().catch(() => {});
      audioContext = null;
    }
    analyser = null;
    volumeLevel.value = 0;
  });

  // C: Copy finalized transcript to clipboard
  const copyFinalTranscript = $(async () => {
    if (!finalTranscript.value) return;
    try {
      await navigator.clipboard.writeText(finalTranscript.value);
      const originalStatus = status.value;
      status.value = 'Copied finalized text!';
      setTimeout(() => {
        if (status.value === 'Copied finalized text!') {
          status.value = originalStatus;
        }
      }, 1500);
    } catch {
      // Fallback
      alert(finalTranscript.value);
    }
  });

  const scheduleReconnect = $(() => {
    clearCountdown();

    if (reconnectAttempts.value >= maxReconnectAttempts) {
      isReconnecting.value = false;
      status.value = 'Reconnection failed after multiple attempts. Please click Reconnect manually.';
      return;
    }

    reconnectAttempts.value++;
    const delaySeconds = Math.min(Math.pow(2, reconnectAttempts.value), 10);

    isReconnecting.value = true;
    nextReconnectIn.value = delaySeconds;
    status.value = `Connection lost. Reconnecting in ${delaySeconds}s... (attempt ${reconnectAttempts.value}/${maxReconnectAttempts})`;

    // Visible countdown
    countdownInterval = setInterval(() => {
      nextReconnectIn.value--;
      if (nextReconnectIn.value > 0) {
        status.value = `Connection lost. Reconnecting in ${nextReconnectIn.value}s... (attempt ${reconnectAttempts.value}/${maxReconnectAttempts})`;
      }
    }, 1000);

    reconnectTimeout = setTimeout(() => {
      clearCountdown();
      connectWebSocket(true);
    }, delaySeconds * 1000);
  });

  const startRecording = $(async () => {
    if (isRecording.value) return;

    reconnectAttempts.value = 0;
    isRecording.value = true;
    status.value = 'Recording...';

    // Phase 2: Reset transcripts for new session
    finalTranscript.value = '';
    partialTranscript.value = '';
    transcript.value = '';

    await connectWebSocket(false);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);

      mediaRecorder.ondataavailable = async (event) => {
        if (event.data.size > 0 && ws && ws.readyState === WebSocket.OPEN) {
          ws.send(event.data);
        }
      };

      mediaRecorder.start(2000);

      // Phase 2: Start live volume visualization
      startVolumeMeter(stream);
    } catch (err) {
      // Mic permission / device error is non-fatal for the reconnect test flow;
      // the WS failure will still trigger the banner via the isRecording flag we already set.
      status.value = 'Mic unavailable (reconnect test mode)';
    }
  });

  const stopRecording = $(() => {
    clearReconnectTimeout();
    clearCountdown();

    if (mediaRecorder) {
      mediaRecorder.stop();
      mediaRecorder.stream.getTracks().forEach(track => track.stop());
      mediaRecorder = null;
    }

    if (ws) {
      try {
        ws.send(JSON.stringify({ type: 'end' }));
      } catch {}
      ws.close();
      ws = null;
    }

    isRecording.value = false;
    isReconnecting.value = false;
    status.value = 'Stopped';
    reconnectAttempts.value = 0;

    // Phase 2: Reset transcripts and volume
    finalTranscript.value = '';
    partialTranscript.value = '';
    transcript.value = '';
    stopVolumeMeter();
  });

  return (
    <div class="container">
      <h1>ASR Real-time Comparison</h1>
      <p style={{ textAlign: 'center', color: '#94a3b8' }}>
        Whisper (tiny/small/medium/large-v3-turbo), Qwen3-ASR &amp; Voxtral (Real-time Web Audio)
      </p>

      <div class="model-selector">
        {models.map((model) => (
          <label key={model.id}>
            <input
              type="radio"
              name="model"
              value={model.id}
              checked={selectedModel.value === model.id}
              onChange$={() => { selectedModel.value = model.id; saveSettings(); }}
            />
            {model.label}
          </label>
        ))}
      </div>

      {/* Phase 2: Settings Panel */}
      <div class="settings-panel">
        <div class="settings-header">
          <strong>Generation Settings</strong>
          <span class="settings-note">Applied on next Start / Reconnect</span>
        </div>

        <div class="settings-presets">
          <button type="button" onClick$={() => {
            beamSize.value = 8; temperature.value = 0.0; repetitionPenalty.value = 1.12; useDedicatedClass.value = true;
            saveSettings();
          }}>High Accuracy (ja)</button>
          <button type="button" onClick$={() => {
            beamSize.value = 6; temperature.value = 0.0; repetitionPenalty.value = 1.15; useDedicatedClass.value = true;
            saveSettings();
          }}>Balanced (recommended)</button>
          <button type="button" onClick$={() => {
            beamSize.value = 3; temperature.value = 0.2; repetitionPenalty.value = 1.10; useDedicatedClass.value = true;
            saveSettings();
          }}>Faster</button>
        </div>

        <div class="settings-controls">
          <label>
            Beam Size
            <input type="number" min="1" max="10" value={beamSize.value}
                   onInput$={(e) => { beamSize.value = Number((e.target as HTMLInputElement).value); saveSettings(); }} />
          </label>

          <label>
            Temperature
            <input type="number" step="0.1" min="0" max="1" value={temperature.value}
                   onInput$={(e) => { temperature.value = Number((e.target as HTMLInputElement).value); saveSettings(); }} />
          </label>

          <label>
            Repetition Penalty
            <input type="number" step="0.01" min="1" max="1.5" value={repetitionPenalty.value}
                   onInput$={(e) => { repetitionPenalty.value = Number((e.target as HTMLInputElement).value); saveSettings(); }} />
          </label>

          <label class="checkbox">
            <input type="checkbox" checked={useDedicatedClass.value}
                   onChange$={(e) => { useDedicatedClass.value = (e.target as HTMLInputElement).checked; saveSettings(); }} />
            Use Dedicated Class (recommended)
          </label>
        </div>
      </div>

      <div class="controls">
        <button onClick$={startRecording} disabled={isRecording.value}>
          🎤 Start Recording
        </button>
        <button onClick$={stopRecording} disabled={!isRecording.value}>
          ⏹ Stop
        </button>

        {(status.value.includes('Disconnected') || status.value.includes('error') || status.value.includes('failed') || isReconnecting.value) ? (
          <button 
            data-testid="reconnect-button"
            onClick$={() => {
              clearCountdown();
              reconnectAttempts.value = 0;
              connectWebSocket(true);
            }} 
            disabled={isRecording.value}
          >
            🔄 {isReconnecting.value ? 'Retry Now' : 'Reconnect Now'}
          </button>
        ) : null}
      </div>

      {/* Detailed Reconnection UI */}
      {isReconnecting.value ? (
        <div class="reconnection-banner" data-testid="reconnection-banner">
          <div class="reconnection-header">
            <span class="spinner">⟳</span>
            <strong>Reconnecting to server...</strong>
          </div>
          <div class="reconnection-details">
            <p>
              Connection to the ASR server was lost. 
              We are automatically trying to reconnect to continue real-time transcription.
            </p>
            <div class="reconnection-meta">
              <span data-testid="reconnect-attempt">Attempt <strong>{reconnectAttempts.value}</strong> of {maxReconnectAttempts}</span>
              {nextReconnectIn.value > 0 && (
                <span data-testid="reconnect-countdown"> • Next attempt in <strong>{nextReconnectIn.value}s</strong></span>
              )}
            </div>
          </div>
          <div class="reconnection-actions">
            <button 
              onClick$={() => {
                clearCountdown();
                reconnectAttempts.value = 0;
                connectWebSocket(true);
              }}
              disabled={isRecording.value}
            >
              Retry Immediately
            </button>
            <button onClick$={stopRecording}>
              Stop Recording
            </button>
          </div>
          <p class="reconnection-note" data-testid="reconnection-note">
            Your current transcript is preserved and will continue after reconnection.
          </p>
        </div>
      ) : (
        <div 
          data-testid="status" 
          style={{ textAlign: 'center', margin: '1rem 0', color: '#64748b' }}
        >
          Status: {status.value}
          {reconnectAttempts.value > 0 && !isRecording.value && (
            <span style={{ marginLeft: '8px', fontSize: '0.85em' }}>
              (Attempts: {reconnectAttempts.value}/{maxReconnectAttempts})
            </span>
          )}
        </div>
      )}

      {/* Phase 2: Real-time Volume Meter (audio level visual feedback) */}
      <div class="volume-meter" data-testid="volume-meter">
        <div class="volume-label">Input Level</div>
        <div class="volume-bar-bg">
          <div
            class="volume-bar-fill"
            style={{ width: `${volumeLevel.value}%` }}
            data-level={volumeLevel.value}
          />
        </div>
        <div class="volume-value">{volumeLevel.value}</div>
      </div>

      {/* Phase 2: Transcript with is_final visual distinction + copy (C) */}
      <div class="transcript-container">
        <div class="transcript">
          {finalTranscript.value && (
            <span class="final-text" onClick$={copyFinalTranscript} title="Click to copy finalized text">
              {finalTranscript.value}
            </span>
          )}
          {partialTranscript.value && (
            <span class="partial-text" data-is-final="false">
              {finalTranscript.value ? ' ' : ''}{partialTranscript.value}
            </span>
          )}
          {!finalTranscript.value && !partialTranscript.value && 'Transcription will appear here in real-time...'}
        </div>

        {finalTranscript.value && (
          <button class="copy-btn" onClick$={copyFinalTranscript} title="Copy finalized transcript">
            📋
          </button>
        )}
      </div>
    </div>
  );
});