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

  // Phase 2 extension (TDD per 修正指示書): per-chunk processing feedback
  // so user sees activity even when a 2s chunk yields empty text from ASR.
  const currentChunkStatus = useSignal<'idle' | 'processing' | 'received'>('idle');
  const lastChunkInfo = useSignal<{
    text: string;
    processingTime: number;
    hadSpeech: boolean;
    ts: number;
  } | null>(null);
  const chunkCount = useSignal(0);

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

  // Hoisted $ handlers for presets (so we can also attach native fallbacks in wiring task for prod client-render path).
  const setHighAccuracy = $(() => {
    beamSize.value = 8; temperature.value = 0.0; repetitionPenalty.value = 1.12; useDedicatedClass.value = true;
    saveSettings();
  });
  const setBalanced = $(() => {
    beamSize.value = 6; temperature.value = 0.0; repetitionPenalty.value = 1.15; useDedicatedClass.value = true;
    saveSettings();
  });
  const setFaster = $(() => {
    beamSize.value = 3; temperature.value = 0.2; repetitionPenalty.value = 1.10; useDedicatedClass.value = true;
    saveSettings();
  });

  // Fix for Qwik static build / prod hydration + optimizer warnings:
  // Previously top-level `let` + assignments inside $() handlers caused
  // "Cannot reassign a variable declared with `const`" in built chunks
  // (connectWebSocket, scheduleReconnect). Use useStore for imperative
  // side-effect objects (WebSocket, MediaRecorder, timers, Audio nodes).
  // This is the Qwik-recommended way for mutable refs that cross closures.
  const refs = useStore({
    mediaRecorder: null as MediaRecorder | null,
    ws: null as WebSocket | null,
    reconnectTimeout: null as any,
    countdownInterval: null as any,
    audioContext: null as AudioContext | null,
    analyser: null as AnalyserNode | null,
    volumeRaf: null as number | null,
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

  // Explicit client render / takeover marker for static shell prod build.
  // This ensures that when entry.client.tsx does render(document, <Root/>), we can detect
  // successful hydration/takeover in E2E and manual verification (per 改修指示書).
  useVisibleTask$(() => {
    try {
      const rootEl = document.getElementById('root');
      if (rootEl) {
        rootEl.setAttribute('data-hydrated', 'true');
        rootEl.classList.add('client-rendered');
      }
      // The debug marker is already in JSX; this makes it observable via attribute too.
    } catch {}
  });

  // (refs store above replaces the previous top-level lets for Qwik optimizer compatibility)

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
    if (refs.reconnectTimeout) {
      clearTimeout(refs.reconnectTimeout);
      refs.reconnectTimeout = null;
    }
  });

  const clearCountdown = $(() => {
    if (refs.countdownInterval) {
      clearInterval(refs.countdownInterval);
      refs.countdownInterval = null;
    }
    nextReconnectIn.value = 0;
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
    refs.countdownInterval = setInterval(() => {
      nextReconnectIn.value--;
      if (nextReconnectIn.value > 0) {
        status.value = `Connection lost. Reconnecting in ${nextReconnectIn.value}s... (attempt ${reconnectAttempts.value}/${maxReconnectAttempts})`;
      }
    }, 1000);

    refs.reconnectTimeout = setTimeout(() => {
      clearCountdown();
      connectWebSocket(true);
    }, delaySeconds * 1000);
  });

  const connectWebSocket = $((isReconnect = false) => {
    clearReconnectTimeout();

    if (refs.ws) {
      try { refs.ws.close(); } catch {}
      refs.ws = null;
    }

    // Dynamic WS URL so it works if served on non-8000 or via proxy (while keeping dev on :8000).
    const wsProtocol = (typeof window !== 'undefined' && window.location.protocol === 'https:') ? 'wss:' : 'ws:';
    const wsHost = (typeof window !== 'undefined' && window.location.host) ? window.location.host : 'localhost:8000';
    refs.ws = new WebSocket(`${wsProtocol}//${wsHost}/api/ws/transcribe`);

    refs.ws.onopen = () => {
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

      refs.ws!.send(JSON.stringify(config));
    };

    refs.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'ready') {
        status.value = `Ready - ${data.model_id}`;
      }

      if (data.type === 'transcription') {
        const hadSpeech = data.had_speech !== false;
        const proc = data.processing_time_seconds || 0;

        // Always record chunk activity so the user sees "something happened"
        // even when data.text is empty (the previous root cause of "話しかけても何も起きない").
        lastChunkInfo.value = {
          text: data.text || '',
          processingTime: proc,
          hadSpeech,
          ts: Date.now(),
        };
        currentChunkStatus.value = 'received';

        // Only accumulate when there is actual text (preserves existing behavior exactly).
        if (data.text) {
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
        } else {
          // Empty text but chunk was processed: give subtle feedback in partial area
          // (keeps the transcript container "alive" for the user).
          partialTranscript.value = hadSpeech
            ? '(speech detected in chunk)'
            : '(no speech detected in this 2s chunk)';
        }

        // Update status with last chunk info (visible processing time feedback).
        status.value = `Ready - ${data.model_id} (last chunk: ${proc.toFixed(2)}s${hadSpeech ? ', speech' : ''})`;
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

    refs.ws.onerror = () => {
      status.value = 'Connection error';
      if (isRecording.value) {
        scheduleReconnect();
      }
    };

    refs.ws.onclose = () => {
      const wasRecording = isRecording.value;
      status.value = 'Disconnected';
      refs.ws = null;

      if (wasRecording) {
        scheduleReconnect();
      }
    };
  });

  // Phase 2: Simple real-time volume meter using Web Audio API Analyser
  const startVolumeMeter = $((stream: MediaStream) => {
    try {
      if (refs.audioContext) {
        refs.audioContext.close();
      }
      refs.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const source = refs.audioContext.createMediaStreamSource(stream);
      refs.analyser = refs.audioContext.createAnalyser();
      refs.analyser.fftSize = 64;
      refs.analyser.minDecibels = -90;
      refs.analyser.maxDecibels = -10;
      refs.analyser.smoothingTimeConstant = 0.7;

      source.connect(refs.analyser);

      const bufferLength = refs.analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      const update = () => {
        if (!refs.analyser || !isRecording.value) {
          volumeLevel.value = 0;
          return;
        }
        refs.analyser.getByteFrequencyData(dataArray);
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) sum += dataArray[i];
        const avg = sum / bufferLength;
        volumeLevel.value = Math.min(100, Math.round((avg / 255) * 100));
        refs.volumeRaf = requestAnimationFrame(update);
      };
      update();
    } catch (e) {
      // Visualizer is non-critical; continue without it
      volumeLevel.value = 0;
    }
  });

  const stopVolumeMeter = $(() => {
    if (refs.volumeRaf) {
      cancelAnimationFrame(refs.volumeRaf);
      refs.volumeRaf = null;
    }
    if (refs.audioContext) {
      refs.audioContext.close().catch(() => {});
      refs.audioContext = null;
    }
    refs.analyser = null;
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

  const startRecording = $(async () => {
    if (isRecording.value) return;

    reconnectAttempts.value = 0;
    isRecording.value = true;
    status.value = 'Recording...';

    // Phase 2: Reset transcripts for new session
    finalTranscript.value = '';
    partialTranscript.value = '';
    transcript.value = '';

    // Phase 2 chunk feedback reset
    currentChunkStatus.value = 'idle';
    lastChunkInfo.value = null;
    chunkCount.value = 0;

    await connectWebSocket(false);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      refs.mediaRecorder = new MediaRecorder(stream);

      refs.mediaRecorder.ondataavailable = async (event) => {
        if (event.data.size > 0 && refs.ws && refs.ws.readyState === WebSocket.OPEN) {
          chunkCount.value++;
          currentChunkStatus.value = 'processing';
          status.value = `Recording... (processing chunk #${chunkCount.value})`;
          refs.ws.send(event.data);
        }
      };

      refs.mediaRecorder.start(2000);

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

    if (refs.mediaRecorder) {
      refs.mediaRecorder.stop();
      refs.mediaRecorder.stream.getTracks().forEach(track => track.stop());
      refs.mediaRecorder = null;
    }

    if (refs.ws) {
      try {
        refs.ws.send(JSON.stringify({ type: 'end' }));
      } catch {}
      refs.ws.close();
      refs.ws = null;
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

    // Phase 2 chunk feedback reset
    currentChunkStatus.value = 'idle';
    lastChunkInfo.value = null;
    chunkCount.value = 0;
  });

  // Native event wiring fallback for prod static client-render path (where Qwik's on*$ + q: attrs may not attach
  // during render-to-document / client render even after thin shell + entry cleanup per 修正指示書).
  // Calling the resolved QRLs updates signals => Qwik reactivity still drives UI updates.
  // This makes buttons & settings work for the :8000 build while preserving $() + Qwik path for dev/SSR.
  useVisibleTask$(async () => {
    try {
      // Start / Stop (the $() QRLs)
      const startEl = document.querySelector('[data-testid="start-recording"]');
      if (startEl) {
        const fn = await startRecording.resolve();
        startEl.addEventListener('click', (e) => { e.preventDefault(); fn(); });
      }
      const stopEl = document.querySelector('[data-testid="stop-recording"]');
      if (stopEl) {
        const fn = await stopRecording.resolve();
        stopEl.addEventListener('click', (e) => { e.preventDefault(); fn(); });
      }

      // Presets (hoisted $ QRLs)
      const presets = document.querySelectorAll('.settings-presets button');
      if (presets[0]) { const fn = await setHighAccuracy.resolve(); presets[0].addEventListener('click', () => fn()); }
      if (presets[1]) { const fn = await setBalanced.resolve(); presets[1].addEventListener('click', () => fn()); }
      if (presets[2]) { const fn = await setFaster.resolve(); presets[2].addEventListener('click', () => fn()); }

      // Settings number inputs etc: use .resolve() like the preset buttons do.
      // This ensures no bare 'saveSettings' identifier ends up in the listener closure source
      // that gets serialized by the Qwik optimizer into the client chunks (the root cause of the ReferenceError).
      const saveFn = await saveSettings.resolve();
      const numInputs = document.querySelectorAll('.settings-controls input[type="number"]');
      if (numInputs[0]) numInputs[0].addEventListener('input', (e: any) => { beamSize.value = Number((e.target as HTMLInputElement).value); saveFn(); });
      if (numInputs[1]) numInputs[1].addEventListener('input', (e: any) => { temperature.value = Number((e.target as HTMLInputElement).value); saveFn(); });
      if (numInputs[2]) numInputs[2].addEventListener('input', (e: any) => { repetitionPenalty.value = Number((e.target as HTMLInputElement).value); saveFn(); });

      // Checkbox
      const cb = document.querySelector('.settings-controls input[type="checkbox"]');
      if (cb) cb.addEventListener('change', (e: any) => { useDedicatedClass.value = (e.target as HTMLInputElement).checked; saveFn(); });

      // Model radios
      document.querySelectorAll('.model-selector input[type="radio"]').forEach((r) => {
        r.addEventListener('change', (e: any) => {
          const t = e.target as HTMLInputElement;
          if (t.checked) { selectedModel.value = t.value; saveFn(); }
        });
      });
    } catch (e) { /* non fatal for wiring fallback */ }
  });

  return (
    <div class="container">
      <div data-testid="hydrated-marker" style={{ color: 'red', fontWeight: 'bold' }}>CLIENT RENDERED</div>
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
          <button type="button" onClick$={setHighAccuracy}>High Accuracy (ja)</button>
          <button type="button" onClick$={setBalanced}>Balanced (recommended)</button>
          <button type="button" onClick$={setFaster}>Faster</button>
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
        <button data-testid="start-recording" onClick$={startRecording} disabled={isRecording.value}>
          🎤 Start Recording
        </button>
        <button data-testid="stop-recording" onClick$={stopRecording} disabled={!isRecording.value}>
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

      {/* Phase 2 extension: per-chunk processing feedback (addresses "話しかけても何も起きない") */}
      <div class="chunk-feedback" data-testid="chunk-feedback" style={{ textAlign: 'center', margin: '0.5rem 0', fontSize: '0.9em', color: '#64748b' }}>
        {currentChunkStatus.value === 'processing' && '⏳ Processing latest 2s chunk...'}
        {lastChunkInfo.value && (
          <div>
            Last chunk #{chunkCount.value}: {lastChunkInfo.value.processingTime.toFixed(2)}s
            {lastChunkInfo.value.hadSpeech ? ' (speech)' : ' (no speech)'} — {lastChunkInfo.value.text || '(empty result)'}
          </div>
        )}
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