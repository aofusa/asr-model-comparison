import { component$, useSignal, useStore, $, useVisibleTask$, noSerialize, type NoSerialize } from '@builder.io/qwik';

function getMicrophoneUnavailableStatus(error?: unknown): string {
  const errorName = error && typeof error === 'object' && 'name' in error
    ? String((error as { name?: unknown }).name)
    : '';

  if (typeof window !== 'undefined' && !window.isSecureContext) {
    return 'Mic unavailable: browser blocks microphone on insecure remote HTTP. Use HTTPS or open this app on localhost.';
  }

  if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
    return 'Mic unavailable: this browser does not expose getUserMedia for this page.';
  }

  if (errorName === 'NotAllowedError' || errorName === 'SecurityError') {
    return 'Mic unavailable: microphone permission was blocked. Allow microphone access in the browser.';
  }

  if (errorName === 'NotFoundError' || errorName === 'DevicesNotFoundError') {
    return 'Mic unavailable: no microphone device was found.';
  }

  return `Mic unavailable${errorName ? ` (${errorName})` : ''}.`;
}

type AudioSourceKind = 'microphone' | 'system' | 'window';

const audioSourceOptions: { value: AudioSourceKind; label: string; description: string }[] = [
  {
    value: 'microphone',
    label: 'Microphone',
    description: 'Use the browser microphone input.',
  },
  {
    value: 'system',
    label: 'System / tab audio',
    description: 'Use browser screen sharing to capture audio played on this device.',
  },
  {
    value: 'window',
    label: 'Window / app audio',
    description: 'Choose a specific window, app, or tab when the browser share picker opens.',
  },
];

function getAudioSourceLabel(source: AudioSourceKind): string {
  return audioSourceOptions.find((option) => option.value === source)?.label || 'Microphone';
}

function getSharedAudioUnavailableStatus(source: AudioSourceKind, error?: unknown): string {
  const errorName = error && typeof error === 'object' && 'name' in error
    ? String((error as { name?: unknown }).name)
    : '';
  const sourceLabel = getAudioSourceLabel(source);

  if (typeof window !== 'undefined' && !window.isSecureContext) {
    return `${sourceLabel} unavailable: browser blocks capture on insecure remote HTTP. Use HTTPS or localhost.`;
  }

  if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getDisplayMedia) {
    return `${sourceLabel} unavailable: this browser does not expose getDisplayMedia for this page.`;
  }

  if (errorName === 'NotAllowedError' || errorName === 'SecurityError') {
    return `${sourceLabel} unavailable: screen/window audio permission was blocked or cancelled.`;
  }

  if (errorName === 'NotFoundError' || errorName === 'NotReadableError') {
    return `${sourceLabel} unavailable: no audio track was shared. Enable "share audio" in the browser picker.`;
  }

  return `${sourceLabel} unavailable${errorName ? ` (${errorName})` : ''}.`;
}

async function requestAudioInputStream(source: AudioSourceKind): Promise<MediaStream> {
  if (source === 'microphone') {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('getUserMedia unavailable');
    }
    return navigator.mediaDevices.getUserMedia({ audio: true });
  }

  if (!navigator.mediaDevices?.getDisplayMedia) {
    throw new Error('getDisplayMedia unavailable');
  }

  const stream = await navigator.mediaDevices.getDisplayMedia({
    video: true,
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
  });

  if (stream.getAudioTracks().length === 0) {
    try { stream.getTracks().forEach(track => track.stop()); } catch {}
    const error = new Error('No shared audio track was selected');
    (error as Error & { name: string }).name = 'NotFoundError';
    throw error;
  }

  return stream;
}

function encodePcm16Wav(samples: Float32Array, sampleRate: number): Blob {
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i++) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

function mergeFloatChunks(chunks: Float32Array[], totalLength: number): Float32Array {
  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

function getPcmStats(samples: Float32Array): { rms: number; peak: number } {
  if (samples.length === 0) {
    return { rms: 0, peak: 0 };
  }

  let sumSquares = 0;
  let peak = 0;
  for (let i = 0; i < samples.length; i++) {
    const abs = Math.abs(samples[i]);
    peak = Math.max(peak, abs);
    sumSquares += samples[i] * samples[i];
  }

  return {
    rms: Math.sqrt(sumSquares / samples.length),
    peak,
  };
}

function isAudiblePcm(samples: Float32Array): boolean {
  const stats = getPcmStats(samples);
  return stats.rms >= 0.006 || stats.peak >= 0.03;
}

function splitAccumulatedForPartial(accumulatedText: string, latestText: string): { finalText: string; partialText: string } {
  const accumulated = accumulatedText.trim();
  const latest = latestText.trim();

  if (!accumulated || !latest) {
    return { finalText: accumulated, partialText: latest };
  }
  if (accumulated === latest) {
    return { finalText: '', partialText: latest };
  }
  if (accumulated.endsWith(latest)) {
    return { finalText: accumulated.slice(0, -latest.length).trim(), partialText: latest };
  }

  return { finalText: accumulated, partialText: latest };
}

const languageOptions = [
  { value: 'auto', label: 'Auto Detect' },
  { value: 'ja', label: 'Japanese' },
  { value: 'en', label: 'English' },
  { value: 'zh', label: 'Chinese' },
  { value: 'ko', label: 'Korean' },
  { value: 'fr', label: 'French' },
  { value: 'de', label: 'German' },
  { value: 'es', label: 'Spanish' },
  { value: 'it', label: 'Italian' },
  { value: 'pt', label: 'Portuguese' },
  { value: 'ru', label: 'Russian' },
  { value: 'ar', label: 'Arabic' },
  { value: 'hi', label: 'Hindi' },
  { value: 'vi', label: 'Vietnamese' },
  { value: 'th', label: 'Thai' },
  { value: 'id', label: 'Indonesian' },
  { value: 'tr', label: 'Turkish' },
  { value: 'nl', label: 'Dutch' },
  { value: 'pl', label: 'Polish' },
  { value: 'sv', label: 'Swedish' },
];

type TranscriptHistoryEntry = {
  id: string;
  modelId: string;
  text: string;
  processingTime: number;
  createdAt: string;
};

function appendTranscriptHistoryValue(
  history: TranscriptHistoryEntry[],
  text: string,
  modelId: string,
  processingTime = 0,
): TranscriptHistoryEntry[] {
  const normalized = text.trim();
  if (!normalized) {
    return history;
  }
  const latest = history[0];
  if (latest?.text === normalized && latest.modelId === modelId) {
    return history;
  }
  return [
    {
      id: `${Date.now()}-${history.length}`,
      modelId,
      text: normalized,
      processingTime,
      createdAt: new Date().toLocaleTimeString(),
    },
    ...history,
  ].slice(0, 50);
}

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
  const modelReady = useSignal(false);
  const modelProgress = useStore({
    phase: 'idle',
    message: 'No model is loading.',
    progress: null as number | null,
    modelId: '',
    elapsedSeconds: null as number | null,
  });

  // Phase 2: Real-time visual feedback
  const volumeLevel = useSignal(0); // 0-100 for meter visualization

  // Phase 2: is_final visual distinction
  const finalTranscript = useSignal('');
  const partialTranscript = useSignal('');
  const transcriptHistory = useSignal<TranscriptHistoryEntry[]>([]);

  // Phase 2 extension (TDD per 修正指示書): per-chunk processing feedback
  // so user sees activity even when a 2s chunk yields empty text from ASR.
  const currentChunkStatus = useSignal<'idle' | 'processing' | 'received'>('idle');
  const lastChunkInfo = useSignal<{
    text: string;
    processingTime: number;
    hadSpeech: boolean;
    chunkIndex: number;
    chunkSizeBytes: number;
    ts: number;
  } | null>(null);
  const chunkCount = useSignal(0);

  // Phase 2: Generation settings (sent on connect / reconnect)
  const beamSize = useSignal(6);
  const temperature = useSignal(0.0);
  const repetitionPenalty = useSignal(1.15);
  const useDedicatedClass = useSignal(true);
  const selectedLanguage = useSignal('auto');
  const translationTarget = useSignal('none');
  const selectedAudioSource = useSignal<AudioSourceKind>('microphone');

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
        selectedLanguage: selectedLanguage.value,
        translationTarget: translationTarget.value,
        selectedAudioSource: selectedAudioSource.value,
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
    mediaRecorder: null as NoSerialize<MediaRecorder> | null,
    ws: null as NoSerialize<WebSocket> | null,
    reconnectTimeout: null as any,
    countdownInterval: null as any,
    audioContext: null as NoSerialize<AudioContext> | null,
    analyser: null as NoSerialize<AnalyserNode> | null,
    micStream: null as NoSerialize<MediaStream> | null,
    pcmSource: null as NoSerialize<MediaStreamAudioSourceNode> | null,
    pcmProcessor: null as NoSerialize<ScriptProcessorNode> | null,
    pcmChunks: null as NoSerialize<Float32Array[]> | null,
    pcmSampleCount: 0,
    volumeRaf: null as number | null,
    reconnectNow: null as NoSerialize<((isReconnect?: boolean) => void | Promise<void>)> | null,
    intentionalStop: false,
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
        if (typeof saved.selectedLanguage === 'string') selectedLanguage.value = saved.selectedLanguage;
        if (typeof saved.translationTarget === 'string') translationTarget.value = saved.translationTarget;
        if (['microphone', 'system', 'window'].includes(saved.selectedAudioSource)) {
          selectedAudioSource.value = saved.selectedAudioSource;
        }
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
    if (isReconnecting.value && refs.reconnectTimeout) {
      return;
    }

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
      const reconnectNow = refs.reconnectNow;
      if (reconnectNow) {
        void reconnectNow(true);
      } else {
        isReconnecting.value = false;
        status.value = 'Reconnect is not ready yet. Please click Reconnect manually.';
      }
    }, delaySeconds * 1000);
  });

  const connectWebSocket = $((isReconnect = false) => {
    clearReconnectTimeout();

    const previousWs = refs.ws;
    if (previousWs) {
      refs.ws = null;
      try { previousWs.close(); } catch {}
    }

    // Dynamic WS URL so it works if served on non-8000 or via proxy (while keeping dev on :8000).
    const wsProtocol = (typeof window !== 'undefined' && window.location.protocol === 'https:') ? 'wss:' : 'ws:';
    const wsHost = (typeof window !== 'undefined' && window.location.host) ? window.location.host : 'localhost:8000';
    const wsUrl = `${wsProtocol}//${wsHost}/api/ws/transcribe`;
    console.log('[connectWebSocket] creating WS to', wsUrl);
    const ws = new WebSocket(wsUrl);
    refs.ws = noSerialize(ws);
    modelReady.value = false;
    modelProgress.phase = 'connecting';
    modelProgress.message = 'Connecting to ASR server.';
    modelProgress.progress = null;
    modelProgress.modelId = selectedModel.value;
    modelProgress.elapsedSeconds = null;

    ws.onopen = () => {
      if (refs.ws !== ws || ws.readyState !== WebSocket.OPEN) {
        return;
      }
      console.log('[connectWebSocket] WS onopen - connected to server');
      clearCountdown();
      reconnectAttempts.value = 0;
      isReconnecting.value = false;
      status.value = isReconnect ? 'Reconnected - Preparing model...' : 'Connected - Preparing model...';
      modelProgress.phase = 'config_sent';
      modelProgress.message = 'Model configuration sent. Waiting for server preparation.';
      modelProgress.modelId = selectedModel.value;

      const config: any = {
        type: 'config',
        model_id: selectedModel.value,
        language: selectedLanguage.value,
        target_language: (selectedModel.value.startsWith('qwen3-asr') || selectedModel.value.startsWith('voxtral'))
          ? translationTarget.value
          : 'none',
        beam_size: beamSize.value,
        temperature: temperature.value,
        repetition_penalty: repetitionPenalty.value,
        use_dedicated_class: useDedicatedClass.value,
        return_timestamps: true,
        audio_source: selectedAudioSource.value,
      };

      // Critical for real-time: send accumulated context on reconnect
      if (isReconnect && previousText.value) {
        config.previous_text = previousText.value;
      }

      ws.send(JSON.stringify(config));
    };

    ws.onmessage = (event) => {
      if (refs.ws !== ws) {
        return;
      }
      const data = JSON.parse(event.data);

      if (data.type === 'model_progress') {
        modelProgress.phase = String(data.phase || 'loading');
        modelProgress.message = String(data.message || 'Preparing model.');
        modelProgress.progress = typeof data.progress === 'number' ? data.progress : null;
        modelProgress.modelId = String(data.model_id || selectedModel.value);
        modelProgress.elapsedSeconds = typeof data.elapsed_seconds === 'number' ? data.elapsed_seconds : null;
        status.value = `Model ${modelProgress.phase}: ${modelProgress.message}`;
        try {
          const statusEl = document.querySelector('[data-testid="status"]');
          if (statusEl) statusEl.textContent = `Status: ${status.value}`;
        } catch {}
        return;
      }

      if (data.type === 'ready') {
        console.log('[WS] received ready from server, model:', data.model_id);
        modelReady.value = true;
        modelProgress.phase = 'ready';
        modelProgress.message = 'Model is loaded and ready.';
        modelProgress.progress = 100;
        modelProgress.modelId = String(data.model_id || selectedModel.value);
        status.value = `Ready - ${data.model_id}`;
        try {
          const statusEl = document.querySelector('[data-testid="status"]');
          if (statusEl) statusEl.textContent = `Status: ${status.value}`;
        } catch {}
      }

      if (data.type === 'transcription') {
        if (!isRecording.value && refs.intentionalStop) {
          return;
        }

        const hadSpeech = data.had_speech !== false;
        const proc = data.processing_time_seconds || 0;
        const latestText = (data.text || '').trim();
        const accumulatedText = typeof data.accumulated_text === 'string' ? data.accumulated_text.trim() : '';

        // Always record chunk activity so the user sees "something happened"
        // even when data.text is empty (the previous root cause of "話しかけても何も起きない").
        lastChunkInfo.value = {
          text: latestText,
          processingTime: proc,
          hadSpeech,
          chunkIndex: data.chunk_index || chunkCount.value,
          chunkSizeBytes: data.chunk_size_bytes || 0,
          ts: Date.now(),
        };
        currentChunkStatus.value = 'received';

        // Only accumulate when there is actual text (preserves existing behavior exactly).
        if (latestText) {
          const isFinal = data.is_final === true;
          const nextContext = accumulatedText || `${previousText.value} ${latestText}`.trim();
          previousText.value = nextContext;

          if (isFinal) {
            // 確定した結果 → finalTranscript に蓄積
            finalTranscript.value = nextContext;
            partialTranscript.value = ''; // partial をクリア
            previousText.value = finalTranscript.value;
            transcript.value = finalTranscript.value; // 後方互換
            transcriptHistory.value = appendTranscriptHistoryValue(
              transcriptHistory.value,
              finalTranscript.value,
              String(data.model_id || selectedModel.value),
              proc,
            );
          } else {
            // 部分結果 → partialTranscript で一時表示
            const split = splitAccumulatedForPartial(nextContext, latestText);
            finalTranscript.value = split.finalText;
            partialTranscript.value = split.partialText;
            transcript.value = nextContext;
          }
        } else if (accumulatedText) {
          previousText.value = accumulatedText;
          finalTranscript.value = accumulatedText;
          transcript.value = accumulatedText;
        } else {
          // Empty text but chunk was processed: give subtle feedback in partial area
          // (keeps the transcript container "alive" for the user).
          partialTranscript.value = hadSpeech
            ? '(speech detected in chunk)'
            : '(no speech detected in this 2s chunk)';
        }

        // Update status with last chunk info (visible processing time feedback).
        status.value = `Ready - ${data.model_id} (last chunk: ${proc.toFixed(2)}s${hadSpeech ? ', speech' : ''})`;
        try {
          const statusEl = document.querySelector('[data-testid="status"]');
          if (statusEl) statusEl.textContent = `Status: ${status.value}`;
        } catch {}
      }

      if (data.type === 'error') {
        currentChunkStatus.value = 'received';
        status.value = `Error: ${data.message || data.code}`;
        if (isRecording.value) {
          scheduleReconnect();
        }
      }

      if (data.type === 'final') {
        const finalText = String(data.text || previousText.value || '').trim();
        if (finalText) {
          previousText.value = finalText;
          finalTranscript.value = finalText;
          partialTranscript.value = '';
          transcript.value = finalText;
          transcriptHistory.value = appendTranscriptHistoryValue(
            transcriptHistory.value,
            finalText,
            String(data.model_id || selectedModel.value),
            0,
          );
        }
        status.value = 'Stream ended';
      }
    };

    ws.onerror = () => {
      if (refs.ws !== ws) {
        return;
      }
      modelReady.value = false;
      status.value = 'Connection error';
      if (isRecording.value) {
        scheduleReconnect();
      }
    };

    ws.onclose = () => {
      if (refs.ws !== ws) {
        return;
      }
      modelReady.value = false;
      const wasRecording = isRecording.value;
      if (refs.intentionalStop) {
        refs.intentionalStop = false;
        status.value = status.value === 'Stream ended' ? status.value : 'Stopped';
      } else {
        status.value = 'Disconnected';
      }
      refs.ws = null;

      if (wasRecording) {
        scheduleReconnect();
      }
    };
  });

  const reconnectNow = $(() => {
    clearCountdown();
    reconnectAttempts.value = 0;
    const reconnect = refs.reconnectNow;
    if (reconnect) {
      void reconnect(true);
    } else {
      status.value = 'Reconnect is not ready yet. Please try again.';
    }
  });

  // Phase 2: Simple real-time volume meter using Web Audio API Analyser
  const startVolumeMeter = $((stream: MediaStream) => {
    try {
      if (refs.audioContext) {
        refs.audioContext.close();
      }
      refs.audioContext = noSerialize(new (window.AudioContext || (window as any).webkitAudioContext)());
      const source = refs.audioContext.createMediaStreamSource(stream);
      refs.analyser = noSerialize(refs.audioContext.createAnalyser());
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

  const startPcmChunkStreaming = $((stream: MediaStream) => {
    try {
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      const sampleRate = audioContext.sampleRate;
      const targetSamples = sampleRate * 2;

      refs.micStream = noSerialize(stream);
      refs.pcmSource = noSerialize(source);
      refs.pcmProcessor = noSerialize(processor);
      refs.pcmChunks = noSerialize([] as Float32Array[]);
      refs.pcmSampleCount = 0;

      processor.onaudioprocess = (event) => {
        if (!isRecording.value || !modelReady.value || !refs.ws || refs.ws.readyState !== WebSocket.OPEN) {
          return;
        }
        if (currentChunkStatus.value === 'processing') {
          return;
        }

        const input = event.inputBuffer.getChannelData(0);
        const copy = new Float32Array(input.length);
        copy.set(input);
        refs.pcmChunks?.push(copy);
        refs.pcmSampleCount += copy.length;

        if (refs.pcmSampleCount < targetSamples) {
          return;
        }

        const chunks = refs.pcmChunks || [];
        const merged = mergeFloatChunks(chunks, refs.pcmSampleCount);
        refs.pcmChunks = noSerialize([] as Float32Array[]);
        refs.pcmSampleCount = 0;

        if (!isAudiblePcm(merged)) {
          currentChunkStatus.value = 'idle';
          status.value = 'Recording... (silent chunk skipped)';
          try {
            const statusEl = document.querySelector('[data-testid="status"]');
            if (statusEl) statusEl.textContent = `Status: ${status.value}`;
          } catch {}
          return;
        }

        chunkCount.value++;
        currentChunkStatus.value = 'processing';
        status.value = `Recording... (processing chunk #${chunkCount.value})`;
        try {
          const statusEl = document.querySelector('[data-testid="status"]');
          if (statusEl) statusEl.textContent = `Status: ${status.value}`;
        } catch {}

        refs.ws.send(encodePcm16Wav(merged, sampleRate));
      };

      source.connect(processor);
      processor.connect(audioContext.destination);
      return true;
    } catch (err) {
      console.warn('[Audio] PCM chunk streaming unavailable, falling back to MediaRecorder:', err);
      return false;
    }
  });

  const stopPcmChunkStreaming = $(() => {
    if (refs.pcmProcessor) {
      try {
        refs.pcmProcessor.disconnect();
        refs.pcmProcessor.onaudioprocess = null;
      } catch {}
      refs.pcmProcessor = null;
    }
    if (refs.pcmSource) {
      try { refs.pcmSource.disconnect(); } catch {}
      refs.pcmSource = null;
    }
    refs.pcmChunks = null;
    refs.pcmSampleCount = 0;
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
    console.log('[StartRecording] handler called, isRecording was:', isRecording.value);
    if (isRecording.value) return;

    const audioSource = selectedAudioSource.value;
    const audioSourceLabel = getAudioSourceLabel(audioSource);
    reconnectAttempts.value = 0;
    modelReady.value = false;
    status.value = `Requesting ${audioSourceLabel}...`;

    // Belt-and-suspenders for flaky Qwik reactivity in static client-render:
    // Directly mutate DOM for critical feedback elements.
    try {
      const statusEl = document.querySelector('[data-testid="status"]');
      if (statusEl) statusEl.textContent = `Status: Requesting ${audioSourceLabel}...`;
      const startBtn = document.querySelector('[data-testid="start-recording"]') as HTMLButtonElement | null;
      if (startBtn) startBtn.disabled = true;
    } catch {}

    let stream: MediaStream;
    try {
      console.log('[StartRecording] requesting audio source:', audioSource);
      stream = await requestAudioInputStream(audioSource);
      console.log('[StartRecording] audio stream acquired, starting MediaRecorder');
    } catch (err) {
      const message = audioSource === 'microphone'
        ? getMicrophoneUnavailableStatus(err)
        : getSharedAudioUnavailableStatus(audioSource, err);
      console.error('[StartRecording] audio source error:', err);
      isRecording.value = false;
      status.value = message;
      try {
        const statusEl = document.querySelector('[data-testid="status"]');
        if (statusEl) statusEl.textContent = `Status: ${message}`;
        const startBtn = document.querySelector('[data-testid="start-recording"]') as HTMLButtonElement | null;
        if (startBtn) startBtn.disabled = false;
      } catch {}
      return;
    }

    isRecording.value = true;
    status.value = `Recording ${audioSourceLabel}...`;
    try {
      const statusEl = document.querySelector('[data-testid="status"]');
      if (statusEl) statusEl.textContent = `Status: Recording ${audioSourceLabel}...`;
    } catch {}

    // Phase 2: Reset transcripts for new session
    finalTranscript.value = '';
    partialTranscript.value = '';
    transcript.value = '';
    previousText.value = '';

    // Phase 2 chunk feedback reset
    currentChunkStatus.value = 'idle';
    lastChunkInfo.value = null;
    chunkCount.value = 0;

    await connectWebSocket(false);

    try {
      const usingPcm = startPcmChunkStreaming(stream);
      if (!usingPcm) {
        refs.mediaRecorder = noSerialize(new MediaRecorder(stream));
        refs.mediaRecorder.ondataavailable = async (event) => {
          if (event.data.size > 0 && modelReady.value && refs.ws && refs.ws.readyState === WebSocket.OPEN) {
            if (currentChunkStatus.value === 'processing') {
              return;
            }
            chunkCount.value++;
            currentChunkStatus.value = 'processing';
            status.value = `Recording... (processing chunk #${chunkCount.value})`;
            refs.ws.send(event.data);
          }
        };
        refs.mediaRecorder.start(2000);
      }

      // Phase 2: Start live volume visualization
      startVolumeMeter(stream);
    } catch (err) {
      const message = `Recording unavailable: ${err instanceof Error ? err.message : 'MediaRecorder failed'}`;
      console.error('[StartRecording] recorder error:', err);
      status.value = message;
      isRecording.value = false;
      try { stream.getTracks().forEach(track => track.stop()); } catch {}
      if (refs.ws) {
        try { refs.ws.close(); } catch {}
        refs.ws = null;
      }
      try {
        const statusEl = document.querySelector('[data-testid="status"]');
        if (statusEl) statusEl.textContent = `Status: ${message}`;
        const startBtn = document.querySelector('[data-testid="start-recording"]') as HTMLButtonElement | null;
        if (startBtn) startBtn.disabled = false;
      } catch {}
    }
  });

  const stopRecording = $(() => {
    clearReconnectTimeout();
    clearCountdown();
    isRecording.value = false;
    isReconnecting.value = false;
    modelReady.value = false;

    if (refs.mediaRecorder) {
      refs.mediaRecorder.stop();
      refs.mediaRecorder.stream.getTracks().forEach(track => track.stop());
      refs.mediaRecorder = null;
    }
    stopPcmChunkStreaming();
    if (refs.micStream) {
      try { refs.micStream.getTracks().forEach(track => track.stop()); } catch {}
      refs.micStream = null;
    }

    if (refs.ws) {
      const wsToClose = refs.ws;
      refs.intentionalStop = true;
      try {
        wsToClose.send(JSON.stringify({ type: 'end' }));
      } catch {}
      setTimeout(() => {
        if (refs.ws === wsToClose) {
          try { wsToClose.close(); } catch {}
          refs.ws = null;
        }
      }, 5000);
    }

    status.value = 'Stopped';
    try {
      const statusEl = document.querySelector('[data-testid="status"]');
      if (statusEl) statusEl.textContent = 'Status: Stopped';
      const startBtn = document.querySelector('[data-testid="start-recording"]') as HTMLButtonElement | null;
      if (startBtn) startBtn.disabled = false;
    } catch {}
    reconnectAttempts.value = 0;

    // Phase 2: Reset volume only. Keep transcript visible after Stop.
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
      const syncSettingsDom = () => {
        const numInputs = document.querySelectorAll('.settings-controls input[type="number"]');
        if (numInputs[0]) (numInputs[0] as HTMLInputElement).value = String(beamSize.value);
        if (numInputs[1]) (numInputs[1] as HTMLInputElement).value = String(temperature.value);
        if (numInputs[2]) (numInputs[2] as HTMLInputElement).value = String(repetitionPenalty.value);
        const cb = document.querySelector('.settings-controls input[type="checkbox"]') as HTMLInputElement | null;
        if (cb) cb.checked = useDedicatedClass.value;
        const languageSelect = document.querySelector('[data-testid="language-select"]') as HTMLSelectElement | null;
        if (languageSelect) languageSelect.value = selectedLanguage.value;
        const translationSelect = document.querySelector('[data-testid="translation-target-select"]') as HTMLSelectElement | null;
        if (translationSelect) translationSelect.value = translationTarget.value;
        const audioRadio = document.querySelector(`input[name="audio-source"][value="${selectedAudioSource.value}"]`) as HTMLInputElement | null;
        if (audioRadio) audioRadio.checked = true;
      };

      const startFn = await startRecording.resolve();
      const stopFn = await stopRecording.resolve();
      const connectFn = await connectWebSocket.resolve();
      refs.reconnectNow = noSerialize(connectFn);
      const startVolumeFn = await startVolumeMeter.resolve();
      const stopVolumeFn = await stopVolumeMeter.resolve();
      const startPcmFn = await startPcmChunkStreaming.resolve();
      const stopPcmFn = await stopPcmChunkStreaming.resolve();
      const highFn = await setHighAccuracy.resolve();
      const balancedFn = await setBalanced.resolve();
      const fasterFn = await setFaster.resolve();
      const saveFn = await saveSettings.resolve();

      const setStatusDom = (value: string) => {
        status.value = value;
        const statusEl = document.querySelector('[data-testid="status"]');
        if (statusEl) statusEl.textContent = `Status: ${value}`;
      };

      const startFallback = async () => {
        if (isRecording.value) return;
        const audioSource = selectedAudioSource.value;
        const audioSourceLabel = getAudioSourceLabel(audioSource);
        reconnectAttempts.value = 0;
        modelReady.value = false;
        setStatusDom(`Requesting ${audioSourceLabel}...`);
        const startBtn = document.querySelector('[data-testid="start-recording"]') as HTMLButtonElement | null;
        if (startBtn) startBtn.disabled = true;
        const stopBtn = document.querySelector('[data-testid="stop-recording"]') as HTMLButtonElement | null;
        if (stopBtn) stopBtn.disabled = true;

        let stream: MediaStream;
        try {
          stream = await requestAudioInputStream(audioSource);
        } catch (err) {
          const message = audioSource === 'microphone'
            ? getMicrophoneUnavailableStatus(err)
            : getSharedAudioUnavailableStatus(audioSource, err);
          isRecording.value = false;
          setStatusDom(message);
          if (startBtn) startBtn.disabled = false;
          if (stopBtn) stopBtn.disabled = true;
          return;
        }

        isRecording.value = true;
        setStatusDom(`Recording ${audioSourceLabel}...`);
        if (stopBtn) stopBtn.disabled = false;

        finalTranscript.value = '';
        partialTranscript.value = '';
        transcript.value = '';
        previousText.value = '';
        currentChunkStatus.value = 'idle';
        lastChunkInfo.value = null;
        chunkCount.value = 0;

        await connectFn(false);

        try {
          const usingPcm = startPcmFn(stream);
          if (!usingPcm) {
            refs.mediaRecorder = noSerialize(new MediaRecorder(stream));
            refs.mediaRecorder.ondataavailable = async (event) => {
              if (event.data.size > 0 && modelReady.value && refs.ws && (refs.ws.readyState === WebSocket.OPEN || refs.ws.readyState === 1)) {
                if (currentChunkStatus.value === 'processing') {
                  return;
                }
                chunkCount.value++;
                currentChunkStatus.value = 'processing';
                setStatusDom(`Recording... (processing chunk #${chunkCount.value})`);
                refs.ws.send(event.data);
              }
            };
            refs.mediaRecorder.start(2000);
          }
          startVolumeFn(stream);
        } catch (err) {
          const message = `Recording unavailable: ${err instanceof Error ? err.message : 'MediaRecorder failed'}`;
          isRecording.value = false;
          setStatusDom(message);
          try { stream.getTracks().forEach(track => track.stop()); } catch {}
          if (refs.ws) {
            try { refs.ws.close(); } catch {}
            refs.ws = null;
          }
          if (startBtn) startBtn.disabled = false;
          if (stopBtn) stopBtn.disabled = true;
        }
      };

      const stopFallback = () => {
        if (refs.reconnectTimeout) {
          clearTimeout(refs.reconnectTimeout);
          refs.reconnectTimeout = null;
        }
        if (refs.countdownInterval) {
          clearInterval(refs.countdownInterval);
          refs.countdownInterval = null;
        }
        if (refs.mediaRecorder) {
          try {
            refs.mediaRecorder.stop();
            refs.mediaRecorder.stream.getTracks().forEach(track => track.stop());
          } catch {}
          refs.mediaRecorder = null;
        }
        isRecording.value = false;
        isReconnecting.value = false;
        modelReady.value = false;
        stopPcmFn();
        if (refs.micStream) {
          try { refs.micStream.getTracks().forEach(track => track.stop()); } catch {}
          refs.micStream = null;
        }
        if (refs.ws) {
          const wsToClose = refs.ws;
          refs.intentionalStop = true;
          try { wsToClose.send(JSON.stringify({ type: 'end' })); } catch {}
          setTimeout(() => {
            if (refs.ws === wsToClose) {
              try { wsToClose.close(); } catch {}
              refs.ws = null;
            }
          }, 5000);
        }
        reconnectAttempts.value = 0;
        setStatusDom('Stopped');
        const startBtn = document.querySelector('[data-testid="start-recording"]') as HTMLButtonElement | null;
        if (startBtn) startBtn.disabled = false;
        const stopBtn = document.querySelector('[data-testid="stop-recording"]') as HTMLButtonElement | null;
        if (stopBtn) stopBtn.disabled = true;
        stopVolumeFn();
      };

      // Delegated fallback handles clicks even if the individual element listener
      // was not attached yet in the static client-render path.
      if (!(document as any)._amcpDelegatedControls) {
        (document as any)._amcpDelegatedControls = true;
        document.addEventListener('click', (e) => {
          const target = e.target as HTMLElement | null;
          const button = target?.closest('button') as HTMLButtonElement | null;
          if (!button) return;

          if (button.matches('[data-testid="start-recording"]') || button.textContent?.includes('Start Recording')) {
            e.preventDefault();
            e.stopImmediatePropagation();
            void startFallback();
            return;
          }
          if (button.matches('[data-testid="stop-recording"]') || button.textContent?.includes('Stop')) {
            e.preventDefault();
            e.stopImmediatePropagation();
            stopFallback();
            return;
          }

          const text = button.textContent || '';
          if (text.includes('High Accuracy')) {
            e.preventDefault();
            highFn();
            beamSize.value = 8; temperature.value = 0.0; repetitionPenalty.value = 1.12; useDedicatedClass.value = true;
            syncSettingsDom();
          } else if (text.includes('Balanced')) {
            e.preventDefault();
            balancedFn();
            beamSize.value = 6; temperature.value = 0.0; repetitionPenalty.value = 1.15; useDedicatedClass.value = true;
            syncSettingsDom();
          } else if (text.includes('Faster')) {
            e.preventDefault();
            fasterFn();
            beamSize.value = 3; temperature.value = 0.2; repetitionPenalty.value = 1.10; useDedicatedClass.value = true;
            syncSettingsDom();
          }
        }, true);
      }
      if (!(document as any)._amcpDelegatedSettings) {
        (document as any)._amcpDelegatedSettings = true;
        document.addEventListener('change', (e) => {
          const target = e.target as HTMLInputElement | HTMLSelectElement | null;
          if (!target) return;

          if (target.matches('.model-selector input[type="radio"]')) {
            const radio = target as HTMLInputElement;
            if (radio.checked) {
              selectedModel.value = radio.value;
              if (!(radio.value.startsWith('qwen3-asr') || radio.value.startsWith('voxtral'))) {
                translationTarget.value = 'none';
                syncSettingsDom();
              }
              saveFn();
            }
            return;
          }

          if (target.matches('[data-testid="language-select"]')) {
            selectedLanguage.value = (target as HTMLSelectElement).value;
            saveFn();
            return;
          }

          if (target.matches('[data-testid="translation-target-select"]')) {
            translationTarget.value = (target as HTMLSelectElement).value;
            saveFn();
            return;
          }

          if (target.matches('input[name="audio-source"]')) {
            const radio = target as HTMLInputElement;
            if (radio.checked && ['microphone', 'system', 'window'].includes(radio.value)) {
              selectedAudioSource.value = radio.value as AudioSourceKind;
              saveFn();
            }
            return;
          }

          if (target.matches('.settings-controls input[type="checkbox"]')) {
            useDedicatedClass.value = (target as HTMLInputElement).checked;
            saveFn();
          }
        }, true);
      }
      document.documentElement.setAttribute('data-amcp-controls-wired', 'true');

      // Give the Qwik render into #root a moment to finish inserting elements (especially after
      // the entry.client.tsx root.innerHTML='' + render). querySelector can miss on the first tick.
      await new Promise(r => setTimeout(r, 50));

      // Start / Stop (the $() QRLs)
      let startEl = document.querySelector('[data-testid="start-recording"]');
      if (!startEl) {
        // Fallback search by text content (robustness)
        startEl = Array.from(document.querySelectorAll('button')).find(b => b.textContent?.includes('Start Recording')) as HTMLElement | null;
      }
      if (startEl) {
        // Avoid duplicate listeners
        if (!(startEl as any)._wiredStart) {
          (startEl as any)._wiredStart = true;
          startEl.addEventListener('click', (e) => { 
            console.log('[NativeWiring] start button clicked via native listener');
            e.preventDefault(); 
            void startFallback(); 
          });
          console.log('[NativeWiring] attached native click to start-recording button');
        }
      }
      let stopEl = document.querySelector('[data-testid="stop-recording"]');
      if (!stopEl) {
        stopEl = Array.from(document.querySelectorAll('button')).find(b => b.textContent?.includes('Stop')) as HTMLElement | null;
      }
      if (stopEl) {
        if (!(stopEl as any)._wiredStop) {
          (stopEl as any)._wiredStop = true;
          stopEl.addEventListener('click', (e) => { 
            console.log('[NativeWiring] stop button clicked via native listener');
            e.preventDefault(); 
            stopFallback(); 
          });
          console.log('[NativeWiring] attached native click to stop-recording button');
        }
      }

      // Presets (hoisted $ QRLs)
      const presets = document.querySelectorAll('.settings-presets button');
      if (presets[0]) { presets[0].addEventListener('click', () => { highFn(); beamSize.value = 8; temperature.value = 0.0; repetitionPenalty.value = 1.12; useDedicatedClass.value = true; syncSettingsDom(); }); }
      if (presets[1]) { presets[1].addEventListener('click', () => { balancedFn(); beamSize.value = 6; temperature.value = 0.0; repetitionPenalty.value = 1.15; useDedicatedClass.value = true; syncSettingsDom(); }); }
      if (presets[2]) { presets[2].addEventListener('click', () => { fasterFn(); beamSize.value = 3; temperature.value = 0.2; repetitionPenalty.value = 1.10; useDedicatedClass.value = true; syncSettingsDom(); }); }

      // Settings number inputs etc: use .resolve() like the preset buttons do.
      // This ensures no bare 'saveSettings' identifier ends up in the listener closure source
      // that gets serialized by the Qwik optimizer into the client chunks (the root cause of the ReferenceError).
      const numInputs = document.querySelectorAll('.settings-controls input[type="number"]');
      if (numInputs[0]) numInputs[0].addEventListener('input', (e: any) => { beamSize.value = Number((e.target as HTMLInputElement).value); saveFn(); });
      if (numInputs[1]) numInputs[1].addEventListener('input', (e: any) => { temperature.value = Number((e.target as HTMLInputElement).value); saveFn(); });
      if (numInputs[2]) numInputs[2].addEventListener('input', (e: any) => { repetitionPenalty.value = Number((e.target as HTMLInputElement).value); saveFn(); });

      // Checkbox
      const cb = document.querySelector('.settings-controls input[type="checkbox"]');
      if (cb) cb.addEventListener('change', (e: any) => { useDedicatedClass.value = (e.target as HTMLInputElement).checked; saveFn(); });
      const languageSelect = document.querySelector('[data-testid="language-select"]');
      if (languageSelect) languageSelect.addEventListener('change', (e: any) => { selectedLanguage.value = (e.target as HTMLSelectElement).value; saveFn(); });
      const translationSelect = document.querySelector('[data-testid="translation-target-select"]');
      if (translationSelect) translationSelect.addEventListener('change', (e: any) => { translationTarget.value = (e.target as HTMLSelectElement).value; saveFn(); });
      document.querySelectorAll('input[name="audio-source"]').forEach((r) => {
        r.addEventListener('change', (e: any) => {
          const t = e.target as HTMLInputElement;
          if (t.checked && ['microphone', 'system', 'window'].includes(t.value)) {
            selectedAudioSource.value = t.value as AudioSourceKind;
            saveFn();
          }
        });
      });

      // Model radios
      document.querySelectorAll('.model-selector input[type="radio"]').forEach((r) => {
        r.addEventListener('change', (e: any) => {
          const t = e.target as HTMLInputElement;
          if (t.checked) {
            selectedModel.value = t.value;
            if (!(t.value.startsWith('qwen3-asr') || t.value.startsWith('voxtral'))) {
              translationTarget.value = 'none';
              syncSettingsDom();
            }
            saveFn();
          }
        });
      });
    } catch (e) {
      console.error('WIRING TASK ERROR (non-fatal):', e);
      /* non fatal for wiring fallback */
    }
    console.log('NATIVE WIRING TASK COMPLETED - start/stop listeners should be attached (or Qwik onClick$ working). Check for [NativeWiring] logs on click.');
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

      <div class="audio-source-selector" data-testid="audio-source-selector">
        <div class="audio-source-header">
          <strong>Audio Input Source</strong>
          <span>Applied on next Start Recording</span>
        </div>
        <div class="audio-source-options">
          {audioSourceOptions.map((source) => (
            <label key={source.value}>
              <input
                type="radio"
                name="audio-source"
                value={source.value}
                checked={selectedAudioSource.value === source.value}
                disabled={isRecording.value}
                onChange$={() => { selectedAudioSource.value = source.value; saveSettings(); }}
              />
              <span>
                <strong>{source.label}</strong>
                <small>{source.description}</small>
              </span>
            </label>
          ))}
        </div>
        <p class="audio-source-note" data-testid="audio-source-note">
          System/window capture uses the browser share picker. Availability depends on the browser, OS, and whether "share audio" is enabled for the selected target.
        </p>
      </div>

      {/* Phase 2: Settings Panel */}
      <div class="settings-panel">
        <div class="settings-header">
          <strong>Generation Settings</strong>
          <span class="settings-note">Applied on next Start / Reconnect</span>
        </div>

        <div class="settings-controls">
          <label>
            Input Language
            <select
              data-testid="language-select"
              value={selectedLanguage.value}
              onChange$={(e) => { selectedLanguage.value = (e.target as HTMLSelectElement).value; saveSettings(); }}
            >
              {languageOptions.map((language) => (
                <option key={language.value} value={language.value}>{language.label}</option>
              ))}
            </select>
          </label>

          <label>
            Translation Output (Qwen/Voxtral)
            <select
              data-testid="translation-target-select"
              value={translationTarget.value}
              onChange$={(e) => { translationTarget.value = (e.target as HTMLSelectElement).value; saveSettings(); }}
            >
              <option value="none">No Translation</option>
              {languageOptions.filter((language) => language.value !== 'auto').map((language) => (
                <option key={language.value} value={language.value}>{language.label}</option>
              ))}
            </select>
          </label>
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
        <button data-testid="start-recording" type="button" disabled={isRecording.value}>
          🎤 Start Recording
        </button>
        <button data-testid="stop-recording" type="button" disabled={!isRecording.value}>
          ⏹ Stop
        </button>

        {(status.value.includes('Disconnected') || status.value.includes('error') || status.value.includes('failed') || isReconnecting.value) ? (
          <button 
            data-testid="reconnect-button"
            onClick$={reconnectNow}
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
              onClick$={reconnectNow}
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

      <div class="model-progress" data-testid="model-progress">
        <div class="model-progress-header">
          <strong>Model preparation</strong>
          <span>{modelProgress.modelId || selectedModel.value}</span>
        </div>
        <div class="model-progress-message">
          {modelProgress.phase === 'idle'
            ? 'Model will be prepared when recording starts.'
            : `${modelProgress.phase}: ${modelProgress.message}`}
          {modelProgress.elapsedSeconds !== null ? ` (${modelProgress.elapsedSeconds.toFixed(2)}s)` : ''}
        </div>
        <div class="model-progress-bar-bg">
          <div
            class="model-progress-bar-fill"
            style={{ width: `${modelProgress.progress ?? (modelReady.value ? 100 : 35)}%` }}
            data-progress={modelProgress.progress ?? ''}
          />
        </div>
      </div>

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
            Last chunk #{lastChunkInfo.value.chunkIndex || chunkCount.value}: {lastChunkInfo.value.processingTime.toFixed(2)}s
            {lastChunkInfo.value.hadSpeech ? ' (speech)' : ' (no speech)'} — {lastChunkInfo.value.text || '(empty result)'}
            {lastChunkInfo.value.chunkSizeBytes > 0 ? ` — ${lastChunkInfo.value.chunkSizeBytes} bytes` : ''}
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

      <div class="transcript-history" data-testid="transcript-history">
        <div class="history-header">
          <strong>Transcript History</strong>
          <span>{transcriptHistory.value.length} item(s)</span>
        </div>
        {transcriptHistory.value.length === 0 ? (
          <div class="history-empty">Finalized transcription results will remain here.</div>
        ) : (
          transcriptHistory.value.map((entry) => (
            <div class="history-item" key={entry.id}>
              <div class="history-meta">
                <span>{entry.createdAt}</span>
                <span>{entry.modelId}</span>
                <span>{entry.processingTime.toFixed(2)}s</span>
              </div>
              <div class="history-text">{entry.text}</div>
            </div>
          ))
        )}
      </div>
    </div>
  );
});