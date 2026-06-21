import { test, expect } from '@playwright/test';

function makePcm16Wav(samples: number[], sampleRate = 16000): number[] {
  const dataSize = samples.length * 2;
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
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (const sample of samples) {
    view.setInt16(offset, sample, true);
    offset += 2;
  }

  return Array.from(new Uint8Array(buffer));
}

test('rust server exposes health and compatible model metadata', async ({ page }) => {
  await page.goto('/health');
  await expect(page.locator('body')).toContainText('amcp-rust-backend');

  const models = await page.evaluate(async () => {
    const response = await fetch('/api/models');
    return response.json();
  });

  expect(models.some((model: { id: string }) => model.id === 'whisper-tiny')).toBe(true);
  expect(models.some((model: { id: string }) => model.id === 'qwen3-asr-0.6b')).toBe(true);
  expect(models.some((model: { id: string }) => model.id === 'voxtral-mini-4b')).toBe(true);

  const status = await page.evaluate(async () => {
    const response = await fetch('/api/status');
    return response.json();
  });
  expect(status.available_backends).toContain('cpu');
  expect(status.runtime_backends.some((backend: { model_id: string }) => backend.model_id === 'qwen3-asr-0.6b')).toBe(true);
  expect(status.runtime_backends.some((backend: { model_id: string }) => backend.model_id === 'voxtral-mini-4b')).toBe(true);
});

test('rust websocket accepts accelerator config and streams transcription result', async ({ page }) => {
  await page.goto('/health');

  const wavBytes = makePcm16Wav([0, 10000, -10000, 0]);
  const messages = await page.evaluate(async (audioBytes) => {
    return new Promise<any[]>((resolve, reject) => {
      const ws = new WebSocket('ws://127.0.0.1:8787/api/ws/transcribe');
      const received: any[] = [];
      const timeout = setTimeout(() => {
        ws.close();
        reject(new Error('websocket test timed out'));
      }, 5000);

      ws.onopen = () => {
        ws.send(JSON.stringify({
          type: 'config',
          model_id: 'whisper-tiny',
          language: 'ja',
          accelerator: 'gpu',
          hardware_accelerator: 'gpu',
        }));
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        received.push(data);
        if (data.type === 'ready') {
          ws.send(new Uint8Array(audioBytes));
        }
        if (data.type === 'transcription') {
          clearTimeout(timeout);
          ws.close();
          resolve(received);
        }
      };

      ws.onerror = () => {
        clearTimeout(timeout);
        reject(new Error('websocket failed'));
      };
    });
  }, wavBytes);

  expect(messages.some((message) => message.type === 'ready')).toBe(true);
  const transcription = messages.find((message) => message.type === 'transcription');
  expect(transcription?.model_id).toBe('whisper-tiny');
  expect(transcription?.accelerator?.preference).toBe('gpu');
  expect(transcription?.had_speech).toBe(true);
  expect(transcription?.input_sample_rate).toBe(16000);
  expect(transcription?.runtime_backend).toBeTruthy();
  expect(transcription?.text).toContain('Recognized');
});
