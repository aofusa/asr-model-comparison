import { test, expect } from '@playwright/test';

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
});

test('rust websocket accepts accelerator config and streams transcription result', async ({ page }) => {
  await page.goto('/health');

  const messages = await page.evaluate(async () => {
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
          ws.send(new Uint8Array([82, 73, 70, 70, 1, 2, 3, 4]));
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
  });

  expect(messages.some((message) => message.type === 'ready')).toBe(true);
  const transcription = messages.find((message) => message.type === 'transcription');
  expect(transcription?.model_id).toBe('whisper-tiny');
  expect(transcription?.accelerator?.preference).toBe('gpu');
  expect(transcription?.text).toContain('Recognized');
});
