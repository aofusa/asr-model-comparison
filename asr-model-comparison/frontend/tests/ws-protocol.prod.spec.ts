import { test, expect } from '@playwright/test';

/**
 * Lightweight WebSocket Protocol Verification Tests (Production Integration)
 *
 * These tests run against a real backend (started via run.ps1) on port 8000.
 * They verify the actual WebSocket streaming protocol with a lightweight model
 * (whisper-tiny recommended for speed).
 *
 * Goal: Confirm that the real protocol (config → ready → audio chunks → results)
 * works end-to-end without heavy models.
 *
 * Usage:
 *   npm run test:e2e:prod
 *
 * Prerequisites:
 *   - Backend must be running (preferably with whisper-tiny available)
 *   - Use model_id="whisper-tiny" for fast execution
 */

const WS_URL = 'ws://localhost:8000/ws/transcribe';

// Minimal valid WAV (1 second of silence, 16kHz mono PCM) encoded as base64.
// This is small enough to embed and sufficient for protocol verification.
const SILENT_WAV_BASE64 =
  'UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA='; // ~44 bytes header + silence

function base64ToArrayBuffer(base64: string): ArrayBuffer {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
}

test.describe('WebSocket Protocol - Lightweight Verification (whisper-tiny)', () => {
  test('should complete full config → ready → audio → final flow with whisper-tiny', async ({ page }) => {
    test.setTimeout(30000);

    await page.goto('/');

    const result = await page.evaluate(async (wsUrl: string) => {
      return new Promise<any>((resolve, reject) => {
        const ws = new WebSocket(wsUrl);
        const received: any[] = [];
        let readyReceived = false;
        let finalReceived = false;

        const timeout = setTimeout(() => {
          ws.close();
          reject(new Error('WebSocket protocol test timed out'));
        }, 25000);

        ws.onopen = () => {
          // Step 1: Send config for lightweight model
          const config = {
            type: 'config',
            model_id: 'whisper-tiny',
            language: 'ja',
            beam_size: 1, // fastest
            use_dedicated_class: false,
            return_timestamps: false,
          };
          ws.send(JSON.stringify(config));
        };

        ws.onmessage = (event) => {
          try {
            const msg = JSON.parse(event.data);
            received.push(msg);

            if (msg.type === 'ready') {
              readyReceived = true;

              // Step 2: Send a small audio chunk
              const audioBuffer = base64ToArrayBuffer(SILENT_WAV_BASE64);
              ws.send(audioBuffer);

              // Step 3: Send end after a short delay
              setTimeout(() => {
                ws.send(JSON.stringify({ type: 'end' }));
              }, 300);
            }

            if (msg.type === 'final') {
              finalReceived = true;
              clearTimeout(timeout);
              ws.close();
              resolve({
                readyReceived,
                finalReceived,
                messages: received,
                finalText: msg.text,
              });
            }

            if (msg.type === 'error') {
              clearTimeout(timeout);
              ws.close();
              reject(new Error('Server error: ' + msg.message));
            }
          } catch (e) {
            // ignore non-JSON for now
          }
        };

        ws.onerror = (err) => {
          clearTimeout(timeout);
          reject(new Error('WebSocket error during protocol test'));
        };

        ws.onclose = () => {
          if (!finalReceived) {
            clearTimeout(timeout);
            resolve({
              readyReceived,
              finalReceived,
              messages: received,
            });
          }
        };
      });
    }, WS_URL);

    expect(result.readyReceived).toBe(true);
    expect(result.finalReceived).toBe(true);

    // At minimum we should have received a "ready" and a "final"
    const types = result.messages.map((m: any) => m.type);
    expect(types).toContain('ready');
    expect(types).toContain('final');
  });
});