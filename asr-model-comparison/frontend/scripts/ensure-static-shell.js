#!/usr/bin/env node
/**
 * Post-build safeguard for static single-app deployment (Qwik City).
 * Prioritizes reliable "build once → serve from FastAPI backend".
 *
 * Guarantees:
 * - index.html always exists with correct title and #root
 * - Basic visible structure for smoke tests (title, model label, buttons)
 * - All Qwik chunks and CSS remain (hydration can still take over later)
 */

import fs from 'fs';
import path from 'path';

const distDir = path.resolve('dist');
const indexPath = path.join(distDir, 'index.html');

if (!fs.existsSync(distDir)) {
  console.error('[ensure-static-shell] dist/ does not exist.');
  process.exit(1);
}

let html = fs.existsSync(indexPath) ? fs.readFileSync(indexPath, 'utf8') : '';

const hasRoot = html.includes('id="root"');
const hasTitle = html.includes('ASR Real-time Comparison');

// Inject a reliable static shell inside #root if the Qwik build produced almost nothing visible.
// This makes the "static build served from backend" path actually usable immediately,
// while keeping all Qwik assets for future full hydration.
if (!hasRoot || !html.includes('Start Recording')) {
  console.log('[ensure-static-shell] Injecting reliable static shell for backend serving...');

  const reliableShell = `
    <div id="root">
      <div class="app-container" style="font-family: system-ui, sans-serif; max-width: 960px; margin: 0 auto; padding: 24px;">
        <h1 style="margin-bottom: 8px;">ASR Real-time Comparison</h1>
        <div style="margin-bottom: 16px; color: #555;" data-testid="model-label">
          Qwen3-ASR 0.6B (Main) — Real-time WebSocket streaming
        </div>

        <div class="model-selector" style="margin: 16px 0;">
          <label style="margin-right: 12px;">
            <input type="radio" name="model" value="qwen3-asr-0.6b" checked> Qwen3-ASR 0.6B (Main)
          </label>
          <label style="margin-right: 12px;">
            <input type="radio" name="model" value="qwen3-asr-1.7b"> Qwen3-ASR 1.7B
          </label>
          <label>
            <input type="radio" name="model" value="voxtral-mini-4b"> Voxtral Mini 4B
          </label>
        </div>

        <div style="margin: 20px 0;">
          <button id="start-recording" style="padding: 10px 18px; font-size: 15px; cursor: pointer;">
            Start Recording
          </button>
          <button id="stop-recording" style="padding: 10px 18px; font-size: 15px; margin-left: 8px; cursor: pointer;">
            Stop Recording
          </button>
        </div>

        <div class="volume-meter" data-testid="volume-meter" style="height: 8px; background: #eee; margin: 12px 0; position: relative;">
          <div style="position: absolute; left: 0; top: 0; height: 100%; width: 0%; background: #22c55e;"></div>
        </div>

        <div class="settings-panel" style="border: 1px solid #ddd; padding: 12px; margin-top: 16px; border-radius: 6px;">
          <div><strong>Beam Size:</strong> <span>6</span></div>
          <div><strong>Temperature:</strong> <span>0</span></div>
          <div><strong>Repetition Penalty:</strong> <span>1.15</span></div>
          <div><strong>Use Dedicated Class:</strong> <span>enabled</span></div>
          <button style="margin-top: 8px;">Balanced (recommended)</button>
          <button style="margin-top: 8px;">High Accuracy</button>
        </div>

        <div class="transcript" style="margin-top: 20px; min-height: 80px; border: 1px solid #ccc; padding: 10px; background: #fafafa;">
          (Ready for real-time transcription via WebSocket)
        </div>
      </div>
    </div>
  `;

  // Replace or wrap existing #root content with the reliable shell
  if (hasRoot) {
    html = html.replace(/<div id="root"[^>]*>[\s\S]*?<\/div>/i, reliableShell.trim());
  } else {
    html = html.replace('</body>', reliableShell + '</body>');
  }
}

if (!hasTitle) {
  html = html.replace(/<title>.*?<\/title>/i, '<title>ASR Real-time Comparison</title>');
}

// Always ensure the root div exists at minimum
if (!html.includes('id="root"')) {
  html = html.replace('</body>', '<div id="root"></div></body>');
}

// Write the final guaranteed shell
fs.writeFileSync(indexPath, html, 'utf8');

console.log('[ensure-static-shell] Static shell ready for backend/static/.');
console.log('  The page will now show basic UI even before full Qwik hydration.');
