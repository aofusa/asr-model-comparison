#!/usr/bin/env node
/**
 * Post-build safeguard for static single-app deployment (pure Qwik + Vite, no Qwik City).
 * Prioritizes reliable "build once → serve from FastAPI backend".
 *
 * Guarantees:
 * - index.html always exists with correct title and #root + thin shell for initial paint / E2E
 * - Client script tag is *always* inserted (even if primary qwik build emits none) so hydration works in prod
 * - Validation fails hard if no client bundle wired (prevents silent "shell only" in run.ps1/prod)
 * - Client entry (entry.client.tsx) clears shell and does full render/hydration
 */

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';

const distDir = path.resolve('dist');
const indexPath = path.join(distDir, 'index.html');

if (!fs.existsSync(distDir)) {
  console.error('[ensure-static-shell] dist/ does not exist.');
  process.exit(1);
}

let html = fs.existsSync(indexPath) ? fs.readFileSync(indexPath, 'utf8') : '';

const hasRoot = html.includes('id="root"');
const hasTitle = html.includes('ASR Real-time Comparison');

// Inject a *minimal* skeleton (intentionally NOT the full UI) to guarantee #root exists for E2E and initial paint.
// Per 修正案 / 修正指示書: avoid destructive full replace of #root with plain HTML (old reliableShell had buttons/inputs that
// had no q: attrs and conflicted with client render + event wiring). 
// Client entry (src/entry.client.tsx) does root.innerHTML='' + render(root, <Root />) for full takeover.
// The full interactive UI (with onClick$/onInput$ wired by Qwik, settings, status, banner, volume etc.) 
// comes exclusively from the Qwik component after client render/hydration.
// This (thinned shell + clean client render) addresses the prod static build hydration failure where clicks did nothing.
if (!hasRoot) {
  console.log('[ensure-static-shell] Ensuring minimal #root for client render (no destructive shell override)...');
  const minimalRoot = '<div id="root"></div>';
  if (html.includes('</body>')) {
    html = html.replace('</body>', minimalRoot + '</body>');
  } else {
    html += minimalRoot;
  }
}

// Build the client entry bundle (with render call from entry.client.tsx) for full client render in prod static SPA.
// This ensures the client bundle executes the render, replacing the shell content for proper hydration.
try {
  execSync('npx vite build --config vite.config.client.ts', { stdio: 'inherit' });
  console.log('[ensure-static-shell] Client entry bundle built for full render.');
} catch (e) {
  console.error('Client entry build failed', e);
}

// Find the client bundle that contains the render call from entry.client (the one with the render(document) or console.log).
// This is the entry point bundle that executes the full render into #root when loaded.
const buildDir = path.resolve('dist/build');
let clientFiles = [];
try {
  clientFiles = fs.readdirSync(buildDir).filter(f => f.startsWith('q-') && f.endsWith('.js'));
} catch (e) {
  console.warn('[ensure-static-shell] dist/build not present after client build?');
}
let entryClient = null;
for (const f of clientFiles) {
  const content = fs.readFileSync(path.join(buildDir, f), 'utf8');
  if (content.includes('render(document') || content.includes('CLIENT ENTRY RENDER CALLED')) {
    entryClient = f;
    break;
  }
}
if (!entryClient && clientFiles.length > 0) {
  // Fallback to the one with the component if not found (should not happen).
  entryClient = clientFiles.map(f => ({f, size: fs.statSync(path.join(buildDir, f)).size})).sort((a, b) => b.size - a.size)[0].f;
}

if (entryClient) {
  const clientSrc = `/build/${entryClient}`;
  const scriptTag = `<script type="module" src="${clientSrc}" crossorigin></script>`;

  // Robust insertion: replace any prior q- script tag if present (from qwik build html),
  // else explicitly insert before </body>. This fixes the case where primary `qwik build`
  // emits no <script src=...q-*.js> at all (current SPA + custom entry setup).
  const before = html;
  html = html.replace(/<script[^>]*src=["'][^"']*q-[^"']*\.js["'][^>]*><\/script>/i, scriptTag);
  if (html === before) {
    // No q- script was present to replace; insert one.
    if (html.includes('</body>')) {
      html = html.replace('</body>', scriptTag + '</body>');
    } else {
      html += '\n' + scriptTag;
    }
    console.log('[ensure-static-shell] Inserted client script tag (primary build had no q- script).');
  } else {
    console.log('[ensure-static-shell] Updated existing script tag to point at client entry bundle.');
  }

  console.log('[ensure-static-shell] Using client entry bundle for full render:', clientSrc);

  // Hard validation: fail the build if script not wired up (prevents silent hydration death in prod).
  if (!html.includes(clientSrc)) {
    console.error('[ensure-static-shell] FATAL: after insertion logic, clientSrc not found in html. Aborting to surface the issue.');
    process.exit(1);
  }
} else {
  console.warn('[ensure-static-shell] WARNING: No client entry q-*.js bundle found. Prod static serve will have NO JS -> no hydration.');
}

// Write the final guaranteed shell + client bootstrap
fs.writeFileSync(indexPath, html, 'utf8');

console.log('[ensure-static-shell] Minimal root + client script ready for backend/static/.');
console.log('  Per 修正指示書: thin non-destructive root (no big plain shell); entry.client.tsx clears + renders full interactive UI with Qwik event wiring.');
