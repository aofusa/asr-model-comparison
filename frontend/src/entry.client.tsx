import { render } from '@builder.io/qwik';
import Root from './root';

// Client entry for pure static SPA build (direct call so top level code runs the render when the bundle loads).
// Per 修正指示書 / 修正案: clean client render for full interactive takeover + proper Qwik $ handler wiring (onClick$ etc).
// We explicitly target #root, clear shell AND remove any q:container attrs left by old shells or previous "resumed" init,
// so Qwik initializes in a clean client render path (not polluted "resumed" from plain shell HTML).
// This addresses the symptom where UI painted but clicks did nothing (no q: event attrs were produced).
console.log('CLIENT ENTRY RENDER CALLED');

// Explicitly target #root (guaranteed by ensure-static-shell.js), clear any previous content,
// and render into it. This is required for reliable Qwik container setup, event delegation
// (for onClick$ etc), and QRL resolution in the pure client-render static build path served by FastAPI.
// Using render(document, ...) leaves the app in a partially hydrated state where buttons appear
// but clicks do nothing (the core symptom reported repeatedly).
const rootEl = document.getElementById('root');
if (rootEl) {
  rootEl.innerHTML = '';
  // Remove any leftover q:container attributes from thin shells or previous renders.
  rootEl.removeAttribute('q:container');
  rootEl.removeAttribute('q:version');
  render(rootEl, <Root />);
} else {
  console.warn('No #root found, falling back to document render');
  render(document, <Root />);
}

console.log('CLIENT ENTRY RENDER CALLED (full client takeover into #root complete)');
