import { render } from '@builder.io/qwik';
import Root from './root';

// Client entry for pure static SPA build (direct call so top level code runs the render when the bundle loads).
// Per 修正指示書 / 修正案: clean client render for full interactive takeover + proper Qwik $ handler wiring (onClick$ etc).
// We explicitly target #root, clear shell AND remove any q:container attrs left by old shells or previous "resumed" init,
// so Qwik initializes in a clean client render path (not polluted "resumed" from plain shell HTML).
// This addresses the symptom where UI painted but clicks did nothing (no q: event attrs were produced).
console.log('CLIENT ENTRY RENDER CALLED');

// Render to document (standard Qwik CSR pattern) so the renderer sets up event delegation and q: attrs properly
// for all on*$ in the initial paint. The #root is kept for some E2E waits (with .catch) and the marker inside component.
console.log('CLIENT ENTRY RENDER CALLED (to document for full event wiring)');
render(document, <Root />);
