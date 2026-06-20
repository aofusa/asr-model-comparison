import { component$ } from '@builder.io/qwik';

// Plain root for static SPA build (pure Qwik, Qwik City removed for dev stability + prod hydration).
// See AGENTS.md for repository-wide implementation guidance.
import RealTimeApp from './routes/index';

import './global.css';

export default component$(() => {
  return <RealTimeApp />;
});