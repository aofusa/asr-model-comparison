import { component$ } from '@builder.io/qwik';

// Plain root for static SPA build (pure Qwik, Qwik City removed for dev stability + prod hydration).
// See AGENT.md + FRONTEND_QWIK_STATIC_BUILD_PROD_HYDRATION_FIX_PLAN.md
import RealTimeApp from './routes/index';

import './global.css';

export default component$(() => {
  return <RealTimeApp />;
});