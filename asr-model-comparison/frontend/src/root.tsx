import { component$ } from '@builder.io/qwik';
import {
  QwikCityProvider,
  ServiceWorkerRegister,
} from '@builder.io/qwik-city';

// Import the real app directly to bypass RouterOutlet for maximum
// compatibility during the transition back to Qwik City.
import RealTimeApp from './routes/index';

import './global.css';

export default component$(() => {
  return (
    <QwikCityProvider>
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <link rel="manifest" href="/manifest.json" />
        <title>ASR Real-time Comparison</title>
      </head>
      <body>
        <RealTimeApp />
        <ServiceWorkerRegister />
      </body>
    </QwikCityProvider>
  );
});