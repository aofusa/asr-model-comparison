import { component$ } from '@builder.io/qwik';
import RealTimeApp from './routes/index';

import './global.css';

export default component$(() => {
  return (
    <>
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <link rel="manifest" href="/manifest.json" />
        <title>ASR Real-time</title>
      </head>
      <body>
        <RealTimeApp />
      </body>
    </>
  );
});