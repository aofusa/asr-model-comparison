import { render, type RenderOptions } from '@builder.io/qwik';
import Root from './root';

// Client entry for dev (Vite + pure Qwik, no Qwik City per AGENT.md).
// This is the module that Vite processes for HMR + render in `npm run dev`.
export default function (opts: RenderOptions) {
  return render(document, <Root />, opts);
}