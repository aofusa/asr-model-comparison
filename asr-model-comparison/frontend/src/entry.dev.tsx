import { render, type RenderOptions } from '@builder.io/qwik';
import Root from './root';

// Qwik City client entry.
// This is the module that Vite processes when building with explicit index.html input.
export default function (opts: RenderOptions) {
  return render(document, <Root />, opts);
}