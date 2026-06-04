# E2E Test Design for Frontend UI Components (whisper-tiny only)

**Branch:** feat/e2e-design-frontend-ui-components-whisper-tiny (from main)
**Scope:** Design ONLY. No implementation of new test code or UI changes. Survey existing code + static shell.
**Model Focus:** All tests designed assuming whisper-tiny is pre-selected (lightweight for E2E). Do not test model switching or other models (heavy).
**Test Type:** Playwright E2E, preferably prod mode (baseURL http://localhost:8000, manual server via run.ps1 with whisper-tiny recommended for speed). Use fake mic as in existing configs.
**Existing Patterns (from survey):**
- Use data-testid heavily for reliability (avoids text duplicates in shell vs hydrated).
- Prod smoke: basic visibility, no real WS.
- real-time.spec.ts: heavy use of page.addInitScript for MockWebSocket to make deterministic (fail fast to trigger reconnect logic).
- WS protocol in page.evaluate for browser WS.
- Fake devices: --use-fake-ui-for-media-stream --use-fake-device-for-media-stream
- Timeouts generous (model load, reconnects).
- Assertions on visibility, text content, attributes (e.g. checked, data-level), counts.
- Conditional rendering tested (e.g. banner not visible initially).
- Settings use localStorage key asr-settings-v1.

## 1. Survey: Frontend UI Components / Features (from code reads)

Main component: frontend/src/routes/index.tsx (Qwik component, all-in-one, no sub-components dir).
Fallback/Static Shell: frontend/scripts/ensure-static-shell.js (injects HTML into dist/index.html for prod single-app / smoke tests). Also mirrored in backend/static/index.html and frontend/dist/index.html.

### Core UI Sections (always or conditionally rendered):
- **Header/Title Area**
  - h1: ASR Real-time Comparison
  - p description (updated to mention Whisper models + others)
  - data-testid=model-label (in shell: shows selected model text like Whisper Tiny - Real-time WebSocket streaming)

- **Model Selector** (radios, form-like)
  - Hardcoded in both dynamic + shell (but for design: pre-selected whisper-tiny, verify checked state, labels present).
  - ids: whisper-tiny (default for tests), others listed but not interacted.
  - onChange updates selectedModel signal + saveSettings.
  - Note: design assumes whisper-tiny selected via initial signal or page state; no click tests for other models.

- **Settings Panel** (Phase 2, .settings-panel)
  - Header + note Applied on next Start / Reconnect
  - Presets buttons: High Accuracy (ja), Balanced (recommended), Faster
  - Controls:
    - number input: Beam Size (min1 max10)
    - number: Temperature (step0.1 min0 max1)
    - number: Repetition Penalty (step0.01 min1 max1.5)
    - checkbox: Use Dedicated Class (recommended)
  - Changes: update signals + saveSettings (localStorage)
  - Presets: set specific values for beam/temp/repetition/useDedicatedClass
  - Visible in both shell (static content) and hydrated.

- **Controls / Recording Buttons**
  - button Start Recording (disabled when recording)
  - button Stop / Stop Recording (disabled when not recording)
  - Conditionally: button data-testid=reconnect-button Reconnect Now (shown on disconnect/error/failed status)
  - Clicks: startRecording (sets isRecording, status, resets transcripts, connectWS, getUserMedia, mediaRecorder.start(2000), startVolumeMeter)
  - stopRecording (stop recorder, send {type:end}, close WS, reset states, stopVolumeMeter)

- **Status Area**
  - data-testid=status: shows Status: Idle|Recording...|Stopped|Connected|Disconnected|Reconnecting...|Error:...
  - Updates from signals, WS onmessage (ready, transcription, final, error), reconnect logic.
  - Shows attempts count when >0 and not recording.

- **Reconnection / Error Recovery Banner (detailed, Phase1 priority)**
  - data-testid=reconnection-banner (visible only when isReconnecting)
  - Inside:
    - reconnection-header with spinner + Reconnecting to server...
    - p description about lost connection + auto retry
    - reconnection-meta:
      - data-testid=reconnect-attempt: Attempt X of 5
      - data-testid=reconnect-countdown: Next attempt in Xs...
    - reconnection-actions:
      - Retry Immediately button (inside banner)
      - Stop Recording button
    - p.reconnection-note data-testid=reconnection-note: Your current transcript is preserved...
  - Triggered by: WS onerror/onclose during recording -> scheduleReconnect
  - Logic (from code): exponential backoff (min(2^attempts,10)s), max 5 attempts, on max: status=Reconnection failed..., banner hides?
  - Top level reconnect button also calls connectWebSocket(true)
  - Stop clears reconnect state, hides banner.

- **Volume Meter / Real-time Visual Feedback (Phase 2)**
  - data-testid=volume-meter (always in DOM/shell, .volume-meter)
  - .volume-label Input Level
  - .volume-bar-bg + .volume-bar-fill (style width % , data-level attr)
  - .volume-value text
  - Starts on recording via startVolumeMeter (uses Web Audio Analyser, fake in tests)
  - Updates in raf loop while recording, sets volumeLevel signal 0-100
  - Stops on stopRecording.

- **Transcript Container (is_final distinction + Copy C)**
  - .transcript-container
  - .transcript :
    - span.final-text (if finalTranscript): shows text, clickable -> copyFinalTranscript
    - span.partial-text data-is-final=false (if partial)
    - fallback text Transcription will appear here in real-time...
  - Conditional copy-btn (if finalTranscript): onClick copy, title Copy finalized transcript
  - copy logic: clipboard.writeText, temp status Copied finalized text!, fallback alert
  - Updated from WS: on transcription msg, if is_final: append to final, clear partial, set previousText
    else: set partial, append to transcript display
  - On start: reset final/partial/transcript
  - Preserved on reconnect (per note)

- **Other Global/Supporting**
  - No extra from root/layout (just renders the app + SW)
  - CSS classes for distinction (final vs partial), but test via testid/text/attr
  - localStorage persistence for settings (including selectedModel) - load on visibleTask, save on changes
  - WS integration is core: connect on start, config with model_id=whisper-tiny + params, send chunks, handle msgs, end on stop.
  - But UI tests mock WS for determinism (as in existing real-time tests)

**Hydration vs Static Shell Note:** Prod E2E often hits injected static shell first (basic buttons, labels, meter, settings static html, transcript). Hydrated Qwik takes over (signals, events, dynamic text like status/volume/transcripts). Tests must work with both (use testids, or/ locators, visibility over strict text).

## 2. Feature List (prioritized for E2E design, whisper-tiny focus)
1. Basic App Load and Static Shell (smoke-like, no interaction)
2. Model Selector Visibility (whisper-tiny pre-selected, radios present)
3. Recording Controls (Start/Stop enable/disable, state changes)
4. Status Display (text updates for idle/recording/stopped/disconnect)
5. Reconnection Banner and Recovery (trigger, visibility, content, buttons, countdown/attempts, max attempts, clear on stop)
6. Volume Meter (presence, updates during record via mock)
7. Transcript Rendering and is_final Distinction (final vs partial spans/attrs, content)
8. Copy Final Transcript (button appears, click copies, temp feedback)
9. Settings Panel and Presets (visibility, buttons, inputs, values)
10. Settings Persistence (A: localStorage roundtrip on load/save)
11. WS-driven Updates in UI (ready -> status; transcription msgs -> transcript updates; but via mocks)
12. Error/Disconnect Handling in UI (status, banner trigger, reconnect affordances)
13. Volume Visualizer Lifecycle (start/stop with recording)
14. Mic Handling (fake in headless, error state if fails)

(From existing tests + code: reconnection is heavily featured with many testids; Phase2 visuals/settings/transcript; basic load.)

## 3. E2E Test Design (skeletons as design only - do not create/edit .spec.ts files yet per still no implement)

**Test File Suggestion (for future impl):** e.g. frontend/tests/ui-components-whisper-tiny.spec.ts or extend smoke.prod.spec.ts / real-time.spec.ts
**Run Mode:** Prod E2E (npm run test:e2e:prod), server started with run.ps1 (whisper-tiny auto via config in test or default). Use chromium + fake mic.
**Setup Patterns (reuse):**
- beforeEach: page.addInitScript for MockWebSocket (fail to trigger reconnect for banner tests)
- page.goto('/')
- expect( locators with getByTestId, getByRole, getByText, .locator )
- timeouts for async (recording start, banner appear 5-8s)
- For WS UI effects: either real (with tiny wav? but design uses mocks) or mock that emits specific msgs (ready, {type:transcription, text:hi, is_final:false}, {type:transcription, text:full, is_final:true}, final)
- Verify no real heavy models; config model_id: whisper-tiny

**Proposed Test Cases (design, with pseudo-code for clarity):**

### Smoke / Load (extend existing smoke.prod.spec.ts)
- test(app loads with whisper-tiny selected in static shell + hydrated)
  await page.goto('/');
  await expect(page.getByText('ASR Real-time Comparison')).toBeVisible();
  const label = page.getByTestId('model-label');
  await expect(label).toBeVisible();
  await expect(label).toContainText(/Whisper Tiny/);  // or whatever shell says
  await expect(page.getByRole('button', {name: /Start Recording/i})).toBeVisible();
  // model selector
  const tinyRadio = page.locator('input[value="whisper-tiny"]');
  await expect(tinyRadio).toBeChecked();
  // other whispers present but not selected
  await expect(page.locator('input[value="whisper-small"]')).toBeVisible();
  // volume, status, transcript basic
  await expect(page.getByTestId('volume-meter')).toBeVisible();
  await expect(page.getByTestId('status')).toContainText('Idle');
  const transcript = page.locator('.transcript');
  await expect(transcript).toBeVisible();
  await expect(transcript).toContainText(/Transcription will appear here/);

- test(settings panel is visible in shell (for prod))
  ... check presets, labels for beam/temp etc (as in existing)

### Recording Lifecycle (UI state, assume whisper-tiny)
- test('Start/Stop Recording toggles states and buttons')
  await page.goto('/');
  const startBtn = page.getByRole('button', {name: /Start Recording/i});
  const stopBtn = page.getByRole('button', {name: /Stop/i});
  await expect(startBtn).toBeEnabled();
  await expect(stopBtn).toBeDisabled();
  await startBtn.click();
  await expect(page.getByTestId('status')).toContainText('Recording...');
  await expect(startBtn).toBeDisabled();
  await expect(stopBtn).toBeEnabled();
  await stopBtn.click();
  await expect(page.getByTestId('status')).toContainText('Stopped');
  await expect(startBtn).toBeEnabled();

- test('volume meter updates during recording (with analyser mock if needed)')
  // similar to existing Phase2 test
  await startBtn.click();
  const meter = page.getByTestId('volume-meter');
  await expect(meter).toBeVisible();
  // expect style or data attr change (may need to mock Web Audio in initScript for deterministic >0 level)

### Reconnection / Error Recovery (core, use mock WS from real-time.spec pattern)
- test('reconnection banner appears on WS failure during recording')
  // beforeEach mock that fails
  await startBtn.click();
  const banner = page.getByTestId('reconnection-banner');
  await expect(banner).toBeVisible({timeout:8000});
  await expect(banner).toContainText('Reconnecting to server');
  await expect(page.getByTestId('reconnect-attempt')).toContainText('Attempt 1 of 5');
  await expect(page.getByTestId('reconnection-note')).toContainText('preserved');

- test('countdown and attempt counter update in banner')
  await start... 
  await expect(page.getByTestId('reconnect-countdown')).toBeVisible();
  await expect(...).toContainText(/Next attempt in \d+s/);

- test('"Retry Immediately" in banner re-triggers and updates attempts')
  ... click retry, assert attempt >=2 , banner stays

- test('top-level reconnect-button works')
  await start...
  await expect(page.getByTestId('reconnect-button')).toBeVisible();
  await page.getByTestId('reconnect-button').click();
  await expect(banner).toBeVisible();

- test('Stop Recording during reconnect clears banner and state')
  await start... trigger banner
  await page.getByRole('button', {name: /Stop/i}).click();
  await expect(banner).not.toBeVisible();
  await expect(page.getByTestId('status')).not.toContainText(/Reconnecting/);

- test('max attempts shows failure in status, allows manual reconnect')
  // loop clicks to hit 5+
  await expect(page.getByTestId('status')).toContainText(/failed after multiple attempts/i);
  // top button may reappear

- test('transcript container remains visible/intact during reconnect attempts')

### Transcript + is_final + Copy
- test('final and partial transcripts render distinctly')
  // use page.evaluate or mock to send transcription msgs with is_final
  // await expect( page.locator('.final-text') ).toBeVisible();
  // await expect( page.locator('[data-is-final="false"]') ).toBeVisible();
  // check content append logic

- test('copy button appears for final, clicking copies (and updates status temp)')
  // after final text present
  const copyBtn = page.locator('.copy-btn');
  await expect(copyBtn).toBeVisible();
  await copyBtn.click();
  await expect(page.getByTestId('status')).toContainText('Copied finalized text!');
  // (hard to assert clipboard in pw, use existing pattern or page.evaluate)

- test('no copy button or final span when no finalized content')

### Settings + Persistence (A)
- test('presets update controls and save')
  await page.goto('/');
  const panel = page.locator('.settings-panel');
  await panel.getByRole('button', {name: /Balanced/}).click();
  // assert inputs have expected values (6,0,1.15,true)
  // (may need to check after save or via signals, but E2E via DOM or re-load)

- test('manual input changes update and persist to localStorage')
  // set beam=8 , reload page, assert value 8 (after load from storage)
  // Note: selectedModel also persisted, but since we force whisper-tiny, verify it.

- test('settings visible and functional in static shell too (prod)')

### Volume + Transcript Lifecycle
- test('volume starts at 0, updates >0 only during recording')
- test('transcripts reset on new Start Recording')
- test('partial/final cleared on stop')

### Edge / Other UI
- test('Mic error sets status appropriately without crashing UI')
- test('copy falls back to alert if clipboard fails (rare)')

**Test Organization (design):**
- Use test.describe for groups: UI Load and Shell (whisper-tiny), Recording Controls, Reconnection and Recovery, Visual Feedback (Volume), Transcript and Copy, Settings and Persistence
- Reuse beforeEach mock for WS error cases.
- For happy path transcription effects on UI: either extend ws-protocol or use evaluate to drive WS + assert DOM.
- All tests: await page.goto('/'); + select/verify whisper-tiny radio if not default in shell.
- Timeouts: 5-10s for async UI/WS.
- Assertions prefer testid + text/attr over brittle classes.
- No real audio: rely on fakes + mocks. For real WS+UI, use tiny silence wav if needed (as in ws-protocol).
- Run against prod build for shell coverage + dev for hydrated (but focus prod per history).
- Since whisper-tiny only: in config mocks, hardcode model_id: whisper-tiny ; assume server started accordingly (no other models tested).

**Coverage Gaps to Note in Design (for future):**
- Actual mic audio levels (needs real device or advanced mock)
- Full localStorage + settings roundtrip with selectedModel (but limited to tiny)
- Hydration takeover timing (Qwik specific, may need waitFor or visibleTask effects)
- Multiple reconnects + transcript append correctness (use real-ish WS events)
- Accessibility (roles, labels) - already using getByRole in existing
- Mobile/responsive (if in scope, add viewport)
- Error states from backend (e.g. model load fail for tiny - but assume works)

**Prerequisites for Running Designed Tests:**
- Server: cd .. ; .\run.ps1 --host 127.0.0.1 --port 8000 (or 0.0.0.0)  [whisper-tiny will be chosen in WS config]
- For prod: npm run test:e2e:prod (with testMatch updated later to include new file)
- Fake mic flags already in config.

This design ensures tests verify expected behavior for the listed UI components when whisper-tiny is active. Feature list derived directly from code survey (no assumptions).

**Next (not done now):** After review, implement by creating/editing .spec.ts based on this design doc.
