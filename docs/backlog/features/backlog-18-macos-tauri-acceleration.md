# ID 18: macOS Rust/Tauri ハードウェアアクセラレーション対応

## 目的

Windows向けに整備済みのRust/Tauri版 `app/` に、macOS Apple Silicon向けの実推論ビルド導線とMetal優先のアクセラレータ選択を追加する。既存WindowsのVulkan/CUDAビルド経路は維持し、macOSでは利用可能なハードウェアアクセラレータを優先しながら安全にCPUへフォールバックする。

## 対象範囲

- Whisper: `whisper-rs` / whisper.cpp のMetal featureを使う。
- Qwen3-ASR: Candle Metal backendを優先し、利用不可時はCPUへフォールバックする。
- Voxtral: Voxtral Realtime対応patched llama.cppのMetal buildを検出し、可能な場合はMetalへoffloadする。Metal runtimeがない場合はCPUで動作する。
- Tauri macOSビルドはTauri CLI経由で行い、Web資産を埋め込む。
- `frontend/` はmacOS対応のために必要な場合を除き編集しない。

## 実装方針

- Windows用 `full-runtime` / `full-runtime-cuda` featureは変更せず、macOS用 `full-runtime-macos` を追加する。
- macOS用npm scriptとビルドラッパーを追加し、`app/dist/AMCP-macos` と `app/dist/web/` を作る。
- `/api/status` と文字起こしレスポンスの `accelerator` / `runtime_backends` でMetal選択とフォールバック状態を確認できるようにする。
- patched llama.cppのmacOS buildでは `lib*.dylib` とMetal runtime artifactを診断し、必要な環境変数をREADMEへ明記する。

## テスト方針

- Rust単体テストでmacOSの優先順位、Metal feature境界、Windows既存優先順位の維持を確認する。
- macOS向けcompile checkを追加し、Whisper/Qwen/Voxtralのfeature境界が壊れていないことを確認する。
- Playwright E2EはRust serverモードに対して既存UI契約を確認する。
- 実モデル検証は `backend/tests/audio_samples/ja_01.mp3` を使い、Whisper/Qwenを優先して実施する。Voxtralはpatched llama.cpp Metal buildとGGUF/mmprojが利用可能な場合に実施する。

## 完了条件

- `npm run test:rust` と `npm run test:e2e` が通る。
- macOS向けcompile checkが通る。
- `validate:macos:*` で日本語サンプル音声の実モデル検証を実行し、結果を確認する。
- `npm run build:macos:app` でmacOS向けTauri成果物を作成できる。
- `AGENTS.md`、ルート `README.md`、`app/README.md` にmacOSビルド・検証手順が記載されている。
