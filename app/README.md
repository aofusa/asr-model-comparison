# AMCP Rust/Tauri App

`app/` は ASR Model Comparison Platform の Rust + Tauri 版アプリケーションです。Python/FastAPIバックエンドをRustへ置き換え、既存のQwikフロントエンドをできるだけ維持したまま、デスクトップアプリとHTTP serverモードの両方で動作させることを目的としています。

## 現在の状態

- `src-tauri/` にRustバックエンドとTauri設定を配置しています。
- 既存UIはリポジトリルートの `frontend/` を利用します。
- Rust serverモードは既存UIが使う主要API契約を提供します。
  - `GET /health`
  - `GET /api/models`
  - `GET /api/status`
  - `POST /api/transcribe`
  - `GET /api/ws/transcribe`
- `auto` / `gpu` / `cpu` のアクセラレータ設定に対応しています。
- 実ASR推論本体はまだ軽量プレースホルダーです。`whisper-rs`、Qwen C FFI、ONNX Runtime連携用のfeature境界を先に用意しています。

## 必要な環境

- Rust 1.95+ / Cargo
- Node.js 24+ / npm
- Playwrightで利用するブラウザ
- Tauriデスクトップビルドを行う場合は、各OS向けのTauri前提環境

初回は `app/` と `frontend/` の依存関係をインストールしてください。

```powershell
cd app
npm install

cd ..\frontend
npm install
```

## 開発起動

### Tauriデスクトップモード

```powershell
cd app
npm run dev
```

`npm run dev` はTauri開発モードを起動します。Tauri設定では `frontend/` のVite開発サーバーを利用します。

### Rust serverモード

```powershell
cd app
npm run server:local
```

既定では `http://127.0.0.1:8000` で起動します。host、port、アクセラレータを指定したい場合は直接Cargoコマンドを使います。

```powershell
cargo run --manifest-path src-tauri/Cargo.toml --bin amcp-server -- server --host 0.0.0.0 --port 8000 --accelerator auto
```

アクセラレータ指定:

- `auto`: デフォルト。GPU/専用アクセラレータを優先し、使えない場合はCPUへフォールバックします。
- `gpu`: GPUを明示的に優先します。利用不可の場合は安全にCPUへフォールバックします。
- `cpu`: GPUを試さずCPUのみを使います。

## ビルド方法

### Tauriアプリをビルド

```powershell
cd app
npm run build
```

このコマンドはTauriのビルドを実行します。Tauri設定により、ビルド前に `frontend/` の本番ビルドも実行されます。

### Rust serverのみをビルド

```powershell
cd app
cargo build --manifest-path src-tauri/Cargo.toml --bin amcp-server
```

## テスト方法

### すべてのappテスト

```powershell
cd app
npm test
```

`npm test` は以下を順に実行します。

- Rust単体テスト
- Playwright E2Eテスト

### Rust単体テストのみ

```powershell
cd app
npm run test:rust
```

主に以下を検証します。

- モデル一覧の互換性
- 単一モデルロード制約
- CPU/GPUアクセラレータ選択とフォールバック
- `/api/models`、`/api/status` のAPI形状

### Playwright E2Eのみ

```powershell
cd app
npm run test:e2e
```

E2EではPlaywright経由で実ブラウザを使い、Rust serverモードに対して以下を検証します。

- `/health` と `/api/models` の応答
- WebSocket `/api/ws/transcribe` のconfig送信、ready応答、文字起こし応答
- `gpu` 指定時のアクセラレータ設定がレスポンスに反映されること

ブラウザを表示して確認したい場合:

```powershell
npm run test:e2e:headed
```

## アプリケーションの使い方

1. `npm run dev` でTauriデスクトップアプリを起動するか、`npm run server:local` でRust serverモードを起動します。
2. UIで音声入力元を選択します。
   - `Mic`: マイク入力
   - `System`: ブラウザの画面共有経由でシステム音声を取得
   - `Window`: 共有ピッカーで選択したウィンドウやアプリの音声を取得
3. 入力言語と翻訳出力を選択します。
4. `Hardware Acceleration` を選択します。
   - 通常は `Auto (prefer GPU)` を推奨します。
   - 問題がある場合は `CPU only` を選ぶと安定性を優先できます。
5. `Advanced settings` でモデルや生成パラメータを調整します。
6. `Start Recording` を押すとWebSocket接続とモデル準備が始まり、準備完了後に音声チャンクが送信されます。
7. 認識結果はライブ表示と履歴に表示されます。

## 実モデル統合方針

現時点のRust実装はAPI/WS契約、アクセラレータ選択、フォールバック、テストを先に固めるための軽量プレースホルダーです。実ASRは以下のfeature境界で段階的に実装します。

- `whisper`: `whisper-rs` / whisper.cpp
- `qwen`: antirez/qwen-asr C実装のFFI
- `voxtral`: `ort` + ONNX Runtime

想定する優先アクセラレータ:

- Whisper
  - Windows/Linux: CUDA -> Vulkan -> CPU
  - macOS: Metal -> Vulkan -> CPU
- Qwen3-ASR
  - macOS: Accelerate/BLAS -> CPU
  - Linux: OpenBLAS -> CPU
  - Windows: BLAS -> CPU
- Voxtral
  - Windows/Linux: CUDA -> CPU
  - macOS: CoreML -> CPU

## 関連ドキュメント

- プロジェクト全体: `../README.md`
- ID 7仕様設計: `../docs/backlog/features/backlog-7-rust-tauri-rewrite.md`
- 既存フロントエンド: `../frontend/README.md`
