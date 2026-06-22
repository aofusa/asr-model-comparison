# AGENTS.md

このファイルは、AIコーディングエージェント（Cursor, Claude Code, Codex など）がこのリポジトリ内のコードを扱う際のガイドラインを提供します。

## プロジェクト概要

ASR Model Comparison Platform (AMCP) — 同一の音声サンプルに対して3つのASR（Automatic Speech Recognition）モデルを比較するためのツール。精度、速度、リソース使用量を測定します。

**比較対象モデル:**
- Whisper (tiny / small / medium / large-v3-turbo)
- Qwen3-ASR (0.6B / 1.7B)
- Voxtral Mini 4B Realtime 2602

**ターゲットプラットフォーム:**
- ROG Ally X (Windows, AMD Ryzen + Radeon)
- M4 MacBook Pro 24GB (macOS, Apple Silicon / MPS)

## 現在の状態

プロジェクト本体はリポジトリルートにあります。FastAPI バックエンド、Rust/Tauri 版 `app/`、Pure Qwik + Vite フロントエンド、WebSocket リアルタイム文字起こし、詳細な再接続UX、音量メーター、チャンク処理フィードバック、履歴表示、Qwen3-ASR / Voxtral / Whisper の単一モデル運用が実装済みです。

`docs/` には仕様とバックログを置きます。公開ユーザー向けの起動方法、WebSocket利用方法、日本語リアルタイム推奨設定は `README.md` に集約します。AI生成の修正指示書、調査メモ、実行ログ、ビルド成果物は Git 管理外にしてください。

## 目標フォルダ構造

```text
repository root/
  backend/
    app/
      main.py            # FastAPI エントリポイント
      api/               # ルート定義
      models/            # Pydantic リクエスト/レスポンススキーマ
      services/          # ASRモデル管理と文字起こしロジック
      utils/             # 音声処理、タイミングヘルパー
    tests/               # pytest テストスイート
      audio_samples/     # テスト音声（日本語、英語、中国語）
    requirements.txt
    pyproject.toml
  frontend/
    src/                 # Qwik アプリケーション
    public/
    package.json
    playwright.config.ts # E2Eテスト
  app/
    src-tauri/           # Rust/Tauri アプリ、serverモード、AMCP.exeビルド
    scripts/
      build-windows-exe.ps1
    dist/                # app\dist\AMCP.exe（生成物、Git管理外）
  run.sh / run.bat       # 統合起動スクリプト
  docs/
    specs.md
    backlog/
```

## コマンド

```bash
# 推奨: フロントエンドをビルドしてバックエンドから配信
./run.sh                 # macOS/Linux
.\run.ps1                # Windows
./run.sh --build-only    # ビルドのみ
.\run.ps1 -BuildOnly     # Windows ビルドのみ
./run.sh --reload        # Python reload + frontend auto rebuild
.\run.ps1 --reload       # Windows reload mode

# Backend 単独
cd backend
python -m uvicorn app.main:app --reload
pytest
USE_REAL_MODELS=1 USE_REAL_WHISPER=1 pytest -s
pytest tests/test_specific.py::test_name

# Frontend
cd ../frontend
npm install
npm run dev
npm run test:e2e
npm run test:e2e:prod

# コード品質
black .
ruff check .
```

### Rust/Tauri版 AMCP.exe

Windows向けの配布用exeは `app/` でTauri CLI経由でビルドする。`cargo build --bin amcp-desktop` だけで直接ビルドすると、本番Webアセットの埋め込みが行われず開発用localhostへ接続しようとするため避けること。

`npm run build:windows:exe` は `full-runtime` featureで `AMCP.exe` をビルドし、Whisper Vulkan、Qwen3-ASR、Voxtral Realtime patched llama.cpp Vulkan の実推論経路を含める。Whisper Vulkan のWindowsビルドは深いtargetパスで失敗しやすいため、ビルドスクリプトは既定で `CARGO_TARGET_DIR=C:\t` と `CMAKE_BUILD_PARALLEL_LEVEL=1` を設定する。Qwen3-ASR のGPU実行はCandle CUDA featureが必要なため、CUDA環境では `full-runtime-cuda` を使い、CUDAがないWindows/AMD環境ではCPUフォールバックになる。

Voxtral Realtime用のパッチ済み `llama.cpp` が `.tmp\llama-cpp-voxtral-pr20638` または `C:\amcp-build\llama-vulkan` にある場合はビルドスクリプトが自動検出する。別の場所にある場合だけ、同じPowerShellセッションでsource/build/binを指定する。

```powershell
cd app

$patchedRoot="C:\Users\5000e\Documents\aofusa\ai\asr-model-comparison-project\.tmp\llama-cpp-voxtral-pr20638"
$patchedBuild="C:\amcp-build\llama-vulkan"
$env:AMCP_VOXTRAL_PATCHED_LLAMA_DIR=$patchedRoot
$env:AMCP_VOXTRAL_PATCHED_LLAMA_LIB_DIR=$patchedBuild
$env:AMCP_VOXTRAL_PATCHED_LLAMA_BIN_DIR="$patchedBuild\bin"

npm run build:windows:exe
```

成果物は `app\dist\AMCP.exe` と serverモード配信用の `app\dist\web\`。serverモードは既定で `0.0.0.0` に待ち受け、ブラウザで `http://127.0.0.1:8000/` または起動ログのLAN URLを開くとフロントエンドが表示される。起動用PowerShellで以下を実行して確認する。

```powershell
.\dist\AMCP.exe --server --port 8000
```

別PowerShellから以下を実行し、`service=amcp-rust-backend` が返れば成功。検証後は起動したプロセスを停止すること。

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/status
Invoke-WebRequest http://127.0.0.1:8000/
```

リモート端末から接続できない場合は、起動ログに表示されるLAN IPのURLとWindows Defender Firewallの受信許可を確認する。管理者PowerShellで許可する例:

```powershell
New-NetFirewallRule -DisplayName "AMCP Rust Server 8000" -Direction Inbound -Program (Resolve-Path .\dist\AMCP.exe) -Action Allow -Protocol TCP -LocalPort 8000 -Profile Private
```

### Rust/Tauri版 macOS / Apple Silicon

macOS向けのTauriビルドは `app/` の `npm run build:macos:app` を使う。`full-runtime-macos` featureで Whisper Metal、Qwen3-ASR Candle Metal、Voxtral ExecuTorch Metal runner を優先する。Windows向け `full-runtime` / `full-runtime-cuda` とはfeatureを分け、WindowsのVulkan/CUDA経路を壊さないこと。
Whisper MetalのネイティブビルドにはCMakeが必要。`build:macos:app` は `MACOSX_DEPLOYMENT_TARGET=11.0` を既定で設定する。

```bash
cd app
npm run build:macos:app
```

Voxtral RealtimeのMetal実推論はmacOSではExecuTorch Metal runnerを優先する。runnerはローカルビルドし、Metal `.pte` モデル一式はHugging Faceの共通キャッシュへ配置してrepo/file名で解決する。

```bash
export AMCP_VOXTRAL_RUNTIME=executorch
export AMCP_VOXTRAL_EXECUTORCH_RUNNER_PATH="$HOME/executorch/cmake-out/examples/models/voxtral_realtime/voxtral_realtime_runner"
export AMCP_VOXTRAL_EXECUTORCH_REPO_ID="mistral-experimental/Voxtral-Mini-4B-Realtime-2602-ExecuTorch"
export AMCP_VOXTRAL_EXECUTORCH_MODEL_FILE="model-metal-int4.pte"
export AMCP_VOXTRAL_EXECUTORCH_PREPROCESSOR_FILE="preprocessor.pte"
export AMCP_VOXTRAL_EXECUTORCH_TOKENIZER_FILE="tekken.json"
export AMCP_VOXTRAL_EXECUTORCH_DYLD_LIBRARY_PATH="/usr/lib:$(brew --prefix libomp)/lib"
npm run validate:macos:voxtral:executorch -- --audio "../backend/tests/audio_samples/ja_01.mp3" --model-id voxtral-mini-4b --language ja --json
```

パッチ済み `llama.cpp` Metal経路も代替として維持する。使う場合は、パッチ済み `llama.cpp` をMetal有効で共有ライブラリビルドし、同じシェルでsource/buildを指定する。

```bash
cmake -S ../.tmp/llama-cpp-voxtral-pr20638 \
  -B ../.tmp/llama-cpp-voxtral-pr20638/build-amcp-metal-release \
  -DGGML_METAL=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_BACKEND_DL=ON
cmake --build ../.tmp/llama-cpp-voxtral-pr20638/build-amcp-metal-release --config Release

export AMCP_VOXTRAL_PATCHED_LLAMA_DIR="$(cd ../.tmp/llama-cpp-voxtral-pr20638 && pwd)"
export AMCP_VOXTRAL_PATCHED_LLAMA_LIB_DIR="$(cd ../.tmp/llama-cpp-voxtral-pr20638/build-amcp-metal-release && pwd)"
export AMCP_VOXTRAL_PATCHED_LLAMA_LINK_METAL=1
npm run build:macos:app
```

Whisper/QwenのみをMetalで検証し、Voxtral ExecuTorch runnerがまだない場合は以下のfeature overrideを使う。

```bash
AMCP_MACOS_FEATURES=desktop,whisper-metal,qwen-metal npm run build:macos:app
```

実モデル検証は必ず日本語サンプル音声で行う。

```bash
npm run validate:macos:whisper -- --audio "../backend/tests/audio_samples/ja_01.mp3" --model-id whisper-tiny --language ja --json
npm run validate:macos:qwen -- --audio "../backend/tests/audio_samples/ja_01.mp3" --model-id qwen3-asr-0.6b --language ja --json
AMCP_VOXTRAL_RUNTIME=executorch npm run validate:macos:voxtral:executorch -- --audio "../backend/tests/audio_samples/ja_01.mp3" --model-id voxtral-mini-4b --language ja --json
```

## アーキテクチャ

**バックエンド:** FastAPI が以下の2つのAPIエンドポイントを提供します:
- `GET /api/models` — 利用可能なモデルとメタデータを一覧で返す
- `POST /api/transcribe` — 音声（FormData）とモデル選択を受け取り、文字起こし結果 + 処理時間を返す

**フロントエンド:** Qwik + Qwik UI。ビルド済みの静的ファイルはFastAPIが配信します。UIの主な機能: モデル選択、マイク録音（MediaRecorder API）、ファイルアップロード、経過時間付きの結果表示。

**モデル管理:** メモリ上には常に1つのASRモデルのみがロードされます。モデルを切り替える場合は、現在のモデルを必ずアンロードしてから次のモデルをロードしてください。GPUアクセラレーションにはCUDA（NVIDIA）またはMPS（Apple Silicon）を使用します。

## 重要な制約

1. **メモリ内の単一モデル** — 2つのモデルを同時にロードしない。モデル切り替え時は必ず先にアンロードする。
2. **後処理の禁止** — Gemma3/Bonsai による後処理は明確にスコープ外とします。
3. **処理時間表示** — リクエスト開始から完了までの経過時間を必ず表示する。
4. **TDD必須** — バックエンド実装の前にpytestテストを、フロントエンド機能実装の前にPlaywrightテストを作成する。すべてのフェーズはテストの合格を必須とする。

## 必須テスト

**pytest (backend):**
- モデル一覧取得
- テスト音声を用いた単一モデルの文字起こし
- モデル切り替え（アンロード → ロード）
- 処理時間計測
- 無効な入力に対するエラーハンドリング

**Playwright (frontend E2E):**
- ページロード
- モデル選択
- マイク録音 → 送信 → 結果表示
- ファイルアップロードフロー
- 処理時間表示

## 主要な依存関係

**Python 3.12+**（現在のローカル検証は Python 3.13 でも通過）

Backend: `fastapi`, `faster-whisper`, `qwen-asr`, `transformers`, `torchaudio`, `pydub`, `python-multipart`, `pytest`, `pytest-asyncio`, `httpx`, `psutil`（リソース監視用、オプション）

Frontend: `qwik`, `playwright` (Vite)

## 現在の実装ノート（2026-06 更新）

**フロントエンドの現状**:
- **Pure Qwik + Vite**（Qwik City は開発サーバーの安定性と環境差異（特にWindows）解消のため撤去）。
- リアルタイム音声入力（MediaRecorder 2秒チャンク + WebSocket `/api/ws/transcribe`）に強く特化。
- **再接続UXが最大の特徴**:
  - 詳細なオレンジ色再接続バナー（ライブカウントダウン、指数バックオフ、Attempt X/5、「Retry Immediately」ボタン、「Stop Recording」ボタン、「transcript is preserved」注記）。
  - Qwik Signals で `isReconnecting` / `nextReconnectIn` / `reconnectAttempts` を精密管理。
  - 再接続時に自動で最新 `previous_text` を送信し文脈継続。
- 音量メーター、無音PCM送信抑止、チャンク処理中の多重送信防止、処理時間/チャンク番号/バイト数の表示を実装済み。
- 確定結果は履歴ペインに蓄積し、過去の文字起こしをスクロールで確認可能。
- モデル選択は単一（ラジオボタン）。複数モデル同時比較はスコープ外。
- Qwen3-ASR（0.6B/1.7B）とVoxtral Mini 4B をメイン、Whisper はサブ。

**テスト状況**:
- Backend pytest: `USE_REAL_MODELS=1 USE_REAL_WHISPER=1 pytest -s` で slow / 実モデル系を含めて通過済み。
- Playwright dev E2E: `npm run test:e2e` で通過済み。
- Playwright production E2E: `run.ps1 -BuildOnly` 後にバックエンド起動し、`npm run test:e2e:prod` で通過済み。
- 詳細再接続シナリオは WS モック注入でカバー。
- TDD 遵守継続中（新機能追加時は Playwright テスト先行）。

**Phase 進行**:
- Phase 1（WebSocket安定化 + 詳細再接続UI + ドキュメント + TDD）完了。
- Phase 2（音量メーター、チャンク処理フィードバック、is_final区別、設定パネル、日本語プリセット）も主要項目は実装済み。
- Rust/Tauri 化は `app/` に実装済み。Windowsでは `npm run build:windows:exe` で `app\dist\AMCP.exe` を作成し、同一exeをデスクトップモードまたは `--server` 付きHTTP serverモードで起動できる。

**制約の継続遵守**:
- 単一モデルメモリ制約厳守。
- 複数モデル同時ロード/比較は行わない。
- 実用的リアルタイム日本語精度向上（previous_text + 専用クラス + 生成パラメータ自動調整）を最優先。

`AGENTS.md` を参照しながら作業すること。実装前にテスト、テスト後に実行確認を徹底。
