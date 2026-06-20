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

プロジェクト本体は `asr-model-comparison/` にあります。FastAPI バックエンド、Pure Qwik + Vite フロントエンド、WebSocket リアルタイム文字起こし、再接続UX、Qwen3-ASR / Voxtral / Whisper の単一モデル運用が実装済みです。

ワークスペース側の `docs/` には仕様・バックログ・利用ガイドを置きます。AI生成の修正指示書、調査メモ、実行ログ、ビルド成果物は Git 管理外にしてください。

## 目標フォルダ構造

```text
asr-model-comparison/
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
  logs/
  run.sh / run.bat       # 統合起動スクリプト
  docker-compose.yml     # オプション
```

## コマンド

```bash
# 推奨: フロントエンドをビルドしてバックエンドから配信
cd asr-model-comparison
./run.sh                 # macOS/Linux
.\run.ps1                # Windows
./run.sh --build-only    # ビルドのみ
.\run.ps1 -BuildOnly     # Windows ビルドのみ

# Backend 単独
cd backend
python -m uvicorn app.main:app --reload
pytest
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

**Python 3.12** (`.python-version` を参照)

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
- モデル選択は単一（ラジオボタン）。複数モデル同時比較はスコープ外。
- Qwen3-ASR（0.6B/1.7B）とVoxtral Mini 4B をメイン、Whisper はサブ。

**テスト状況**:
- Playwright E2E: 基本ロード・モデル選択・再接続ボタン存在は安定パス。
- 詳細再接続シナリオは WS モック注入でカバー（自動実行で一部タイミング差あり。手動/headed で機能検証済み）。
- TDD 遵守継続中（新機能追加時は Playwright テスト先行）。

**Phase 進行**:
- Phase 1（WebSocket安定化 + 詳細再接続UI + ドキュメント + TDD）完了。
- 次は Phase 2（リアルタイム視覚的フィードバック：音量メーター/波形、is_final区別、設定パネル、日本語プリセット）へ移行。

**制約の継続遵守**:
- 単一モデルメモリ制約厳守。
- 複数モデル同時ロード/比較は行わない。
- 実用的リアルタイム日本語精度向上（previous_text + 専用クラス + 生成パラメータ自動調整）を最優先。

`AGENTS.md` を参照しながら作業すること。実装前にテスト、テスト後に実行確認を徹底。
