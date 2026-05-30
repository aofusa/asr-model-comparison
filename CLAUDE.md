# CLAUDE.md

このファイルは、Claude Code (claude.ai/code) がこのリポジトリ内のコードを扱う際のガイドラインを提供します。

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

プロジェクトは初期計画段階です。すべてのドキュメントは `docs/` に日本語で格納されています。実装コードは `asr-model-comparison/` ディレクトリ内に配置します。

## 目標フォルダ構造

```
asr-model-comparison/
  backend/
    app/
      main.py            # FastAPI エントリポイント
      api/               # ルート定義
      models/            # Pydantic リクエスト/レスポンススキーマ
      services/          # ASRモデル管理と文字起こしロジック
      utils/             # 音声処理、タイミングヘルパー
    tests/               # pytest テストスイート
    requirements.txt
    pyproject.toml
  frontend/
    src/                 # Qwik アプリケーション
    public/
    package.json
    playwright.config.ts # E2Eテスト
  audio_samples/         # テスト音声（日本語、英語、中国語）
  logs/
  run.sh / run.bat       # 統合起動スクリプト
  docker-compose.yml     # オプション
```

## コマンド（実装後）

```bash
# Backend
cd asr-model-comparison/backend
uvicorn app.main:app --reload
pytest
pytest tests/test_specific.py::test_name   # 単一テスト

# Frontend
cd asr-model-comparison/frontend
npm install
npm run dev
npx playwright test                        # E2Eテスト

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

Frontend: `qwik`, `@qwikui/*`, `playwright`
