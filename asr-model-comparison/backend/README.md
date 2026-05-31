# Backend (FastAPI)

ASR Model Comparison Platform のバックエンドです。FastAPI を使用して ASR モデルの管理・推論・WebSocket ストリーミングを提供しています。

## 技術スタック

- Python 3.12+
- FastAPI
- Uvicorn
- Pydantic
- (モデルによる) faster-whisper, transformers など

## ディレクトリ構成

```
backend/
├── app/
│   ├── api/
│   │   └── routers/          # API ルーター (transcribe, models, status, websocket)
│   ├── models/               # Pydantic モデル
│   ├── services/
│   │   ├── asr_backends/     # 各 ASR モデルのアダプター (Whisper, Qwen3, Voxtral)
│   │   ├── model_manager.py  # 単一モデル管理のコアロジック
│   │   └── factory.py
│   └── utils/                # ユーティリティ
├── tests/                    # pytest テスト
│   └── audio_samples/        # 実音声テスト用ファイル
├── static/                   # 本番ビルド時のフロントエンド静的ファイル (シングルアプリモード用)
├── requirements.txt
└── pyproject.toml
```

## セットアップ

### 1. 仮想環境の作成

```bash
cd backend
python -m venv .venv
```

### 2. 依存関係のインストール

**Windows (PowerShell)**
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux**
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## 実行方法

### 単独でバックエンドを起動する場合

```bash
# 仮想環境を有効化
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\Activate.ps1 # Windows

cd backend
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- API: `http://localhost:8000`
- ドキュメント: `http://localhost:8000/docs`

### 推奨：プロジェクト全体を起動する場合

プロジェクトルート (`asr-model-comparison/`) から以下のコマンドを実行してください：

```bash
# Windows
.\run.ps1

# macOS / Linux
./run.sh
```

これにより、フロントエンドのビルド + バックエンドの起動が一度に行われます（シングルアプリモード）。

## テスト

```bash
pytest
pytest -m slow          # 実モデルを使った重いテスト
```

## 本番ビルド（シングルアプリモード）

プロジェクトルートから以下を実行すると、フロントエンドをビルドして `backend/static/` に配置します：

```bash
./run.sh --build-only     # macOS/Linux
.\run.ps1 -BuildOnly      # Windows
```

その後、以下で本番向けに起動できます：

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 注意点

- 本プロジェクトは **単一モデル** のみメモリに保持する設計です。
- モデル切り替え時は自動で以前のモデルがアンロードされます。
- WebSocket エンドポイント (`/api/ws/transcribe`) はリアルタイム音声ストリーミング用です。

## 関連ドキュメント

- プロジェクト全体の README: `../README.md`
