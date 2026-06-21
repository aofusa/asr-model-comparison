# ASR Model Comparison Platform (AMCP)

Qwen3-ASR および Voxtral を中心とした、実用的なリアルタイム音声認識プラットフォームです。Whisper はサブ（比較用）として位置づけています。

本プロジェクトでは、**単一モデル運用**、ブラウザマイクによる高品質な**日本語リアルタイム文字起こし**、文脈を維持した**堅牢な再接続機能**を重視しています。

## 特徴

- **シングルアプリ展開**  
  フロントエンドをビルドした後、FastAPIバックエンド1プロセスだけで API + フロントエンド全体を提供します。
  ログは標準出力に出力されます（ファイル保存したい場合はリダイレクトしてください）。

- **リアルタイム WebSocket ストリーミング**  
  低遅延のマイク入力と `previous_text` による文脈維持に対応。

- **優れた再接続 UX**  
  指数バックオフ、再接続時の文脈自動継続、ネットワーク障害時の明確なフィードバックを提供。

- **日本語最適化設定**  
  日本語精度を高めるための推奨生成パラメータを採用。

- **専用モデルクラス対応**  
  Qwen3-ASR / Voxtral の高品質推論パスをサポート。

## クイックスタート

初回実行時は、バックエンドの Python 仮想環境とフロントエンドの Node.js 依存関係を準備してください。

### Windows 初回セットアップ
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

cd ..\frontend
npm install
```

ROG Ally X など CUDA のない Windows / AMD 環境では、CPU 版 PyTorch を明示的に入れると安定します。

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### macOS / Linux 初回セットアップ
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

cd ../frontend
npm install
```

Apple Silicon では通常の PyTorch インストールで MPS が利用されます。初回の実モデル利用時は Hugging Face などからモデルがダウンロードされるため、数分かかる場合があります。

### Windows
```powershell
.\run.ps1                    # 推奨
# または
run.bat
```

### macOS / Linux
```bash
chmod +x run.sh
./run.sh
```

アプリケーションは `http://localhost:8000` で利用可能になります。

外部からアクセス可能にする場合：
```bash
./run.sh --host 0.0.0.0 --port 8000
```

## 画面の使い方

AMCP は、音声を選んで文字起こしし、必要に応じて翻訳も確認するためのアプリです。専門知識がなくても、画面上部の 3 つの流れに沿って操作できます。

1. **音声入力を選ぶ**  
   `Microphone` は自分のマイク音声、`System / tab audio` はブラウザの画面共有で共有した音声、`Window / app audio` は共有ピッカーで選んだウィンドウやアプリの音声を使います。システム音声やウィンドウ音声は、ブラウザと OS が「音声を共有」に対応している必要があります。

2. **認識モデルを選ぶ**  
   動作確認には軽い `Whisper Tiny`、日本語のリアルタイム認識には `Qwen3-ASR 0.6B` または `Voxtral Mini 4B` を推奨します。モデルは同時に複数使わず、常に 1 つだけ選びます。

3. **言語と翻訳を選ぶ**  
   `Input Language` は話している言語です。通常は `Japanese` または `Auto Detect` を選びます。翻訳が不要な場合は `Translation Output` を `No Translation` にします。翻訳を有効にした場合も、画面には翻訳前の `Heard Text` と翻訳後の `Translation` の両方が表示されます。

4. **Start Recording を押す**  
   初回はブラウザからマイクや共有音声の許可を求められます。許可後、モデルの準備が完了するとリアルタイムに文字起こしが始まります。

5. **結果を確認する**  
   `Heard Text` には直近の音声認識原文、`Translation` には直近の翻訳結果が表示されます。古いセグメントは順次 `Transcript History` に移動するため、ライブ表示では最新の内容だけを読みやすく確認できます。`Input Level` で音量、`Last chunk` で直近の処理時間とチャンク情報を確認できます。

6. **Stop で終了する**  
   録音を止めても、直前の結果と履歴は画面に残ります。通信が切れた場合は自動再接続し、再接続後も直前の文脈を引き継ぎます。

## ビルドと起動

`run.ps1` / `run.sh` / `run.bat` は、`--build-only` を付けない通常起動でも毎回フロントエンドを本番ビルドし、最新の成果物を `backend/static/` にコピーしてからバックエンドを起動します。これにより FastAPI 1プロセスで API と最新フロントエンドを配信できます。

ビルドだけ実行したい場合は `--build-only` オプションを使用してください。

```bash
# Windows
.\run.ps1 -BuildOnly
# または
run.bat --build-only

# macOS / Linux
./run.sh --build-only
```

## リアルタイム利用のための推奨設定（日本語）

日本語で高精度なリアルタイム認識を行う場合、Qwen3-ASR または Voxtral を選択し、以下の設定を推奨します。Whisper は軽量検証や比較用として有用ですが、本プロジェクトでは Qwen3-ASR / Voxtral を主軸にしています。

```json
{
  "model_id": "qwen3-asr-0.6b",
  "language": "ja",
  "beam_size": 6,
  "temperature": 0.0,
  "repetition_penalty": 1.15,
  "return_timestamps": true,
  "previous_text": "前回の認識テキスト"
}
```

### 主なパラメータの意味
- `language="ja"`：日本語出力を強制
- `beam_size=6`：認識精度を向上（特にQwen3/Voxtralで有効。5〜8程度が目安）
- `temperature=0.0`：幻覚を抑え一貫性を高める
- `repetition_penalty=1.1〜1.15`：長い日本語文で繰り返しを防ぐ
- `previous_text`：直前の認識結果を渡すことで、文脈を維持し精度を向上させる
- `return_timestamps=true`：ライブ字幕やチャンク単位のUI更新で扱いやすくする

## WebSocket ストリーミング（推奨）

低遅延で実用的なリアルタイム認識には WebSocket (`/api/ws/transcribe`) を使用します。ブラウザ側はマイク入力を短い音声チャンクに分けて送信し、バックエンドは接続ごとにモデルを1回だけロードします。

### 基本的な流れ
1. WebSocket に接続
2. すぐに config メッセージを送信
3. `{"type": "ready"}` を待つ（モデルは1回だけロード）
4. バイナリ音声チャンクを送信（1〜3秒程度推奨）
5. 構造化された結果を受信

### Config メッセージ例
```json
{
  "type": "config",
  "model_id": "qwen3-asr-0.6b",
  "language": "ja",
  "beam_size": 6,
  "temperature": 0.0,
  "repetition_penalty": 1.15,
  "use_dedicated_class": true,
  "return_timestamps": true
}
```

### 受信メッセージ例
```json
{
  "type": "transcription",
  "model_id": "qwen3-asr-0.6b",
  "text": "翻訳または認識された表示用テキスト",
  "transcript_text": "翻訳前の認識テキスト",
  "translated_text": "翻訳後のテキスト（翻訳無効時は null）",
  "is_final": false,
  "chunks": [],
  "processing_time_seconds": 1.23,
  "had_speech": true,
  "chunk_index": 1,
  "accumulated_text": "これまでの認識結果",
  "accumulated_transcript_text": "これまでの翻訳前認識結果",
  "accumulated_translated_text": "これまでの翻訳結果"
}
```

### ストリーム終了
クライアントから `{"type": "end"}` を送信すると、蓄積された最終結果が返されます。

## 再接続戦略（推奨）

- エラーや予期しない切断時は 1〜3 秒待ってから再接続
- 再接続時に最新の `previous_text` を config に含めて送信
- フロントエンドは指数バックオフ、即時再試行、録音停止、現在の文字起こし保持をUIで表示
- これにより文脈が途切れず、長いリアルタイムセッションでも精度を維持できます

### 運用上の注意
- 初回利用時は Hugging Face などからモデルがダウンロードされるため、時間とディスク容量が必要です。
- Qwen3-ASR 1.7B や Voxtral Mini 4B は重いため、CPU環境ではチャンク処理に数秒かかることがあります。
- メモリ不足やモデルロード失敗時は 503 エラーになります。まず `qwen3-asr-0.6b` または `whisper-tiny` で動作確認してください。
- ブラウザマイクはHTTPSまたはlocalhostでの利用が前提です。リモートHTTPではブラウザがマイクをブロックします。

## モデル選択の目安

| モデル                | おすすめ用途                     | 特徴 |
|-----------------------|----------------------------------|------|
| whisper-tiny          | 動作確認・軽量E2E                | CPUでも軽く、プロトコル検証に向く |
| qwen3-asr-0.6b        | バランス最強（推奨）             | 高速で実用的 |
| qwen3-asr-1.7b        | 最高精度を求める場合             | 重いが認識品質が高い |
| voxtral-mini-4b       | 音響特性が異なる代替案           | 専用クラス対応が良好 |

## ディレクトリ構成

```
asr-model-comparison-project/
├── backend/                     # FastAPI バックエンド
│   ├── app/                     # メインアプリケーションコード
│   ├── tests/                   # テスト（audio_samples 含む）
│   ├── static/                  # ビルド時に生成（Git管理外）
│   └── README.md                # バックエンド固有のドキュメント
├── frontend/                    # Qwik + Vite フロントエンド
│   ├── src/                     # ソースコード
│   ├── tests/                   # Playwright E2Eテスト
│   └── README.md                # フロントエンド固有のドキュメント
├── run.sh / run.bat / run.ps1   # 統合起動スクリプト（--build-only 対応）
├── docs/                        # 仕様、バックログ
└── README.md                    # 本ファイル（プロジェクト全体）
```

## トラブルシューティング

- ROG Ally X でメモリ不足になる場合は、`whisper-tiny` や `whisper-small` から試してください。
- 初回推論が遅い場合は、モデルのダウンロードや初期ロード中の可能性があります。
- `"CUDA not available"` は AMD / CPU 環境では正常です。CPU または利用可能なアクセラレーションへフォールバックします。
- フロントエンド配信に必要な `backend/static/` は通常起動時にも自動更新されます。サーバーを起動せずに更新だけしたい場合は `run.ps1 -BuildOnly` または `./run.sh --build-only` を使ってください。

## ライセンス

Apache License 2.0

## 著者

Copyright 2026 aofusa


