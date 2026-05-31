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

### Windows
```powershell
cd asr-model-comparison
.\run.ps1                    # 推奨
# または
run.bat
```

### macOS / Linux
```bash
cd asr-model-comparison
chmod +x run.sh
./run.sh
```

アプリケーションは `http://localhost:8000` で利用可能になります。

外部からアクセス可能にする場合：
```bash
./run.sh --host 0.0.0.0 --port 8000
```

## 本番ビルド

ビルドのみを実行したい場合は `--build-only` オプションを使用してください。

```bash
# Windows
.\run.ps1 -BuildOnly
# または
run.bat --build-only

# macOS / Linux
./run.sh --build-only
```

## リアルタイム利用のための推奨設定（日本語）

日本語で高精度なリアルタイム認識を行う場合、以下の設定を推奨します。

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
- `beam_size=6`：認識精度を大幅に向上（特にQwen3/Voxtralで有効）
- `temperature=0.0`：幻覚を抑え一貫性を高める
- `repetition_penalty=1.15`：長い日本語文で繰り返しを防ぐ
- `previous_text`：直前の認識結果を渡すことで、文脈を維持し精度を向上させる（リアルタイムで非常に重要）

## WebSocket ストリーミング（推奨）

低遅延で実用的なリアルタイム認識には WebSocket (`/api/ws/transcribe`) を使用します。

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

### ストリーム終了
クライアントから `{"type": "end"}` を送信すると、蓄積された最終結果が返されます。

## 再接続戦略（推奨）

- エラーや予期しない切断時は 1〜3 秒待ってから再接続
- 再接続時に最新の `previous_text` を config に含めて送信
- これにより文脈が途切れず、高い精度を維持できます

## モデル選択の目安

| モデル                | おすすめ用途                     | 特徴 |
|-----------------------|----------------------------------|------|
| qwen3-asr-0.6b        | バランス最強（推奨）             | 高速で実用的 |
| qwen3-asr-1.7b        | 最高精度を求める場合             | 重いが認識品質が高い |
| voxtral-mini-4b       | 音響特性が異なる代替案           | 専用クラス対応が良好 |

## ディレクトリ構成

```
asr-model-comparison/
├── backend/                     # FastAPI バックエンド
│   ├── app/                     # メインアプリケーションコード
│   ├── tests/                   # テスト（audio_samples 含む）
│   └── README.md                # バックエンド固有のドキュメント
├── frontend/                    # Qwik + Vite フロントエンド
│   ├── src/                     # ソースコード
│   ├── tests/                   # Playwright E2Eテスト
│   └── README.md                # フロントエンド固有のドキュメント
├── run.sh / run.bat / run.ps1   # 統合起動スクリプト（--build-only 対応）
└── README.md                    # 本ファイル（プロジェクト全体）
```

## ライセンス

Apache License 2.0

## 著者

Copyright aofusa. 2026


