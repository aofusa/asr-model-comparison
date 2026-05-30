**プロジェクト要求定義書仕様書**（修正版）

---

### 1. プロジェクト概要

**プロジェクト名**: ASR Model Comparison Platform (AMCP)

**目的**:  
Whisper（tiny/small/medium/large-v3-turbo）、Qwen3-ASR（0.6B/1.7B）、Voxtral Mini 4B Realtime 2602の各ASRモデルの性能を同一環境同一音声で比較評価するためのWebアプリケーションを構築する。

主な比較軸：  
- 文字起こし精度（日本語英語中国語）
- 処理速度（推論時間）
- リソース使用量（VRAM、CPU）

**ターゲット実行環境**:  
- ROG Ally X（Windows / AMD Ryzen + Radeon）
- M4 MacBook Pro 24GB（macOS / Apple Silicon）

---

### 2. 機能要求（Functional Requirements）

#### 2.1 フロントエンド機能
- **音声録音機能**: Webブラウザのマイクからリアルタイム録音（MediaRecorder API使用）
- **音声ファイルアップロード機能**: 録音済み音声ファイル（.wav, .mp3, .m4aなど）のアップロード対応
- **モデル選択**: 以下のモデルをUIから**単一選択**可能
  - Whisper: tiny, small, medium, large-v3-turbo
  - Qwen3-ASR: 0.6B, 1.7B
  - Voxtral: Mini 4B Realtime 2602
- **処理フロー**:
  1. モデルを選択
  2. 録音開始 / 停止 または ファイルアップロード
  3. 選択したモデルで文字起こしを実行
  4. 結果表示（原文テキスト、処理時間、メトリクス）
- **実行時間表示**: ASR処理の開始から完了までの**処理時間（秒）**を必ず表示
- **UI技術**: **Qwik + Qwik UI**

#### 2.2 バックエンド機能
- **APIエンドポイント**:
  - `GET /api/models`：利用可能なモデル一覧を返す
  - `POST /api/transcribe`：音声データ + 選択モデルを受け取り、文字起こし結果を返す
- **モデル管理**: 
  - モデルは**常に単一**でロードする（同時ロードは行わない）
  - モデル切り替え時は前回のモデルをアンロードし、新しいモデルをロード
- **ログ記録**: 各実行ごとに処理時間、推定メモリ使用量を記録（後で確認可能）

**注意**: 現時点ではGemma3 270MやBonsai 1.7Bによる後処理（句読点補正整形など）は**実装しない**。将来的に追加可能とするため、API設計は拡張性を考慮する。

---

### 3. 非機能要求（Non-Functional Requirements）

**パフォーマンス**:
- モデルロード時間と推論時間を最適化
- 処理時間の正確な計測と表示

**リソース制約**（24GB環境想定）:
- Qwen3-ASR 0.6B：軽量（推奨4-6GB）
- Qwen3-ASR 1.7B：バランス（推奨8-12GB）
- Voxtral 4B Realtime：重め（推奨16GB以上、量子化を検討）

**セキュリティ**:
- 完全ローカル実行前提
- 音声データは一時フォルダまたはメモリ上で処理

**互換性**:
- Python 3.11以上
- CUDA（NVIDIA） / MPS（Apple Silicon）両対応

---

### 4. 技術仕様（Technical Specification）

#### 4.1 バックエンド
- **言語**: Python 3.11+
- **フレームワーク**: **FastAPI**
- **モデル推論**:
  - Whisper：`faster-whisper`
  - Qwen3-ASR：公式 `qwen-asr` パッケージ
  - Voxtral：`transformers`
- **オーディオ処理**: torchaudio, pydub
- **モデルロード戦略**: Lazy Loading + 単一モデル保持（切り替え時にアンロード）

#### 4.2 フロントエンド
- **フレームワーク**: **Qwik + Qwik UI**
- **ビルド**: Pythonバックエンドから静的ファイル（`dist`フォルダ）をserve
- **通信**: REST API（JSON形式）
- **UIコンポーネント**:
  - モデル選択（ラジオボタンまたはセレクトボックス）
  - 録音ボタン（リアルタイム波形表示推奨）
  - 結果表示エリア（テキスト + 処理時間）
  - 処理中ローディング表示

#### 4.3 音声処理フロー
1. ブラウザでマイク録音（`navigator.mediaDevices.getUserMedia` + `MediaRecorder`）
2. BlobデータをFormDataでFastAPIに送信
3. サーバー側で音声ファイルを処理
4. 処理時間を計測（開始〜完了）
5. 結果（テキスト + 処理時間）をJSONで返却

---

### 5. モデル別リソース目安（参考）

| モデル                        | ディスク容量   | 最小VRAM   | 推奨VRAM    | 速度特性          |
|-------------------------------|----------------|------------|-------------|-------------------|
| Whisper tiny                 | ~150MB        | 1.5GB     | 2GB        | 最速             |
| Whisper small/medium         | 0.5-1.5GB     | 3-6GB     | 6-8GB      | バランス         |
| Whisper large-v3-turbo       | ~1.6GB        | 5-6GB     | 8-10GB     | 高精度高速     |
| Qwen3-ASR 0.6B               | 1.2-1.8GB     | 2.5GB     | 4-6GB      | **超高速**       |
| Qwen3-ASR 1.7B               | 3.5-4.5GB     | 5.5GB     | 8-12GB     | **バランス最強** |
| Voxtral Mini 4B Realtime     | 8-9.5GB       | 12GB      | 16GB+      | 高精度Realtime |

---

### 6. 開発テスト計画

**フェーズ**:
1. 環境構築 & 各ASRモデルの単体テスト
2. FastAPIバックエンド実装（単一モデルロード機能）
3. Qwikフロントエンド実装
4. 処理時間計測表示機能
5. 最適化（量子化、メモリ管理、モデル切り替え処理）

**テスト音声**: 日本語英語中国語の同一内容の音声を複数用意

**Deliverables**:
- 完全なプロジェクトフォルダ構造
- `requirements.txt`
- 起動スクリプト（`run.sh`, `run.bat`）
- 詳細な`README.md`（環境構築手順、量子化設定含む）

---

この修正版仕様書をAIエージェントに渡して開発を進めてください。

**必要であれば**、以下の追加資料も作成できます：
- 詳細なプロジェクトフォルダ構成案
- FastAPIのAPI設計（OpenAPI仕様）
- Qwikフロントエンドの画面遷移案

さらに修正や追加の指示があればお知らせください。
