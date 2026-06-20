# Backlog 9-13: ASR Accuracy, Translation Output, and Usable UI

## 対象 ID

- ID 9: 音声認識精度を向上する。
- ID 10: 翻訳有効化時、翻訳前と翻訳後の両方を出力する。
- ID 11: 翻訳しない場合でも Qwen3-ASR 0.6B・Qwen3-ASR 1.7B・Voxtral Mini 4B で正常に認識した内容を出力する。
- ID 12: README.md にフロントエンド画面の使い方と機能説明をユーザ向けに作成する。
- ID 13: フロントエンド画面を非エンジニアでも使いやすいように整理する。

## 仕様

### ASR 精度改善

- 日本語リアルタイム認識では `beam_size`、`temperature`、`repetition_penalty`、`previous_text` を維持し、Qwen3-ASR / Voxtral の専用クラス利用を標準とする。
- WebSocket 再接続時は原文認識結果を `previous_text` として送る。翻訳文は ASR 文脈には使わない。
- 無音チャンクは送信前またはサーバ側で抑止し、チャンク処理の可視フィードバックを維持する。

### 翻訳出力契約

WebSocket `transcription` / `final` と REST `POST /api/transcribe` は以下を返す。

- `text`: 画面で主表示するテキスト。翻訳有効時は翻訳結果、無効時は原文認識結果。
- `transcript_text`: 翻訳前の原文 ASR 結果。
- `translated_text`: 翻訳有効時の翻訳結果。翻訳無効時は `null`。
- `accumulated_transcript_text`: WebSocket セッション内で蓄積した原文認識結果。
- `accumulated_translated_text`: WebSocket セッション内で蓄積した翻訳結果。

翻訳無効時は、Qwen3-ASR 0.6B / 1.7B / Voxtral Mini 4B の `text` と `transcript_text` が同じ認識結果になり、`translated_text` は `null` になる。

### UI 設計

- 画面上部に「音声を選ぶ」「認識と翻訳を選ぶ」「開始する」の流れを表示する。
- モデル選択、音声入力、言語/翻訳設定を段階的なセクションへ分ける。
- 高度な生成パラメータは `Advanced accuracy settings` にまとめる。
- 結果表示は `Heard Text` と `Translation` を分け、翻訳が有効な場合も原文を確認できる。
- 履歴には翻訳結果だけでなく、原文と翻訳結果の対応を残す。

## テスト設計

- Backend unit:
  - 翻訳有効時に `transcript_text` と `translated_text` が両方返る。
  - 翻訳無効時に Qwen3-ASR / Voxtral の原文が `text` と `transcript_text` に残る。
  - Qwen3-ASR は翻訳前の ASR 出力を保持し、翻訳後テキストを別フィールドへ格納する。
  - Voxtral は翻訳有効時も原文 ASR と翻訳生成を分けて行う。
- Frontend E2E:
  - 非エンジニア向けの主要ワークフロー表示を確認する。
  - 実デバイス入力は使わず、WebSocket / MediaRecorder モックで原文枠と翻訳枠の表示を確認する。

## 実デバイステストの扱い

ブラウザや OS の権限、共有ピッカー、実音声デバイスに依存するため、実デバイスから音声を取得する自動テストは対象外とする。代わりに WebSocket と音声 API をモックし、ブラウザ UI とプロトコル契約を検証する。
