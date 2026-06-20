# 入力元音声の選択機能

## 対象

Backlog ID 5: 現在のマイク入力に加えて、端末上で流れる音や特定ウィンドウ/アプリケーションの音を選択してリアルタイム ASR を開始できるようにする。

## 実装可否

現在の構成（Python FastAPI バックエンド + ブラウザフロントエンド）で実装可能。ただし、音声入力元の選択と権限管理はブラウザ API の制約を受ける。

- マイク: `navigator.mediaDevices.getUserMedia({ audio: true })` で取得する。
- 端末/タブ/画面共有音声: `navigator.mediaDevices.getDisplayMedia({ video: true, audio: ... })` で取得する。
- 特定ウィンドウ/アプリケーション: ブラウザの画面共有ピッカーでユーザーが選択できる範囲に依存する。OS とブラウザによっては、ウィンドウ単位で音声を共有できない、または「音声を共有」チェックが必要になる。

FastAPI 側はブラウザから届く音声チャンクを既存の WebSocket `/api/ws/transcribe` で処理できるため、入力元ごとの追加ネイティブ実装は不要。

## 仕様

1. フロントエンドに `Audio Input Source` UI を追加する。
2. 選択肢は以下とする。
   - `Microphone`: 従来のマイク入力。
   - `System / tab audio`: 画面共有ピッカーで端末・タブ等の音声を共有する。
   - `Window / app audio`: 画面共有ピッカーで特定ウィンドウ、アプリ、タブを選ぶ。
3. `Start Recording` 実行時に、選択中の入力元に応じて `getUserMedia` または `getDisplayMedia` を呼び分ける。
4. `getDisplayMedia` で音声トラックが返らない場合は録音を開始せず、共有ピッカーで音声共有を有効にする旨を表示する。
5. WebSocket config に `audio_source` を含め、バックエンドは `ready` 応答にも同値を含める。
6. 音声チャンクの送信、PCM 変換、無音抑止、再接続、ASR 推論は既存の処理を共有する。

## テスト方針

- バックエンド単体テスト: WebSocket config の `audio_source` が受け取られ、`ready` 応答に反映されることを確認する。
- E2E テスト: 実デバイスは使わず、`getUserMedia` / `getDisplayMedia` / `MediaRecorder` / `WebSocket` をモックして UI 選択と API 呼び分けを検証する。
- 実デバイスや OS の共有ピッカー自体は自動テストの対象外とする。

## 制限

- リモート HTTP ではブラウザがキャプチャをブロックする場合があるため、`localhost` または HTTPS が必要。
- 端末全体・ウィンドウ・アプリ単位の音声共有可否はブラウザ、OS、選択対象に依存する。
- ブラウザのセキュリティ仕様上、アプリ側が共有対象を自動選択することはできない。ユーザー操作によるピッカー選択が必須。
