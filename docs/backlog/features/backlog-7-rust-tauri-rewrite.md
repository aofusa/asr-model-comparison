# ID 7: Rust/Tauri移植 仕様・設計

## 目的

Python/FastAPIバックエンドをRustへ置き換え、既存Qwikフロントエンドを可能な限り維持したまま、TauriデスクトップアプリとHTTP serverモードの両方で動作させる。

## 対象範囲

- リポジトリルートの`app/`配下にRust/Tauriアプリを作成する。
- 既存フロントエンドは`frontend/`のVite dev/build成果物をTauriから利用する。
- Rustサーバは既存UIが使う以下の契約を維持する。
  - `GET /health`
  - `GET /api/models`
  - `GET /api/status`
  - `POST /api/transcribe`
  - `GET /api/ws/transcribe`
- serverモードでは`--host`、`--port`、`--accelerator auto|gpu|cpu`を指定できる。
- デスクトップモードでも同じASRマネージャとアクセラレータ選択ロジックを使う。
- Tauri Android/iOSビルド導線を用意し、モバイルでも同等UIとアクセラレータ選択を使う。

## ASRバックエンド設計

単一モデルロード制約を維持するため、Rust側に`HybridModelManager`を置く。モデル切り替え時は前モデルをアンロードする進捗イベントを出し、常に1モデルだけをロード済み状態として扱う。

モデル別の実装境界は以下とする。

| モデル | Rust統合方針 | 優先アクセラレータ |
| --- | --- | --- |
| Whisper | `whisper-rs` / whisper.cpp | Windows/Linux: CUDA -> Vulkan -> CPU、macOS: Metal -> Vulkan -> CPU |
| Qwen3-ASR | C FFI / GPU kernel / ONNX / WGPU など最速実装を選択 | Windows: CUDA -> DirectML -> Vulkan -> WGPU -> OpenVINO -> BLAS -> CPU、Linux: CUDA -> Vulkan -> WGPU -> OpenVINO -> BLAS -> CPU、macOS: Metal -> CoreML -> WGPU -> BLAS -> CPU |
| Voxtral | `ort` + ONNX Runtime | Windows/Linux: CUDA -> CPU、macOS: CoreML -> CPU |

初期実装ではAPI契約・進捗・フォールバックを検証可能な軽量プレースホルダーを置き、各実モデルは`whisper`、`qwen`、`voxtral` featureで差し替える。

モバイルでは以下を優先する。

| プラットフォーム | Whisper | Qwen3-ASR | Voxtral |
| --- | --- | --- | --- |
| iOS | Metal -> CPU | Metal -> CoreML -> Accelerate/BLAS -> CPU | CoreML -> CPU |
| Android | Vulkan -> CPU | NNAPI -> Vulkan -> WGPU -> BLAS -> CPU | NNAPI -> Vulkan -> CPU |

## アクセラレータ選択

UI/CLIから`auto`、`gpu`、`cpu`を選択できる。

- `auto`: デフォルト。モデルとOSに応じたGPU/専用EPを積極的に選び、失敗時はCPUへフォールバックする。
- `gpu`: GPUを明示的に優先する。ただし利用可能なGPUバックエンドがない場合はCPUへ安全にフォールバックする。
- `cpu`: GPU/専用EPを試さずCPUを使う。

選択結果はWebSocket/HTTPレスポンスの`accelerator`に含め、UIやログで確認できるようにする。

Tauriデスクトップ/モバイルでは、WebViewから同一アプリ内Rust APIへ接続できるよう、埋め込みHTTP/WebSocketサーバーを`127.0.0.1:8765`で起動する。ブラウザ単体開発では従来通り現在のoriginへ接続する。

## テスト方針

- Rust単体テスト
  - モデル一覧の互換性
  - 単一モデルロード制約
  - OS/モデル別アクセラレータ選択とCPUフォールバック
  - `/api/models`、`/api/status`のAPI形状
- Playwright E2E
  - 既存UIがロードできること
  - CPU/GPU設定が表示され、WebSocket configへ送られること
  - Rust serverモードに対する軽量WebSocketプロトコル検証

## 実装状況

- `app/src-tauri`にRustクレートとTauri設定を追加済み。
- serverモードCLI、API互換ルート、WebSocket契約、アクセラレータ選択、単体テストを追加済み。
- Tauri環境向けの埋め込みRust APIサーバーと、UIのTauri API接続先切替を追加済み。
- Android/iOS用Tauri CLIスクリプトと、iOS/Android向けアクセラレータ優先順を追加済み。
- PCM WAVデコード、`symphonia`による多形式音声デコード、ffmpegフォールバック、16kHzモノラル化、RMS/peak計算、無音判定をRustで追加済み。
- HTTP/WS応答に音声解析結果と`had_speech`を反映済み。
- Qwen3-ASR向けにCUDA/DirectML/Metal/CoreML/Vulkan/WGPU/OpenVINO/NNAPI/BLASの優先戦略を追加済み。
- Windows優先の実機アクセラレータ検出と`AMCP_AVAILABLE_BACKENDS`明示指定を追加済み。
- `whisper-rs` feature有効時に、環境変数で指定したwhisper.cpp互換モデルを使うWhisper実推論を追加済み。
- Whisperモデルのローカルキャッシュ確認、自動ダウンロード、`bytes_downloaded` / `total_bytes` 付きバイト単位進捗を追加済み。
- `/api/status`とWS応答でランタイムバックエンド状態を返し、Qwen3-ASR C FFI / Voxtral ONNX のfeature境界、artifact診断、翻訳ランナー診断、モデル準備時の`validating`進捗を追加済み。
- Qwen3-ASR C FFIの動的ライブラリロード、`qwen_load`/`qwen_transcribe_audio`/`qwen_free`シンボル検証、モデルディレクトリ検証、実音声サンプル推論呼び出しを追加済み。
- Voxtral ONNX Runtimeセッション初期化、分割ONNX構成 (`audio_encoder.onnx`、`embed_tokens.onnx`、`decoder_model_merged.onnx`、`tokenizer.json`) の設定解決、log-mel入力生成、KV cache付き自己回帰デコード、DirectML/CUDA featureコンパイルを追加済み。
- 翻訳レスポンス契約、日本語->英語翻訳前の小さい日本語数字正規化、翻訳コマンドランナー境界、Python版と同じ `Helsinki-NLP/opus-mt-ja-en` を呼ぶ `app/scripts/translate_hf.py` を追加済み。
- Windows実機で実モデルの速度/品質を測る `validate` CLI を追加済み。
- 実モデル統合はfeature境界を定義済み。各ネイティブ依存のCI整備後に有効化する。

## 未完了・残タスク

- Qwen/Voxtral/翻訳モデルの外部ネイティブランナー内部ロードに対する詳細進捗、Voxtral ONNX/翻訳モデルファイルを配置したWindows実機での実測は未完了。
- Android/iOS実機でのマイク/共有音声取得、モデル配置、バイナリサイズ、権限設定の検証が必要。
