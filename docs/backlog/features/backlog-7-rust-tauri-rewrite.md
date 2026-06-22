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
| Qwen3-ASR | CandleベースのRust-native実装 | Windows/Linux: CUDA -> CPU、macOS: Metal -> CPU |
| Voxtral | Voxtral Realtime対応patched llama.cpp + GGUF/mmproj | Windows/Linux: Vulkan/CPU、macOS: CPU優先（追加検証待ち） |

API契約・進捗・フォールバックを検証可能な軽量プレースホルダーを維持しつつ、各実モデルは`whisper`、`qwen`、`voxtral` featureで差し替える。通常の配布用 `AMCP.exe` はfeatureなしビルドではplaceholder応答になるため、実推論配布にはfeatureを含むビルド構成とネイティブDLL/モデル配置検証が必要。

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
- `/api/status`とWS応答でランタイムバックエンド状態を返し、Qwen3-ASR Candle / Voxtral patched llama.cpp のfeature境界、artifact診断、model-native翻訳診断、モデル準備時の`validating`進捗を追加済み。
- Qwen Candleモデルディレクトリ、Qwen config/tokenizer/weight検証、Voxtral Realtime GGUF/mmproj、patched llama.cppリンク設定、model-native翻訳の事前ロード進捗を追加済み。
- Qwen3-ASRは外部C DLL依存を廃止し、`qwen3-asr`/Candleでモデルロード、自動キャッシュ、実音声サンプル推論呼び出しを行う経路へ移行済み。
- Qwen3-ASR Candleで原文ASRと翻訳先言語指定の2回生成を行うmodel-native翻訳を追加済み。
- Voxtralは旧ORT/ONNX経路を削除し、Voxtral Realtime対応patched llama.cppのdual-stream音声デコードと同一GGUF text decoderによる翻訳へ統一済み。
- 翻訳レスポンス契約、日本語->英語翻訳前の小さい日本語数字正規化、外部Python/コマンドランナー非依存の翻訳境界を追加済み。
- Windows実機で実モデルの速度/品質を測る `validate` CLI と、モデル配置を確認する `--diagnostics-only` を追加済み。
- AMCP.exe serverモードはコンソールログ、`0.0.0.0`待受、LAN向けURL表示、モデル準備/音声チャンク受信ログを追加済み。
- 実モデル統合はfeature境界を定義済み。配布用AMCP.exeへ実推論featureを同梱するビルド構成、各ネイティブ依存のCI整備後に有効化する。

## 未完了・残タスク

- 配布用 `AMCP.exe` の実推論feature同梱ビルド。現状の通常配布ビルドはAPI/音声前処理/ログ確認は可能だが、featureなしでは `runtime_backend=placeholder` になる。
- Qwen/Voxtralのネイティブランナーが独自に出す内部パーセンテージ/ログの取り込み。
- macOS/Linux/Android/iOSでのハードウェアアクセラレータも含めたクロスプラットフォーム対応。
- Android/iOS実機でのマイク/共有音声取得、モデル配置、バイナリサイズ、権限設定の検証が必要。
