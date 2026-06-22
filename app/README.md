# AMCP Rust/Tauri App

`app/` は ASR Model Comparison Platform の Rust + Tauri 版アプリケーションです。Python/FastAPIバックエンドをRustへ置き換え、既存のQwikフロントエンドをできるだけ維持したまま、デスクトップアプリとHTTP serverモードの両方で動作させることを目的としています。

## 現在の状態

- `src-tauri/` にRustバックエンドとTauri設定を配置しています。
- 既存UIはリポジトリルートの `frontend/` を利用します。
- Rust serverモードは既存UIが使う主要API契約を提供します。
  - `GET /health`
  - `GET /api/models`
  - `GET /api/status`
  - `POST /api/transcribe`
  - `GET /api/ws/transcribe`
- `auto` / `gpu` / `cpu` のアクセラレータ設定に対応しています。
- Tauriデスクトップ/モバイル環境では、UIから同一アプリ内のRust APIサーバー (`http://127.0.0.1:8765`) へ接続します。
- Whisper、Qwen3-ASR Candle、Voxtral Realtime patched llama.cppはfeature有効時に実推論へ接続します。翻訳は外部Python/コマンドランナーを使わず、Qwen3-ASR Candleの出力言語プロンプト、またはVoxtral patched GGUFのtext decoderによるmodel-native翻訳を優先します。

## Python版との機能網羅状況

現時点のRust/Tauri版は、Python製バックエンドと既存フロントエンドの全機能を完全には網羅していません。

実装済み:

- 既存UIの主要WebSocket config契約
- モデル一覧API
- ステータスAPI
- HTTP/WS文字起こしAPIのレスポンス形状
- PCM WAVデコード、16kHzモノラル化、RMS/peak計算
- `symphonia`によるWAV/MP3/OGG/Vorbis/MP4/AAC/FLAC等の多形式デコード
- WebM/Opus等の未対応形式に対する`ffmpeg`フォールバック
- 無音判定と`had_speech=false`応答
- `whisper-rs` feature有効時のWhisper実推論
- Whisperモデルのローカルキャッシュ確認と自動ダウンロード
- 翻訳レスポンス契約 (`transcript_text` / `translated_text` / `target_language`) の維持
- 日本語 -> 英語翻訳前の小さい日本語数字正規化
- Qwen3-ASR Candleの出力言語プロンプト、およびVoxtral Realtime patched llama.cppによるASR/翻訳
- 単一モデルロード制約の管理
- モデル準備進捗イベント (`loading` / `validating` / `downloading` / `downloaded` / `ready`)
- Whisper自動ダウンロード時の `bytes_downloaded` / `total_bytes` 付きバイト単位進捗
- Windows実機で実モデルの速度/品質を測る `validate` CLI
- `auto` / `gpu` / `cpu` の選択と安全なCPUフォールバック
- Windows優先の実機アクセラレータ検出 (`CUDA_PATH`/`nvidia-smi`、`VK_SDK`/`vulkaninfo`、DirectML、OpenVINO、WGPU等)
- Qwen3-ASR向けのCandle CUDA/Metal/CPU優先戦略（DirectML/WGPUはCandleのQwen実行経路が対応するまでCPUフォールバック）
- `/api/status`とWS応答でのランタイムバックエンド状態 (`whisper-rs` / `qwen-native` / `voxtral-llamacpp` / `placeholder`) と、モデルファイル/ディレクトリ/model-native翻訳のartifact診断
- Qwen3-ASR Candle / Voxtral patched llama.cpp のfeature境界と設定検証 (`AMCP_QWEN_*`、`AMCP_VOXTRAL_*`)
- Qwen3-ASR Candleのモデルロード、自動キャッシュ、モデルディレクトリ検証、実音声サンプル推論呼び出し
- Voxtral Realtime GGUF/mmproj構成の設定解決、dual-stream音声デコード、同一GGUF text decoderによる翻訳
- Tauriデスクトップ/モバイルWebViewからRust APIへ接続するための埋め込みサーバー
- Android/iOS向けTauriビルドスクリプト

未完了:

- Qwen/Voxtralのネイティブランナー内部ロードに対する詳細進捗
- Qwen/Voxtral実モデルファイルを配置したWindows実機での出力品質・速度検証
- Android/iOS実機でのマイク・画面音声取得制約の検証
- Android/iOS向けの実モデルバイナリ、モデル配置、アプリサイズ最適化

## 必要な環境

- Rust 1.95+ / Cargo
- Node.js 24+ / npm
- Playwrightで利用するブラウザ
- Tauriデスクトップビルドを行う場合は、各OS向けのTauri前提環境
- WebM/Opusなど`symphonia`未対応形式をリアルタイム入力する場合は`ffmpeg`をPATHに追加

初回は `app/` と `frontend/` の依存関係をインストールしてください。

```powershell
cd app
npm install

cd ..\frontend
npm install
```

## 開発起動

### Tauriデスクトップモード

```powershell
cd app
npm run dev
```

`npm run dev` はTauri開発モードを起動します。Tauri設定では `frontend/` のVite開発サーバーを利用します。

Tauri環境では、アプリ起動時にRust APIサーバーが `127.0.0.1:8765` で起動し、UIのWebSocket通信は自動的にそのサーバーへ向きます。

### Rust serverモード

```powershell
cd app
npm run server:local
```

既定では `http://127.0.0.1:8000` で起動します。host、port、アクセラレータを指定したい場合は直接Cargoコマンドを使います。

```powershell
cargo run --manifest-path src-tauri/Cargo.toml --bin amcp-server -- server --host 0.0.0.0 --port 8000 --accelerator auto
```

アクセラレータ指定:

- `auto`: デフォルト。GPU/専用アクセラレータを優先し、使えない場合はCPUへフォールバックします。
- `gpu`: GPUを明示的に優先します。利用不可の場合は安全にCPUへフォールバックします。
- `cpu`: GPUを試さずCPUのみを使います。

### Whisper実推論

Whisperは `whisper-rs` / whisper.cpp を使った実推論に対応しています。通常ビルドでは依存を有効化せず、`whisper` feature付きで起動した場合のみ実推論を使います。

既定ではwhisper.cpp互換のGGMLモデルを `ggerganov/whisper.cpp` から Hugging Face Hub 標準キャッシュへ取得します。Whisper Rust実装は whisper.cpp 用GGMLモデルを使うため、Python backend の faster-whisper / CTranslate2 モデルとは別物ですが、保存先は `HF_HUB_CACHE` または `HF_HOME\hub` 配下に統一されます。保存先を明示的に変えたい場合は `AMCP_MODEL_DIR` を指定してください。

```powershell
$env:AMCP_MODEL_DIR="C:\models\amcp"
cd app
npm run server:whisper
```

すでにモデルを用意している場合は、環境変数でパスを指定すると自動ダウンロードより優先されます。

```powershell
$env:AMCP_WHISPER_TINY_MODEL_PATH="C:\models\ggml-tiny.bin"
cd app
npm run server:whisper
```

全Whisperモデル共通のパスとして指定する場合:

```powershell
$env:AMCP_WHISPER_MODEL_PATH="C:\models\ggml-tiny.bin"
npm run server:whisper
```

WindowsでGPU優先にしたい場合:

```powershell
# Vulkan
npm run server:whisper:vulkan

# CUDA
npm run server:whisper:cuda
```

対象モデルのダウンロードURLを解決できない場合、アプリは開発用プレースホルダー推論へ安全にフォールバックします。

### Qwen3-ASR Candle実推論

Qwen3-ASRは `qwen` feature付きでCandleベースのRust実装へ接続し、外部C DLLなしで16kHz mono f32サンプルを実推論します。未指定時は Python backend の `from_pretrained()` と同じ Hugging Face Hub 標準キャッシュ (`HF_HUB_CACHE`、または `HF_HOME\hub`) を使い、`config.json`、`model.safetensors`、tokenizerソースを同じ snapshot から参照します。`AMCP_QWEN_MODEL_DIR` を指定した場合はそのディレクトリを使い、`AMCP_MODEL_DIR` を指定した場合は従来通り `AMCP_MODEL_DIR\qwen\Qwen--Qwen3-ASR-0.6B` などを使います。

```powershell
$env:AMCP_QWEN_MODEL_DIR="C:\models\qwen\Qwen--Qwen3-ASR-0.6B"
cd app
npm run server:qwen
```

CUDA環境では `qwen-cuda` featureを使えます。CUDA以外のWindows GPU/DirectML/WGPUはCandle上流のQwen3-ASR backendが未対応のため、現時点ではCPUへフォールバックします。

```powershell
npm run server:qwen:cuda
```

自動ダウンロードを禁止したい場合は `AMCP_QWEN_DISABLE_DOWNLOAD=1` を設定してください。言語指定が `auto` 以外の場合は、Qwen3-ASRの言語プロンプトへ変換します。

### 翻訳実推論

翻訳は外部Python/コマンドランナーを使いません。Qwen3-ASRは `target_language` 指定時にRust/Candle経路で原文ASRと翻訳を同一API処理内で返します。ja/en翻訳では既定で `voiceping-ai/qwen3-asr-ja-en-speech-translation` を翻訳用checkpointとして使い、`AMCP_QWEN_TRANSLATION_MODEL_ID` / `AMCP_QWEN_TRANSLATION_MODEL_DIR` で別checkpointへ差し替えできます。`AMCP_QWEN_TRANSLATION_MODEL_ID` に同じQwen3-ASRモデルを明示した場合は、音声エンコードを1回だけ行い同じ音声埋め込みから原文ASRと翻訳先言語プロンプトの生成を実行します。Voxtralはpatched llama.cpp経路で原文ASRを生成し、`target_language` が指定された場合は同じGGUF text decoderで翻訳プロンプトを追加実行します。レスポンスは `transcript_text` に原文、`translated_text` に翻訳結果、`text` に表示用テキストを保持します。

### Voxtral Realtime patched llama.cpp実推論

Voxtralは、Voxtral Realtime対応パッチ済み `llama.cpp` / `libmtmd` をアプリ本体へC ABIブリッジでリンクする経路を標準で使います。日本語以外の音声認識や、日本語から他言語への翻訳タスクも同じruntimeへルーティングします。

例:

```powershell
$env:AMCP_VOXTRAL_RUNTIME="llamacpp"
$env:AMCP_VOXTRAL_LLAMA_MODEL_PATH="C:\models\voxtral\Voxtral-Mini-4B-Realtime.gguf"
$env:AMCP_VOXTRAL_LLAMA_MMPROJ_PATH="C:\models\voxtral\mmproj-Voxtral-Mini-4B-Realtime.gguf"
cd app
npm run server:voxtral:llamacpp
```

Hugging Faceの共通キャッシュを使う場合は、GGUFとmmprojを通常のHub cacheへ配置し、repo/file名だけを指定できます。明示pathが未設定の場合、`HF_HUB_CACHE`、`HF_HOME`、既定の `%USERPROFILE%\.cache\huggingface\hub` の順に参照します。

```powershell
$env:AMCP_VOXTRAL_RUNTIME="llamacpp"
$env:AMCP_VOXTRAL_LLAMA_REPO_ID="mistralai/Voxtral-Mini-4B-Realtime-2602-GGUF"
$env:AMCP_VOXTRAL_LLAMA_MODEL_FILE="Voxtral-Mini-4B-Realtime.gguf"
$env:AMCP_VOXTRAL_LLAMA_MMPROJ_FILE="mmproj-Voxtral-Mini-4B-Realtime.gguf"
npm run server:voxtral:llamacpp
```

主な環境変数:

- `AMCP_VOXTRAL_RUNTIME=llamacpp`: Voxtralで組み込み `llama.cpp` runnerを明示的に使います。未指定でも `language=ja` または `target_language=ja` のVoxtral処理は `llama.cpp` へ回します。
- `AMCP_VOXTRAL_LLAMA_MODEL_PATH`: Voxtral Realtimeのtext model GGUFです。未設定時はHF共通キャッシュを参照します。
- `AMCP_VOXTRAL_LLAMA_MMPROJ_PATH`: Voxtral Realtimeの音声encoder mmproj GGUFです。未設定時はHF共通キャッシュを参照します。
- `AMCP_VOXTRAL_LLAMA_REPO_ID`: HF共通キャッシュで参照するrepo IDです。既定は `mistralai/Voxtral-Mini-4B-Realtime-2602-GGUF` です。
- `AMCP_VOXTRAL_LLAMA_MODEL_FILE`: repo snapshot内のtext GGUFファイル名です。未設定かつ候補が1つだけの場合は自動検出します。
- `AMCP_VOXTRAL_LLAMA_MMPROJ_FILE`: repo snapshot内のmmproj GGUFファイル名です。未設定かつ候補が1つだけの場合は自動検出します。
- `AMCP_VOXTRAL_LLAMA_GPU_LAYERS`: GPUへoffloadするlayer数です。既定は999で、可能な限りGPUへ寄せます。
- `AMCP_VOXTRAL_LLAMA_CONTEXT_SIZE`: llama.cpp context sizeです。既定は4096です。
- `AMCP_VOXTRAL_LLAMA_MAX_TOKENS`: llama.cpp生成トークン数です。未指定時は `AMCP_VOXTRAL_MAX_TOKENS`、それも未指定なら512です。

`voxtral-llamacpp` featureは診断とルーティング用adapterを有効化します。実推論には通常 `voxtral-llamacpp-native` featureで `llama-cpp-4` の `mtmd` 組み込みを使います。Windows AMD GPU環境では `voxtral-llamacpp-vulkan` featureを使い、Vulkan SDK、runtime、driverを事前に用意してください。GGUFとmmprojの入手先・ファイル名は配布元のVoxtral Realtime GGUF releaseに合わせてください。

Voxtral Mini 4B Realtime GGUFは、標準の `llama-cpp-4` 同梱 `llama.cpp` では `ada_norm` tensorとdual-stream音声処理が不足する場合があります。その場合は `voxtral-llamacpp-realtime-patched` featureを使い、Voxtral Realtime対応パッチ済み `llama.cpp` をアプリ本体へC ABIブリッジでリンクします。この経路は外部runnerプロセスを起動せず、パッチ済み `llama.cpp` ライブラリを同一プロセス内で呼び出します。

パッチ済み経路のビルド時環境変数:

- `AMCP_VOXTRAL_PATCHED_LLAMA_DIR`: パッチ済み `llama.cpp` のsource rootです。`include/llama.h` と `tools/mtmd/mtmd-helper.h` を含むディレクトリを指定します。
- `AMCP_VOXTRAL_PATCHED_LLAMA_LIB_DIR`: CMakeビルド済みのimport libraryまたは共有ライブラリがあるディレクトリです。
- `AMCP_VOXTRAL_PATCHED_LLAMA_BIN_DIR`: WindowsでDLLが `bin` に出る場合の任意指定です。build時のlink search pathに追加します。

例:

```powershell
$env:AMCP_VOXTRAL_RUNTIME="llamacpp"
$env:AMCP_VOXTRAL_PATCHED_LLAMA_DIR="C:\src\llama.cpp-voxtral-realtime"
$env:AMCP_VOXTRAL_PATCHED_LLAMA_LIB_DIR="C:\src\llama.cpp-voxtral-realtime\build"
$env:AMCP_VOXTRAL_PATCHED_LLAMA_BIN_DIR="C:\src\llama.cpp-voxtral-realtime\build\bin"
npm run server:voxtral:llamacpp:realtime
```

Windowsではbuild時に `AMCP_VOXTRAL_PATCHED_LLAMA_BIN_DIR` 配下の `llama.dll`、`mtmd.dll`、`ggml*.dll` をCargoのprofile出力ディレクトリへコピーします。これにより、以前の `llama-cpp-4` ビルドで残った同名DLLを誤ってロードすることを避けます。

日本語サンプルで検証する場合:

```powershell
$env:AMCP_VOXTRAL_RUNTIME="llamacpp"
$env:AMCP_VOXTRAL_LLAMA_MODEL_PATH="C:\models\voxtral\Voxtral-Mini-4B-Realtime.gguf"
$env:AMCP_VOXTRAL_LLAMA_MMPROJ_PATH="C:\models\voxtral\mmproj-Voxtral-Mini-4B-Realtime.gguf"
npm run validate:windows:voxtral:llamacpp -- --audio "..\backend\tests\audio_samples\ja_01.mp3" --model-id voxtral-mini-4b --language ja --json
```

パッチ済み経路で検証する場合:

```powershell
npm run validate:windows:voxtral:llamacpp:realtime -- --audio "..\backend\tests\audio_samples\ja_01.mp3" --model-id voxtral-mini-4b --language ja --json
```

## ビルド方法

### Windows配布用の単一exeを作成

Windows向けに配布する場合は、Tauriのリリースビルドで生成されるデスクトップ実行ファイルを `app/dist/AMCP.exe` にコピーします。

```powershell
cd app
npm run build:windows:exe
```

Tauri CLIの一次出力:

```text
app\src-tauri\target\release\amcp-desktop.exe
```

配布用コピー:

```text
app\dist\AMCP.exe
```

`npm run build:windows:exe` は、Tauri CLIが生成した `amcp-desktop.exe` を `app\dist\AMCP.exe` へコピーして配布名に揃えるためのラッパーです。更新されたか確認する場合は、以下のように更新時刻とハッシュを比較してください。
コード変更がなくTauriの出力バイナリが同一の場合、SHA256は前回と同じになります。このラッパーは配布コピーを実行したことが分かるように、コピー後の `app\dist\AMCP.exe` の更新時刻を実行時刻へ更新します。

```powershell
Get-Item .\dist\AMCP.exe, .\src-tauri\target\release\amcp-desktop.exe | Select-Object FullName, LastWriteTime, Length
Get-FileHash .\dist\AMCP.exe, .\src-tauri\target\release\amcp-desktop.exe -Algorithm SHA256
```

この `AMCP.exe` はフロントエンド資産とRust APIサーバーを同梱した単一の実行ファイルです。実行時はアプリ内でRust APIサーバーを `127.0.0.1:8765` に起動し、Tauri WebViewから接続します。
必ず `npm run build:windows:exe` または `npm run build` のようにTauri CLI経由でビルドしてください。`cargo build --bin amcp-desktop` で直接作ったexeは、Tauriの本番Webアセット埋め込みが行われず、開発用の `localhost:5173` に接続しようとします。

同じ `AMCP.exe` は、明示的に `--server` を付けた場合だけHTTP serverモードで起動します。

```powershell
.\dist\AMCP.exe --server --host 127.0.0.1 --port 8000
```

引数なしでダブルクリックした場合は、通常のTauriデスクトップアプリとして起動します。

配布時の注意:

- Windows 10/11 を対象にしています。
- Microsoft Edge WebView2 Runtime が必要です。通常のWindows 10/11環境には入っていますが、未導入環境ではMicrosoft公式のWebView2 Runtimeを導入してください。
- Whisper / Qwen3-ASR / Voxtral のモデルファイルはexeに同梱しません。必要に応じて `AMCP_MODEL_DIR`、`AMCP_QWEN_MODEL_DIR`、`AMCP_VOXTRAL_MODEL_DIR` などで配置先を指定するか、対応実装の自動ダウンロードを利用します。
- 単一exe配布では、必ず `app\dist\AMCP.exe` を渡してください。インストーラー形式が必要になった場合は、`.ico` アイコンを追加し、Tauriの `bundle.active` と `bundle.targets` を有効化してください。

### Tauriアプリをビルド

```powershell
cd app
npm run build
```

このコマンドはTauriのビルドを実行します。Tauri設定により、ビルド前に `frontend/` の本番ビルドも実行されます。Windows配布用の `app\dist\AMCP.exe` を作る場合は、上記の `npm run build:windows:exe` を使用してください。

### Rust serverのみをビルド

```powershell
cd app
cargo build --manifest-path src-tauri/Cargo.toml --bin amcp-server
```

### Androidビルド

初回のみAndroidプロジェクトを初期化します。

```powershell
cd app
npm run android:init
```

開発実行:

```powershell
npm run android:dev
```

リリースビルド:

```powershell
npm run android:build
```

Androidでは、WhisperはVulkan優先、VoxtralはVulkan -> CPU、Qwen3-ASRはBLAS -> CPUの順で利用する設計です。実機でのアクセラレータ利用可否は端末、OS、ドライバ、ネイティブ依存の組み込み状況に依存します。

### iOSビルド

初回のみiOSプロジェクトを初期化します。

```bash
cd app
npm run ios:init
```

開発実行:

```bash
npm run ios:dev
```

リリースビルド:

```bash
npm run ios:build
```

iOSでは、WhisperはMetal優先、VoxtralはCoreML -> CPU、Qwen3-ASRはAccelerate/BLAS -> CPUの順で利用する設計です。iOSビルドにはmacOS、Xcode、iOS向けTauri前提環境が必要です。

## テスト方法

### すべてのappテスト

```powershell
cd app
npm test
```

`npm test` は以下を順に実行します。

- Rust単体テスト
- Playwright E2Eテスト

### Rust単体テストのみ

```powershell
cd app
npm run test:rust
```

Whisper featureのコンパイル確認:

```powershell
npm run test:whisper:compile
```

Qwen Candle featureのコンパイル確認:

```powershell
npm run test:qwen:compile
npm run test:qwen:cuda:compile
```

Voxtral patched llama.cpp featureのコンパイル確認:

```powershell
npm run test:voxtral:compile
npm run test:voxtral:directml:compile
npm run test:voxtral:cuda:compile
```

主に以下を検証します。

- モデル一覧の互換性
- 単一モデルロード制約
- CPU/GPUアクセラレータ選択とフォールバック
- `/api/models`、`/api/status` のAPI形状
- `/api/status.runtime_backends[].artifacts` による不足モデルファイル診断
- `/api/status.translation` によるmodel-native翻訳設定診断

### Windows実機モデル検証

実モデルファイルを配置したWindows環境では、`validate` CLIで音声ファイル単位の速度と品質を確認できます。
日本語の実音声検証には、リポジトリ内の `..\backend\tests\audio_samples\ja_01.mp3` を使います。

実測前に現在のモデル配置と翻訳設定を確認する場合:

```powershell
cd app
npm run validate:windows -- --diagnostics-only --json
```

Whisper:

```powershell
cd app
npm run validate:windows:whisper -- --audio "..\backend\tests\audio_samples\ja_01.mp3" --model-id whisper-tiny --language ja --expected-text "期待する文字起こし" --json
```

Qwen Candle:

```powershell
$env:AMCP_QWEN_MODEL_DIR="C:\models\qwen\Qwen--Qwen3-ASR-0.6B"
npm run validate:windows:qwen -- --audio "..\backend\tests\audio_samples\ja_01.mp3" --model-id qwen3-asr-0.6b --language ja --expected-text "期待する文字起こし" --json
```

Voxtral:

```powershell
$env:AMCP_VOXTRAL_MODEL_DIR="C:\models\voxtral"
npm run validate:windows:voxtral:directml -- --audio "..\backend\tests\audio_samples\ja_01.mp3" --model-id voxtral-mini-4b --language ja --expected-text "期待する文字起こし" --json
```

出力には `audio_duration_seconds`、`wall_time_seconds`、`realtime_factor`、`runtime_backend`、`accelerator`、`character_error_rate` が含まれます。`realtime_factor` は 1.0 未満なら実時間より高速です。

### Playwright E2Eのみ

```powershell
cd app
npm run test:e2e
```

E2EではPlaywright経由で実ブラウザを使い、Rust serverモードに対して以下を検証します。

- `/health` と `/api/models` の応答
- WebSocket `/api/ws/transcribe` のconfig送信、ready応答、文字起こし応答
- `gpu` 指定時のアクセラレータ設定がレスポンスに反映されること

ブラウザを表示して確認したい場合:

```powershell
npm run test:e2e:headed
```

## アプリケーションの使い方

1. `npm run dev` でTauriデスクトップアプリを起動するか、`npm run server:local` でRust serverモードを起動します。
2. UIで音声入力元を選択します。
   - `Mic`: マイク入力
   - `System`: ブラウザの画面共有経由でシステム音声を取得
   - `Window`: 共有ピッカーで選択したウィンドウやアプリの音声を取得
3. 入力言語と翻訳出力を選択します。
4. `Hardware Acceleration` を選択します。
   - 通常は `Auto (prefer GPU)` を推奨します。
   - 問題がある場合は `CPU only` を選ぶと安定性を優先できます。
5. `Advanced settings` でモデルや生成パラメータを調整します。
6. `Start Recording` を押すとWebSocket接続とモデル準備が始まり、準備完了後に音声チャンクが送信されます。
7. 認識結果はライブ表示と履歴に表示されます。

## 実モデル統合方針

Rust実装はAPI/WS契約、アクセラレータ選択、フォールバック、実モデル検証CLIを維持しつつ、以下のfeature境界で実ASRへ接続します。

- `whisper`: `whisper-rs` / whisper.cpp
- `qwen`: in-tree Qwen3-ASR Candle implementation (`candle-core` / `candle-nn` / `tokenizers` / `safetensors`)
- `voxtral`: patched Voxtral Realtime `llama.cpp` / `libmtmd`

想定する優先アクセラレータ:

- Whisper
  - Windows/Linux: CUDA -> Vulkan -> CPU
  - macOS: Metal -> Vulkan -> CPU
  - iOS: Metal -> CPU
  - Android: Vulkan -> CPU
- Qwen3-ASR
  - Windows/Linux: CUDA -> CPU (`qwen-cuda` feature時)
  - macOS: Metal -> CPU (Candle upstream対応時)
  - iOS/Android: CPU fallback
- Voxtral
  - Windows: DirectML -> CUDA -> CPU
  - Linux: CUDA -> CPU
  - macOS: CoreML -> CPU
  - iOS: CoreML -> CPU
  - Android: NNAPI -> Vulkan -> CPU

## 関連ドキュメント

- プロジェクト全体: `../README.md`
- ID 7仕様設計: `../docs/backlog/features/backlog-7-rust-tauri-rewrite.md`
- 既存フロントエンド: `../frontend/README.md`
