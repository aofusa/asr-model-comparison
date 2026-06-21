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

## ASRバックエンド設計

単一モデルロード制約を維持するため、Rust側に`HybridModelManager`を置く。モデル切り替え時は前モデルをアンロードする進捗イベントを出し、常に1モデルだけをロード済み状態として扱う。

モデル別の実装境界は以下とする。

| モデル | Rust統合方針 | 優先アクセラレータ |
| --- | --- | --- |
| Whisper | `whisper-rs` / whisper.cpp | Windows/Linux: CUDA -> Vulkan -> CPU、macOS: Metal -> Vulkan -> CPU |
| Qwen3-ASR | antirez/qwen-asr C FFI | macOS: Accelerate/BLAS、Linux: OpenBLAS、Windows: BLAS -> CPU |
| Voxtral | `ort` + ONNX Runtime | Windows/Linux: CUDA -> CPU、macOS: CoreML -> CPU |

初期実装ではAPI契約・進捗・フォールバックを検証可能な軽量プレースホルダーを置き、各実モデルは`whisper`、`qwen`、`voxtral` featureで差し替える。

## アクセラレータ選択

UI/CLIから`auto`、`gpu`、`cpu`を選択できる。

- `auto`: デフォルト。モデルとOSに応じたGPU/専用EPを積極的に選び、失敗時はCPUへフォールバックする。
- `gpu`: GPUを明示的に優先する。ただし利用可能なGPUバックエンドがない場合はCPUへ安全にフォールバックする。
- `cpu`: GPU/専用EPを試さずCPUを使う。

選択結果はWebSocket/HTTPレスポンスの`accelerator`に含め、UIやログで確認できるようにする。

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
- 実モデル統合はfeature境界を定義済み。各ネイティブ依存のCI整備後に有効化する。
