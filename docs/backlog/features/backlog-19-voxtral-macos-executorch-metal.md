# ID 19: Voxtral macOS ExecuTorch Metal 実行経路

## 目的

macOS Apple Silicon上でVoxtral Realtimeをハードウェアアクセラレータ付きで実行するため、Windows向けpatched llama.cpp経路とは別に、ExecuTorch Metal runnerを利用する実行経路を追加する。

## 背景

既存のVoxtral実装はpatched `llama.cpp` / `libmtmd` をC ABIで同一プロセスへリンクする。Windows Vulkan配布ではこの方式が扱いやすい一方、macOSではVoxtral Realtime向けにExecuTorch Metalの`.pte`モデルとrunnerが公開されており、Apple GPUで低遅延実行しやすい。

## 設計

- `AMCP_VOXTRAL_RUNTIME=executorch` でExecuTorch Metal runnerを明示する。
- macOSでExecuTorch runnerと必要ファイルが揃っている場合は、VoxtralのMetal providerとして診断に表示する。
- 入力音声は既存Rust前処理で16kHz monoに正規化し、一時WAVへ書き出してrunnerの `--audio_path` に渡す。
- 必要な環境変数:
  - `AMCP_VOXTRAL_EXECUTORCH_RUNNER_PATH`
  - `AMCP_VOXTRAL_EXECUTORCH_MODEL_PATH`
  - `AMCP_VOXTRAL_EXECUTORCH_PREPROCESSOR_PATH`
  - `AMCP_VOXTRAL_EXECUTORCH_TOKENIZER_PATH`
  - `AMCP_VOXTRAL_EXECUTORCH_STREAMING=1`（streaming exportを使う場合）
  - `AMCP_VOXTRAL_EXECUTORCH_DYLD_LIBRARY_PATH`（runnerに必要なdylib検索パス）
- 既存のpatched llama.cpp経路は維持し、WindowsのVulkan/CUDA系featureは壊さない。

## テスト方針

- Rust単体テストでExecuTorch設定検出、artifact診断、出力テキスト抽出、macOS provider優先順位を確認する。
- ExecuTorch runnerが未配置の環境では、実行検証はdiagnostics-onlyまで行う。
- runnerとMetal `.pte` 一式がある環境では `ja_01.mp3` で `validate:macos:voxtral:executorch` を実行する。

## 完了条件

- `cargo test` が通る。
- `cargo check --features voxtral-executorch` が通る。
- `/api/status` でExecuTorch Metal runnerのartifact状態が確認できる。
- README類にmacOS Metal runnerのビルド・モデル配置・検証手順が記載される。
