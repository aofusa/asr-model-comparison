# ASR Model Comparison Project ワークスペース

このディレクトリ（`asr-model-comparison-project`）は、**ローカルワークスペースのルート**として機能します。

ここには以下のものが含まれます：

- コード以外の成果物（仕様書、設計ドキュメント、AIエージェント向け指示書、実行ログなど）
- 実際のオープンソースプロジェクト（`asr-model-comparison/` サブディレクトリ内）

## ディレクトリ構成

```
asr-model-comparison-project/          ← ワークスペースルート（そのまま公開はしません）
├── README.md                          ← 本ファイル（リポジトリ構成の説明）
├── docs/                              ← 仕様書、詳細ガイド、バックログ
│   ├── backlog/                       ← 管理対象のバックログ
│   ├── specs.md                       ← 管理対象の仕様書
│   ├── real-time-japanese-usage.md    ← 管理対象の利用ガイド
│   └── artifact/                      ← AI生成物などのローカル退避先（Git管理外）
├── logs/                              ← （任意）手動でログを保存したい場合に使用可能（スクリプトは標準出力に出力）
├── AGENTS.md                          ← AIコーディングエージェント向け指示書
└── asr-model-comparison/              ← 実際のGitリポジトリルート
    ├── backend/
    │   └── tests/
    │       └── audio_samples/         # テスト音声（slow/実モデル検証用）
    ├── frontend/
    ├── run.* ( --build-only 対応 )
    ├── README.md                      ← アプリケーション向けドキュメント
    ├── LICENSE                        ← Apache 2.0
    └── ...
```

## 公開されるリポジトリ

`asr-model-comparison/` ディレクトリが、**Gitで管理されGitHubなどに公開されるプロジェクト本体**です。

ソースコード、ビルドスクリプト、ユーザー向けドキュメントはすべて `asr-model-comparison/` 内に配置されています。
セットアップと実行方法は `asr-model-comparison/README.md` に集約しています。

## ドキュメント方針

- 公開ユーザー向けの起動・インストール手順は `asr-model-comparison/README.md` にまとめます。
- バックエンド / フロントエンド固有の補足は、それぞれ `asr-model-comparison/backend/README.md` と `asr-model-comparison/frontend/README.md` に置きます。
- 仕様、バックログ、検討中の設計メモはワークスペース側の `docs/` に置きます。
- AI生成の修正指示書、調査メモ、実行ログなどは `docs/artifact/` に退避し、Git管理には含めません。

## 開発・コントリビュートについて

アプリケーション本体を開発する場合は、`asr-model-comparison/` ディレクトリ内で作業してください。

大規模な仕様書や設計ドキュメントは、本ワークスペースルートの `docs/` ディレクトリに配置してください。

## ライセンス

このワークスペースと `asr-model-comparison/` 内のコードは Apache License 2.0 の下で提供されています。  
詳細は `LICENSE` および `asr-model-comparison/LICENSE` を参照してください。

---

**著者**: Copyright 2026 aofusa