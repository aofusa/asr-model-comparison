# Frontend (Qwik + Vite)

ASR Model Comparison Platform のフロントエンドです。Qwik + Vite を使用して構築されています。

## 技術スタック

- Qwik (v1.9+)
- Vite
- TypeScript
- Playwright (E2Eテスト)

> **注意**: 本プロジェクトは **Qwik City ではなく、純粋な Qwik + Vite** の構成を採用しています。
> （開発サーバーの安定性と Windows 環境での動作性を優先したため）

## ディレクトリ構成

```
frontend/
├── src/
│   ├── components/          # Qwik コンポーネント
│   ├── routes/              # ルート（現在はほぼ単一ページ）
│   │   └── index.tsx        # メインのリアルタイム UI
│   ├── entry.dev.tsx
│   ├── entry.ssr.tsx
│   └── global.css
├── public/                  # 静的アセット
├── tests/                   # Playwright E2E テスト
├── vite.config.ts
├── tsconfig.json
├── package.json
└── playwright.config.ts
```

## セットアップ

### 1. 依存関係のインストール

```bash
cd frontend
npm install
```

## 開発サーバーの起動

### 単独でフロントエンドを開発する場合

```bash
npm run dev
```

→ `http://localhost:5173` で開発サーバーが起動します。

> **注意**: この状態ではバックエンド（8000番ポート）と連携しないため、一部の機能（モデル一覧取得、WebSocketなど）は動作しません。

### 推奨：プロジェクト全体で起動する場合

プロジェクトルート (`asr-model-comparison/`) から以下のコマンドを実行してください：

```bash
# Windows
.\run.ps1

# macOS / Linux
./run.sh
```

これによりバックエンドと連携した状態で開発できます（ホットリロードは Vite 側で有効）。

## 本番ビルド

### 単独でビルドする場合

```bash
npm run build
```

ビルド成果物は `dist/` に出力されます。

### 推奨：プロジェクト全体のビルド

プロジェクトルートから以下を実行してください：

```bash
# Windows
.\run.ps1 -BuildOnly

# macOS / Linux
./run.sh --build-only
```

これにより、ビルド成果物が自動的に `backend/static/` にコピーされ、シングルアプリモードで利用可能になります。

## テスト

### E2Eテスト (Playwright)

**開発用（Vite開発サーバーに対して）**:
```bash
npm run test:e2e
```

**本番ビルド検証用（ビルド後のフロントエンド + バックエンドに対して）**:
```bash
npm run test:e2e:prod
```

本番用テストを実行する前に、シングルアプリモードでサーバーを起動しておく必要があります：
```bash
# プロジェクトルートで実行
./run.sh          # macOS/Linux
.\run.ps1         # Windows
```

本番用設定ファイル `playwright.config.prod.ts` では `webServer` を無効化しており、事前に `http://localhost:8000` でアプリケーションが起動していることを前提としています。

#### 本番用E2Eの構成（2026-06時点）

- `smoke.prod.spec.ts`：静的ビルド成果物 + 基本UIの配信確認（高速）
- `ws-protocol.prod.spec.ts`：**軽量実通信プロトコル検証**（推奨モデル: `whisper-tiny`）

本番E2Eは「ビルドが正しく配信されているか」と「実際にバックエンドとWebSocket通信ができるか」の2軸で検証します。

重いモデルを使った詳細なUI/UXテストは通常の開発用E2E（`npm run test:e2e`）で実施してください。

## 特徴・設計方針

- **シングルページアプリケーション**として設計
- 重い処理はバックエンド（WebSocket経由）に委譲
- 再接続処理や文脈維持（`previous_text`）に強い UX を提供
- 開発サーバーは Vite を直接使用（Qwik City は使用せず）

## 関連ドキュメント

- プロジェクト全体の README: `../README.md`
- Playwright テスト: `tests/real-time.spec.ts`
