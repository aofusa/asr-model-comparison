<#
.SYNOPSIS
    ASR Model Comparison Platform - Single App Startup Script (PowerShell)

.DESCRIPTION
    このスクリプトは常に「シングルアプリモード」で起動します。
    1. フロントエンドを本番ビルド
    2. ビルド成果物を backend\static に配置
    3. バックエンド1プロセスだけで API + フロントエンド全体を配信

    セキュリティのため、デフォルトのホストは 127.0.0.1（localhost のみ）です。
    外部からアクセスしたい場合は明示的に -Host 0.0.0.0 （または -Address 0.0.0.0） を指定してください。

    ログは標準出力（コンソール）に出力されます。
    ファイルに保存したい場合はリダイレクトしてください（例: .\run.ps1 > logs\app.log 2>&1）。

.EXAMPLE
    .\run.ps1
    .\run.ps1 -Host 0.0.0.0 -Port 9000
    .\run.ps1 -BuildOnly
    .\run.ps1 -Address 0.0.0.0 -Port 8000 -BuildOnly
#>

$ErrorActionPreference = "Stop"

# Manual argument parsing to support both --host and -Host style from any shell
$Address = "127.0.0.1"
$Port = 8000
$BuildOnly = $false

for ($i = 0; $i -lt $args.Count; $i++) {
    switch ($args[$i]) {
        "--host"       { $Address = $args[++$i]; break }
        "-Host"        { $Address = $args[++$i]; break }
        "--port"       { $Port = [int]$args[++$i]; break }
        "-Port"        { $Port = [int]$args[++$i]; break }
        "--build-only" { $BuildOnly = $true; break }
        "-BuildOnly"   { $BuildOnly = $true; break }
        "-h"           { 
            Write-Host "Usage: .\run.ps1 [--host HOST] [--port PORT] [--build-only]"
            Write-Host ""
            Write-Host "Logs are output to standard output (console) by default."
            Write-Host "To save logs: .\run.ps1 > logs\app.log 2>&1"
            exit 0 
        }
        default {
            Write-Warning "Unknown argument: $($args[$i])"
        }
    }
}

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir = Join-Path $ProjectRoot "backend"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  ASR Model Comparison Platform (AMCP) - Single App Mode" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Frontend build logic (inline)
$FrontendDir = Join-Path $ProjectRoot "frontend"
$BackendStaticDir = Join-Path $ProjectRoot "backend\static"

Write-Host ">>> Building frontend (Qwik City static build)..." -ForegroundColor Yellow

Push-Location $FrontendDir

try {
    if (-not (Test-Path "node_modules")) {
        Write-Host "    - Installing dependencies first..." -ForegroundColor Cyan
        npm install
    }

    # Use the project build script (Qwik City + qwik build).
    # This ensures proper static output including index.html for backend serving.
    npm run build
} finally {
    Pop-Location
}

Write-Host ">>> Copying build output to backend\static\ ..." -ForegroundColor Yellow

if (Test-Path $BackendStaticDir) {
    Remove-Item -Recurse -Force $BackendStaticDir
}
New-Item -ItemType Directory -Path $BackendStaticDir -Force | Out-Null

Copy-Item -Path (Join-Path $FrontendDir "dist\*") -Destination $BackendStaticDir -Recurse -Force

Write-Host "Build complete!" -ForegroundColor Green

if ($BuildOnly) {
    Write-Host "Build-only mode. Exiting without starting server." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host ">>> Starting backend (serving both API and Frontend)..." -ForegroundColor Yellow
Set-Location $BackendDir

# 仮想環境の検出と有効化
$venvActivate = $null
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    $venvActivate = ".\.venv\Scripts\Activate.ps1"
    Write-Host "    - Using virtual environment (.venv)" -ForegroundColor Green
} elseif (Test-Path ".\venv\Scripts\Activate.ps1") {
    $venvActivate = ".\venv\Scripts\Activate.ps1"
    Write-Host "    - Using virtual environment (venv)" -ForegroundColor Green
} else {
    Write-Host "    - No virtual environment detected (using system Python)" -ForegroundColor Yellow
}

if ($venvActivate) {
    . $venvActivate
}

Write-Host ""
Write-Host "Starting server on http://$Address`:$Port" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop." -ForegroundColor Gray
Write-Host ""

# サーバー起動（本番向けなので --reload は付けない）
& python -m uvicorn app.main:app --host $Address --port $Port

if ($LASTEXITCODE -ne 0) {
    Write-Error "Server exited with error code $LASTEXITCODE"
    exit $LASTEXITCODE
}
