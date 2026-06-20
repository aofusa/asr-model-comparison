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

    --reload を指定すると開発モードになります:
    - uvicorn --reload で Python ファイル変更時に自動サーバ再起動
    - フロントエンドソース変更を検知して自動でビルド + static 更新

    ログは標準出力（コンソール）に出力されます。
    ファイルに保存したい場合はリダイレクトしてください（例: .\run.ps1 > logs\app.log 2>&1）。

.EXAMPLE
    .\run.ps1
    .\run.ps1 -Host 0.0.0.0 -Port 9000
    .\run.ps1 -BuildOnly
    .\run.ps1 -Address 0.0.0.0 -Port 8000 -BuildOnly
    .\run.ps1 --reload
    .\run.ps1 --reload --host 0.0.0.0
#>

$ErrorActionPreference = "Stop"

# Manual argument parsing to support both --host and -Host style from any shell
$Address = "127.0.0.1"
$Port = 8000
$BuildOnly = $false
$Reload = $false

for ($i = 0; $i -lt $args.Count; $i++) {
    switch ($args[$i]) {
        "--host"       { $Address = $args[++$i]; break }
        "-Host"        { $Address = $args[++$i]; break }
        "--port"       { $Port = [int]$args[++$i]; break }
        "-Port"        { $Port = [int]$args[++$i]; break }
        "--build-only" { $BuildOnly = $true; break }
        "-BuildOnly"   { $BuildOnly = $true; break }
        "--reload"     { $Reload = $true; break }
        "-Reload"      { $Reload = $true; break }
        "-r"           { $Reload = $true; break }
        "-h"           { 
            Write-Host "Usage: .\run.ps1 [--host HOST] [--port PORT] [--build-only] [--reload]"
            Write-Host ""
            Write-Host "  --reload     Development mode: uvicorn --reload + auto frontend rebuild on source changes"
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

function Test-PortAvailable {
    param(
        [int]$Port
    )

    $listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    if ($listeners.Count -eq 0) {
        return
    }

    Write-Host "ERROR: Port $Port is already in use. Stop the existing server before starting AMCP." -ForegroundColor Red
    foreach ($listener in $listeners) {
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)" -ErrorAction SilentlyContinue
        $address = "$($listener.LocalAddress):$($listener.LocalPort)"
        if ($proc) {
            Write-Host "  - $address pid=$($listener.OwningProcess)" -ForegroundColor Yellow
            Write-Host "    $($proc.CommandLine)" -ForegroundColor DarkYellow
        } else {
            Write-Host "  - $address pid=$($listener.OwningProcess)" -ForegroundColor Yellow
        }
    }
    Write-Host ""
    Write-Host "Example cleanup command:" -ForegroundColor Cyan
    Write-Host "  Get-NetTCPConnection -LocalPort $Port -State Listen | ForEach-Object { Stop-Process -Id `$_.OwningProcess -Force }" -ForegroundColor Cyan
    exit 1
}

if (-not $BuildOnly) {
    Test-PortAvailable -Port $Port
}

# FrontendDir / StaticDir used by build and watcher
$FrontendDir = Join-Path $ProjectRoot "frontend"
$BackendStaticDir = Join-Path $ProjectRoot "backend\static"

function Invoke-FrontendBuild {
    param(
        [string]$ProjectRoot
    )
    $feDir = Join-Path $ProjectRoot "frontend"
    $staticDir = Join-Path $ProjectRoot "backend\static"

    Write-Host ">>> Building frontend (Qwik City static build)..." -ForegroundColor Yellow

    Push-Location $feDir

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

    if (Test-Path $staticDir) {
        Remove-Item -Recurse -Force $staticDir
    }
    New-Item -ItemType Directory -Path $staticDir -Force | Out-Null

    Copy-Item -Path (Join-Path $feDir "dist\*") -Destination $staticDir -Recurse -Force

    Write-Host "Build complete!" -ForegroundColor Green
}

# Always build initially (required for single-app mode)
Invoke-FrontendBuild -ProjectRoot $ProjectRoot

if ($BuildOnly) {
    Write-Host "Build-only mode. Exiting without starting server." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host ">>> Starting backend (serving both API and Frontend)..." -ForegroundColor Yellow

# Use Push-Location + try/finally + Pop-Location so that after the server
# process exits (including when user presses Ctrl+C), the caller's original
# working directory is restored. Previously Set-Location would leave the
# PowerShell session in the backend/ directory.
Push-Location $BackendDir

$serverJob = $null
$watcherJob = $null

try {
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

    $uvicornArgs = @("app.main:app", "--host", $Address, "--port", $Port)
    if ($Reload) {
        $uvicornArgs += "--reload"
    }

    Write-Host ""
    Write-Host "Starting server on http://$Address`:$Port" -ForegroundColor Green
    if ($Reload) {
        Write-Host "RELOAD MODE: --reload enabled for Python changes + frontend watcher for auto rebuild." -ForegroundColor Cyan
    }
    Write-Host "Press Ctrl+C to stop." -ForegroundColor Gray
    Write-Host ""

    if ($Reload) {
        # === RELOAD MODE: run uvicorn and watcher as jobs, forward output in loop ===
        Write-Host "Starting uvicorn (in background job with --reload)..." -ForegroundColor Yellow

        $serverJob = Start-Job -ScriptBlock {
            param($backDir, $vArgs, $venvPath)
            Set-Location $backDir
            if ($venvPath -and (Test-Path $venvPath)) {
                . $venvPath
            }
            & python -m uvicorn @vArgs 2>&1
        } -ArgumentList $BackendDir, $uvicornArgs, $venvActivate

        Write-Host "Starting frontend watcher job (polls src/ and public/ for changes)..." -ForegroundColor Yellow

        $watcherJob = Start-Job -ScriptBlock {
            param($feDir, $statDir)
            Write-Output "[Watcher] Monitoring frontend sources for changes (poll every ~1.5s)..."

            $lastState = @{}
            $dirsToWatch = @("src", "public")

            function Get-CurrentState {
                param($base, $subdirs)
                $state = @{}
                foreach ($sd in $subdirs) {
                    $p = Join-Path $base $sd
                    if (Test-Path $p) {
                        Get-ChildItem -Path $p -Recurse -File -ErrorAction SilentlyContinue | ForEach-Object {
                            $state[$_.FullName] = $_.LastWriteTimeUtc.ToString("o") + "|" + $_.Length
                        }
                    }
                }
                return $state
            }

            # Prime the state (after initial build)
            $lastState = Get-CurrentState -base $feDir -subdirs $dirsToWatch
            Start-Sleep -Seconds 1

            while ($true) {
                Start-Sleep -Seconds 1.5
                $current = Get-CurrentState -base $feDir -subdirs $dirsToWatch

                $changed = $false
                foreach ($k in $current.Keys) {
                    if (-not $lastState.ContainsKey($k) -or $lastState[$k] -ne $current[$k]) {
                        $changed = $true
                        break
                    }
                }
                if (-not $changed -and ($lastState.Count -ne $current.Count)) {
                    $changed = $true
                }

                if ($changed -and $lastState.Count -gt 0) {
                    Write-Output "[Watcher] Change detected. Rebuilding frontend..."

                    Push-Location $feDir
                    try {
                        npm run build 2>&1 | ForEach-Object { Write-Output "[build] $_" }
                    } finally {
                        Pop-Location
                    }

                    if (Test-Path $statDir) {
                        Remove-Item -Recurse -Force $statDir -ErrorAction SilentlyContinue
                    }
                    New-Item -ItemType Directory -Path $statDir -Force | Out-Null | Out-Null
                    Copy-Item -Path (Join-Path $feDir "dist\*") -Destination $statDir -Recurse -Force

                    Write-Output "[Watcher] Rebuild complete. Static assets updated (live without server restart)."
                    $lastState = $current
                } else {
                    $lastState = $current
                }
            }
        } -ArgumentList $FrontendDir, $BackendStaticDir

        # Forward output from both jobs until interrupted
        try {
            while ($true) {
                if ($serverJob) {
                    Receive-Job -Job $serverJob -ErrorAction SilentlyContinue | ForEach-Object { Write-Host $_ }
                    if ($serverJob.State -in @('Completed','Failed','Stopped')) {
                        Write-Host "[Server job ended]" -ForegroundColor Yellow
                        break
                    }
                }
                if ($watcherJob) {
                    Receive-Job -Job $watcherJob -ErrorAction SilentlyContinue | ForEach-Object { Write-Host $_ -ForegroundColor DarkCyan }
                }
                Start-Sleep -Milliseconds 700
            }
        } finally {
            if ($serverJob) { Stop-Job -Job $serverJob -ErrorAction SilentlyContinue | Out-Null; Remove-Job -Job $serverJob -ErrorAction SilentlyContinue | Out-Null }
            if ($watcherJob) { Stop-Job -Job $watcherJob -ErrorAction SilentlyContinue | Out-Null; Remove-Job -Job $watcherJob -ErrorAction SilentlyContinue | Out-Null }
        }
    } else {
        # Normal (non-reload) mode: blocking start as before
        & python -m uvicorn @uvicornArgs

        if ($LASTEXITCODE -ne 0) {
            Write-Error "Server exited with error code $LASTEXITCODE"
            exit $LASTEXITCODE
        }
    }
} finally {
    Pop-Location
    # Ensure any stray jobs are cleaned (belt and suspenders)
    if ($serverJob) { Stop-Job -Job $serverJob -ErrorAction SilentlyContinue | Out-Null; Remove-Job -Job $serverJob -ErrorAction SilentlyContinue | Out-Null }
    if ($watcherJob) { Stop-Job -Job $watcherJob -ErrorAction SilentlyContinue | Out-Null; Remove-Job -Job $watcherJob -ErrorAction SilentlyContinue | Out-Null }
}
