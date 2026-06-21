$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$appRoot = Split-Path -Parent $scriptDir
$sourceExe = Join-Path $appRoot "src-tauri\target\release\amcp-desktop.exe"
$distDir = Join-Path $appRoot "dist"
$distExe = Join-Path $distDir "AMCP.exe"

Push-Location $appRoot
try {
    npm --prefix ../frontend run build
    if ($LASTEXITCODE -ne 0) {
        throw "Frontend build failed with exit code $LASTEXITCODE"
    }

    cargo build --manifest-path src-tauri/Cargo.toml --release --features desktop --bin amcp-desktop
    if ($LASTEXITCODE -ne 0) {
        throw "Rust desktop release build failed with exit code $LASTEXITCODE"
    }

    if (-not (Test-Path $sourceExe)) {
        throw "Tauri release executable was not found: $sourceExe"
    }

    New-Item -ItemType Directory -Force -Path $distDir | Out-Null
    Copy-Item -Force -Path $sourceExe -Destination $distExe

    $item = Get-Item $distExe
    Write-Host "Windows distributable executable created:"
    Write-Host "  $($item.FullName)"
    Write-Host "  $([Math]::Round($item.Length / 1MB, 2)) MB"
}
finally {
    Pop-Location
}
