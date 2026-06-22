$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$appRoot = Split-Path -Parent $scriptDir
$repoRoot = Split-Path -Parent $appRoot
$sourceExe = $null
$distDir = Join-Path $appRoot "dist"
$distExe = Join-Path $distDir "AMCP.exe"
$distWebDir = Join-Path $distDir "web"
$frontendDistDir = Join-Path $repoRoot "frontend\dist"
$voxtralPatchedBinDir = $null

function Set-DefaultEnvPath {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Path
    )

    if (-not [Environment]::GetEnvironmentVariable($Name, "Process") -and (Test-Path $Path)) {
        [Environment]::SetEnvironmentVariable($Name, (Resolve-Path $Path).Path, "Process")
    }
}

function Initialize-AcceleratedCargoBuildEnv {
    if (-not [Environment]::GetEnvironmentVariable("CARGO_TARGET_DIR", "Process")) {
        [Environment]::SetEnvironmentVariable("CARGO_TARGET_DIR", "C:\t", "Process")
    }
    if (-not [Environment]::GetEnvironmentVariable("CMAKE_BUILD_PARALLEL_LEVEL", "Process")) {
        [Environment]::SetEnvironmentVariable("CMAKE_BUILD_PARALLEL_LEVEL", "1", "Process")
    }

    # whisper-rs Vulkan builds reliably with the Visual Studio generator when
    # the target path is short. A stale Ninja override conflicts with the
    # Visual Studio instance selected by the cmake crate.
    [Environment]::SetEnvironmentVariable("CMAKE_GENERATOR", $null, "Process")
    [Environment]::SetEnvironmentVariable("CMAKE_GENERATOR_INSTANCE", $null, "Process")
}

function Initialize-VoxtralPatchedLlamaEnv {
    $script:voxtralPatchedBinDir = $null
    $patchedRoot = Join-Path $repoRoot ".tmp\llama-cpp-voxtral-pr20638"
    $shortVulkanBuild = "C:\amcp-build\llama-vulkan"
    $patchedVulkanBuild = Join-Path $patchedRoot "build-amcp-vulkan-release"
    $patchedCpuBuild = Join-Path $patchedRoot "build-amcp-cpu-release"
    $patchedBuild = if (Test-Path (Join-Path $shortVulkanBuild "bin\ggml-vulkan.dll")) {
        $shortVulkanBuild
    } elseif (Test-Path (Join-Path $patchedVulkanBuild "bin\ggml-vulkan.dll")) {
        $patchedVulkanBuild
    } else {
        $patchedCpuBuild
    }
    $patchedBin = Join-Path $patchedBuild "bin"

    [Environment]::SetEnvironmentVariable("AMCP_VOXTRAL_PATCHED_LLAMA_DIR", (Resolve-Path $patchedRoot).Path, "Process")
    [Environment]::SetEnvironmentVariable("AMCP_VOXTRAL_PATCHED_LLAMA_LIB_DIR", (Resolve-Path $patchedBuild).Path, "Process")
    [Environment]::SetEnvironmentVariable("AMCP_VOXTRAL_PATCHED_LLAMA_BIN_DIR", (Resolve-Path $patchedBin).Path, "Process")
    if (Test-Path $patchedBin) {
        $script:voxtralPatchedBinDir = (Resolve-Path $patchedBin).Path
    }
    if (Test-Path (Join-Path $patchedBin "ggml-vulkan.dll")) {
        [Environment]::SetEnvironmentVariable("AMCP_VOXTRAL_PATCHED_LLAMA_LINK_VULKAN", "1", "Process")
    }
}

function Copy-VoxtralPatchedLlamaDlls {
    if (-not $script:voxtralPatchedBinDir) {
        return
    }

    $dlls = @(
        "llama.dll",
        "mtmd.dll",
        "ggml.dll",
        "ggml-base.dll",
        "ggml-cpu.dll",
        "ggml-vulkan.dll"
    )

    foreach ($dll in $dlls) {
        $source = Join-Path $script:voxtralPatchedBinDir $dll
        if (Test-Path $source) {
            Copy-Item -Force -Path $source -Destination (Join-Path $distDir $dll)
        }
    }
}

function Invoke-WithRetry {
    param(
        [Parameter(Mandatory = $true)][scriptblock]$Action,
        [Parameter(Mandatory = $true)][string]$Description,
        [int]$Attempts = 10,
        [int]$DelayMilliseconds = 1000
    )

    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        try {
            & $Action
            return
        }
        catch {
            if ($attempt -eq $Attempts) {
                throw
            }
            Write-Warning "$Description failed on attempt $attempt/$Attempts; retrying in $DelayMilliseconds ms. $($_.Exception.Message)"
            Start-Sleep -Milliseconds $DelayMilliseconds
        }
    }
}

Push-Location $appRoot
try {
    Initialize-AcceleratedCargoBuildEnv
    Initialize-VoxtralPatchedLlamaEnv
    $cargoTargetDir = [Environment]::GetEnvironmentVariable("CARGO_TARGET_DIR", "Process")
    $sourceExe = Join-Path $cargoTargetDir "release\amcp-desktop.exe"
    if (Test-Path $sourceExe) {
        Invoke-WithRetry -Description "Removing old release executable" -Action {
            Remove-Item -Force -Path $sourceExe
        }
    }
    if (Test-Path $distExe) {
        try {
            Invoke-WithRetry -Description "Removing old distributable executable" -Action {
                Remove-Item -Force -Path $distExe
            }
        }
        catch {
            $backupExe = Join-Path $distDir ("AMCP.old-{0}.exe" -f (Get-Date -Format "yyyyMMddHHmmss"))
            Write-Warning "Could not remove old distributable executable; moving it aside to $backupExe. $($_.Exception.Message)"
            Invoke-WithRetry -Description "Moving old distributable executable aside" -Action {
                Move-Item -Force -Path $distExe -Destination $backupExe
            }
        }
    }

    npm run build
    if ($LASTEXITCODE -ne 0) {
        throw "Tauri build failed with exit code $LASTEXITCODE"
    }

    if (-not (Test-Path $sourceExe)) {
        throw "Tauri release executable was not found: $sourceExe"
    }

    New-Item -ItemType Directory -Force -Path $distDir | Out-Null
    Invoke-WithRetry -Description "Copying distributable executable" -Action {
        Copy-Item -Force -Path $sourceExe -Destination $distExe
    }
    Copy-VoxtralPatchedLlamaDlls
    if (Test-Path $distWebDir) {
        Remove-Item -Recurse -Force -Path $distWebDir
    }
    if (-not (Test-Path (Join-Path $frontendDistDir "index.html"))) {
        throw "Frontend build output was not found: $frontendDistDir"
    }
    Copy-Item -Recurse -Force -Path $frontendDistDir -Destination $distWebDir

    $item = Get-Item $distExe
    $item.LastWriteTime = Get-Date
    $item.Refresh()
    Write-Host "Windows distributable executable created:"
    Write-Host "  Source: $sourceExe"
    Write-Host "  $($item.FullName)"
    Write-Host "  $([Math]::Round($item.Length / 1MB, 2)) MB"
    Write-Host "  Frontend: $distWebDir"
}
finally {
    Pop-Location
}
