@echo off
setlocal

:: ASR Model Comparison Platform - Single App Startup Script (Windows)
::
:: このバッチファイルは PowerShell スクリプト（run.ps1）の薄いラッパーです。
:: PowerShell から呼び出しても、cmd.exe から呼び出しても安定して動作します。
::
:: 推奨の呼び出し方:
::   run.bat
::   run.bat --host 0.0.0.0 --port 9000
::   run.bat --build-only
::   run.bat --reload
::
:: ログは標準出力に出力されます。
:: ファイルに保存したい場合はリダイレクトしてください（例: run.bat > logs\app.log 2>&1）。

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

:: PowerShell を使用して run.ps1 を呼び出す（最も堅牢）
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%\run.ps1" %*

endlocal
