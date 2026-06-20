$venvPy = if (Test-Path 'backend/.venv/Scripts/python.exe') { 'backend/.venv/Scripts/python.exe' } elseif (Test-Path 'backend/venv/Scripts/python.exe') { 'backend/venv/Scripts/python.exe' } else { 'python' }
Write-Host 'Using' $venvPy
if (Test-Path 'backend/static') { Remove-Item 'backend/static' -Recurse -Force -EA SilentlyContinue }
New-Item -ItemType Directory 'backend/static' | Out-Null
Copy-Item 'frontend/dist/*' 'backend/static' -Recurse -Force
Write-Host 'Static copied'
$pyArgs = @('-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', '8000')
$log = 'uvicorn.log'
$p = Start-Process -PassThru -WindowStyle Hidden -FilePath $venvPy -ArgumentList $pyArgs -WorkingDirectory 'backend' -RedirectStandardOutput $log -RedirectStandardError "$log.err"
$p.Id | Out-File 'server.pid'
Write-Host 'Launched PID' $p.Id