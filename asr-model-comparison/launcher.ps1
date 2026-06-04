$proc = Start-Process -PassThru -WindowStyle Hidden -FilePath pwsh -ArgumentList '-NoProfile', '-Command', "cd '$PWD'; .\\run.ps1 --host 127.0.0.1 --port 8000"
$proc.Id | Out-File server.pid
Write-Host 'Launched server wrapper PID:' $proc.Id