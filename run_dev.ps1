# Запуск полного стека Earth Change Detector (Python ML Engine + Go Gateway + Web UI)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  TerraChange Platform: Starting Services..." -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$CurrentDir = $PSScriptRoot
$PythonExe = Join-Path $CurrentDir "ml-service\.venv\Scripts\python.exe"
$GoServerExe = Join-Path $CurrentDir "backend-go\server.exe"

# 1. Запуск Python ML Engine
Write-Host "[1/2] Запуск Python ML Service на порту 8000..." -ForegroundColor Yellow
$MLProcess = Start-Process -FilePath $PythonExe -ArgumentList "server.py" -WorkingDirectory (Join-Path $CurrentDir "ml-service") -PassThru

Start-Sleep -Seconds 2

# 2. Запуск скомпилированного Go Gateway
Write-Host "[2/2] Запуск Go API Gateway & Web Server на порту 8080..." -ForegroundColor Yellow
$GoProcess = Start-Process -FilePath $GoServerExe -WorkingDirectory (Join-Path $CurrentDir "backend-go") -PassThru

Start-Sleep -Seconds 2

Write-Host "`nВсе сервисы успешно запущены!" -ForegroundColor Green
Write-Host "Веб-интерфейс доступен по адресу: http://localhost:8080" -ForegroundColor Green
Start-Process "http://localhost:8080"

Write-Host "`nНажмите Enter для остановки сервисов..." -ForegroundColor Gray
$null = Read-Host

Stop-Process -Id $MLProcess.Id -Force -ErrorAction SilentlyContinue
Stop-Process -Id $GoProcess.Id -Force -ErrorAction SilentlyContinue
Write-Host "Сервисы остановлены." -ForegroundColor Cyan
