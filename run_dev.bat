@echo off
chcp 65001 > nul
echo ==========================================================
echo   TerraChange: Запуск Go Gateway и Python ML Engine
echo ==========================================================

set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%ml-service\.venv\Scripts\python.exe
set GO_EXE=%SCRIPT_DIR%backend-go\server.exe

echo [1/2] Запуск Python ML Engine на порту 8000...
start "TerraChange ML Engine (Python)" /min "%PYTHON_EXE%" "%SCRIPT_DIR%ml-service\server.py"

timeout /t 2 /nobreak > nul

echo [2/2] Запуск Go Gateway на порту 8080...
cd /d "%SCRIPT_DIR%backend-go"
start "TerraChange Gateway (Go)" "%GO_EXE%"

timeout /t 2 /nobreak > nul

echo.
echo Открытие веб-интерфейса в браузере...
start http://localhost:8080

echo.
echo Сервисы работают. Чтобы остановить, закройте окна консоли.
pause
