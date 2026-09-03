@echo off
chcp 65001 >/dev/null
echo ========================================
echo   制造业Agent - 一键启动
echo ========================================
echo.
echo [1/2] 后端...
start "Backend" cmd /c "cd /d %~dp0.. && python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000"
echo 等待加载(首次2-3分钟)...
timeout /t 10 /nobreak >/dev/null
echo [2/2] 前端...
start "Frontend" cmd /c "cd /d %~dp0..\frontend\chat && npm run dev"
echo.
echo http://localhost:5173
pause
