@echo off
REM Overnight ANVIL test-and-triage loop. Double-click and leave running.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Soak-Anvil.ps1" %*
echo.
echo Soak finished. See test-reports\soak-summary.md
pause >nul
