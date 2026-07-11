@echo off
REM Double-click launcher for ANVIL. Runs the PowerShell starter with the
REM execution policy bypassed for this one process only (no system changes).
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-Anvil.ps1" %*
if %ERRORLEVEL% NEQ 0 (
  echo.
  echo ANVIL exited with an error. Press any key to close.
  pause >nul
)
