@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Check-Prerequisites.ps1"
if errorlevel 1 (
  echo The desktop app was not started. See the prerequisite message above.
  pause
  exit /b 1
)
start "" "%~dp0anime_graph_desktop.exe"
exit /b 0
