@echo off
cd /d "%~dp0"
if not exist "..\.venv\Scripts\python.exe" (
  echo First create the environment using the commands in the repository README.
  pause
  exit /b 1
)
"..\.venv\Scripts\python.exe" play_game.py %*
if errorlevel 1 pause
