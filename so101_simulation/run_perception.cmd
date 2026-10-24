@echo off
cd /d "%~dp0"
if not exist "..\.venv\Scripts\python.exe" (
  echo First create the environment using the commands in PERCEPTION.md.
  pause
  exit /b 1
)
"..\.venv\Scripts\python.exe" run_perception.py %*
if errorlevel 1 pause
