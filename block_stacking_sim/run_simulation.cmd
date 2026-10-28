@echo off
cd /d "%~dp0"
if not exist "..\.venv\Scripts\python.exe" (
  echo Create the project .venv and install block_stacking_sim/requirements.txt first.
  pause
  exit /b 1
)
"..\.venv\Scripts\python.exe" -u run_simulation.py %*
set "result=%errorlevel%"
if not "%result%"=="0" pause
exit /b %result%
