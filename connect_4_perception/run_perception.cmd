@echo off
setlocal
cd /d "%~dp0"
if not exist "..\.venv\Scripts\python.exe" (
  echo Create the project .venv and install requirements.txt first. See README.md.
  pause
  exit /b 1
)
"..\.venv\Scripts\python.exe" "%~dp0run_perception.py" %*
set "simulation_exit=%errorlevel%"
if not "%simulation_exit%"=="0" pause
exit /b %simulation_exit%
