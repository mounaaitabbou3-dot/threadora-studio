@echo off
REM Starts the private THREADORA studio at http://127.0.0.1:8765 and opens your browser.
cd /d "%~dp0.."
set PYTHONUTF8=1
if not exist .venv\Scripts\activate.bat (
  echo Run windows\install_windows.bat first.
  pause & exit /b 1
)
call .venv\Scripts\activate.bat
python run_web.py
pause
