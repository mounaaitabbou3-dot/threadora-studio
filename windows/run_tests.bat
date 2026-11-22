@echo off
REM Runs every THREADORA test: engine unit tests, web API tests and the real browser test.
cd /d "%~dp0.."
set PYTHONUTF8=1
call .venv\Scripts\activate.bat
python -m pytest -q -rs
pause
