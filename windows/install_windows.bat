@echo off
REM THREADORA one-time setup for Windows 10/11. Run from the THREADORA folder (double-click is fine).
setlocal
cd /d "%~dp0.."
set PYTHONUTF8=1
echo === THREADORA setup ===
where py >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.11 or newer from https://www.python.org/downloads/windows/
  echo and tick "Add python.exe to PATH" during setup, then run this file again.
  pause & exit /b 1
)
if not exist .venv (
  echo Creating private Python environment in .venv ...
  py -3 -m venv .venv || (echo Could not create .venv & pause & exit /b 1)
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt || (echo Package install failed & pause & exit /b 1)
python -m playwright install chromium
if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
  echo Tesseract OCR found.
) else (
  echo.
  echo Tesseract OCR is NOT installed yet. It is required for the lettering checks.
  echo Install it with:  winget install -e --id UB-Mannheim.TesseractOCR
  echo or download the installer from https://github.com/UB-Mannheim/tesseract/wiki
)
echo.
echo Running the test suite ...
python -m pytest -q
echo.
echo Setup finished. Start the studio with windows\start_threadora.bat
pause
