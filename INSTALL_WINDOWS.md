# THREADORA Studio — Windows installation

The studio runs entirely on your own PC. Pattern generation uses no Claude, AI or other paid
online service; after installation it works offline.

## What you need (one time)

1. **Python 3.11 or newer** — https://www.python.org/downloads/windows/
   During setup tick **"Add python.exe to PATH"**.
2. **Tesseract OCR** (reads lettering for the legibility checks). Open *PowerShell* and run:
   ```
   winget install -e --id UB-Mannheim.TesseractOCR
   ```
   or download the installer from https://github.com/UB-Mannheim/tesseract/wiki and keep the
   default folder `C:\Program Files\Tesseract-OCR`. The app finds it there automatically.
   If you install it elsewhere, set `THREADORA_TESSERACT_CMD` to the full path of `tesseract.exe`.
   Without Tesseract the app refuses to generate (it reports TECHNICAL FAILURE) instead of
   skipping the lettering checks.

## Install

1. Copy the whole `threadora` folder (the one containing `run_web.py`) to your PC,
   for example `C:\THREADORA\threadora`. Copy files as they are: do not open and re-save
   `threadora\data\dmc_floss.csv` in Excel, because the engine checks its fingerprint.
2. Double-click `windows\install_windows.bat`.
   It creates a private Python environment (`.venv`), installs the packages, installs the
   test browser, and runs the full test suite. The last lines should say all tests passed.

## Use

1. Double-click `windows\start_threadora.bat`. Your browser opens **http://127.0.0.1:8765**.
   Keep the black window open while you work; close it to stop the studio.
2. Upload the original image, pick 3×3 … 20×20 in or Custom (inches or cm), optionally type
   the text that appears in the artwork, and click **Generate PDF**.
3. Wait for the result (seconds for simple art, a few minutes for detailed designs):
   - **THREADORA VERIFIED — COMPLETE**: tests and detail checks passed → **Download PDF**.
   - **THREADORA SIZE-FIDELITY CONFLICT**: the details do not survive at that size on a
     practical fabric. You get the reason, the same chart's size on other fabrics, and only a
     *review copy* to look at. Pick another size or fabric and generate again.
   - **THREADORA TECHNICAL FAILURE**: something failed; the reason is shown.

Orders are saved in `threadora_orders\<date-time-id>\`:
`source\` (your original, byte-for-byte), `order\THREADORA_*.pdf` (customer PDF) and
`order\private\` (master stitch matrix JSON, preview, source hash, audit, test log).
Only the PDF goes to the customer.

## Privacy

The studio listens only on `127.0.0.1`, so nobody else on the network can open it.
To open it from another device on your home network, first set an access token, then
start it on all interfaces:
```
set THREADORA_ACCESS_TOKEN=choose-a-long-secret
.venv\Scripts\python run_web.py --host 0.0.0.0
```
The page then asks for the token once. Do not expose it to the internet.

## Tests

Double-click `windows\run_tests.bat`, or run `python -m pytest -q` in the folder.
It runs the engine unit tests, the web API tests, a real end-to-end order through the engine,
and a headless-browser test that uploads an image, generates and downloads a PDF.

## Troubleshooting

- *"Python was not found"*: reinstall Python with "Add python.exe to PATH" ticked.
- *Header says "Tesseract OCR not found"*: install Tesseract (above) and restart the studio.
- *Port 8765 in use*: `.venv\Scripts\python run_web.py --port 8800`.
- *Command line instead of the browser*:
  `.venv\Scripts\python -m threadora.cli image.png --width 5 --height 5 --units in --out orders\name`
