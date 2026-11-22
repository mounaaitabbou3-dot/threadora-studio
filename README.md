# THREADORA V10 — universal cross-stitch engine (private)

One image + requested width × height → one A4 color+symbol PDF, plus private audit files.

## Online (Streamlit Community Cloud) — recommended
`app.py` is the Streamlit entry point; `requirements.txt` (pinned, Python 3.12) and `packages.txt`
(Tesseract) are read by Streamlit Cloud. Password in secrets as `THREADORA_PASSWORD`.
Browser-only deployment steps: `DEPLOY_STREAMLIT.md`.

## Local web studio (optional)
```
python run_web.py                  # http://127.0.0.1:8765 — private, offline, no AI services
```
Windows: see `INSTALL_WINDOWS.md` (`windows\install_windows.bat`, then `windows\start_threadora.bat`).

`webapp/` is a FastAPI backend (`server.py`, `jobs.py`) and a plain HTML/CSS/JS frontend (`static/`).
It only calls `threadora.pipeline.run_order`, the same tested engine as the command line, in one background
worker, and stores each order in `threadora_orders/<id>/` (original bytes, PDF, private matrix/audit/test log).
API: `GET /api/health`, `GET /api/options`, `POST /api/jobs` (multipart: image, size=3|5|7|9|12|16|20|custom,
width, height, units, phrase, name, year, display_fabric), `GET /api/jobs/{id}`, `GET /api/jobs/{id}/pdf`,
`GET /api/jobs/{id}/image/{source|preview|comparison}`. Optional `THREADORA_ACCESS_TOKEN`.
If Tesseract is missing the app returns TECHNICAL FAILURE rather than skip lettering checks.

## Command line
```
pip install -r requirements-dev.txt      # plus: apt install tesseract-ocr
python -m threadora.cli IMAGE --width 5 --height 5 --units in \
    --phrase "In Loving Memory" --name Cecilia --year 2026 --out orders/cecilia_5x5
streamlit run app.py --server.address 127.0.0.1     # same app locally (needs .streamlit/secrets.toml)
python -m pytest                                      # engine + web API + browser tests
python -m pytest --order-dir orders/X --source IMAGE  # verify one generated order
```
`--display-fabric "Evenweave 28 over one"` prints the same verified chart for another fabric
(the finished size then changes; the order size is never changed silently).

## Pipeline (threadora/)
- `source.py` – SHA-256 of original bytes, EXIF orientation, PNG alpha → NO_STITCH (alpha ≥ 128 is stitched),
  conservative border flood-fill for opaque images, printed-chart detection (type B).
- `sizing.py` – fabrics 11–22ct Aida, 25–40ct evenweave over one; integer grid that fits inside the request
  with the aspect ratio kept. Fabrics above 28 stitches/inch are flagged not practical.
- `features.py` – lettering lines (detached alpha components grouped into lines, OCR via Tesseract),
  ornaments, small high-contrast blobs (eyes, highlights); fidelity gates vs the matrix preview:
  OCR on an actual-size simulation, letter parts and counters, mask IoU, blob contrast, SSIM, ΔE2000, edge F1.
- `convert.py` – exact area integration (linear light), edge-aware and dominant-color variants,
  k-means in Lab snapped to DMC (CIEDE2000, chroma tie-break), near-duplicate thread merge, confetti cleanup.
- `engine.py` – simplest fabric whose grid passes every gate, then fewest colors within quality tolerance.
- `pdf.py` – cover, materials+size table, legend with exact counts, matrix overview, charts with
  global labels, bold 10×10 lines, 2-cell ghost overlaps (grey, never counted), mini-map navigation.
- Fonts: DejaVu bundled in `threadora/data/fonts` (free license) so PDFs match on every OS.
- `qa.py` + `tests/` – rebuilds the matrix from the PDF text layer and asserts equality, page/label/font/
  clipping/legibility checks.

Status lines: `THREADORA VERIFIED — COMPLETE` only when tests pass, gates pass, size kept and the fabric is
practical; otherwise `THREADORA SIZE-FIDELITY CONFLICT` or `THREADORA TECHNICAL FAILURE` with the blocker.

## Known limits
- Type B (existing chart) sources are detected and stopped: no legend-free symbol→DMC guessing.
- No backstitch or fractional stitches are generated yet (none are invented).
- Script lettering that Tesseract cannot read in the source is checked structurally only.
- DMC colors are screen approximations (see `threadora/data/DATA_SOURCES.md`).
