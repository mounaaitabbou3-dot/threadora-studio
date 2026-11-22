# THREADORA V13 — universal cross-stitch engine (private)

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
`--display-fabric "Aida 14"` makes the engine use that fabric. The stitch grid is fitted to that fabric
INSIDE the requested size (fabric and grid are chosen together). If the details cannot be kept on it, the
result is SIZE-FIDELITY CONFLICT with alternatives. The requested size is never changed.

## Pipeline (threadora/)
- `source.py` – SHA-256 of original bytes, EXIF orientation, PNG alpha → NO_STITCH (alpha ≥ 128 is stitched),
  conservative border flood-fill for opaque images, printed-chart detection (type B).
- `sizing.py` – fabrics 11–22ct Aida, 25–40ct evenweave over one; integer grid that fits inside the request
  with the aspect ratio kept. Fabrics above 28 stitches/inch are flagged not practical. `physical_size()` is the
  only finished-size formula (stitches ÷ stitches per inch); `size_problems()` enforces "never larger than
  ordered, fills one side, aspect kept". The PDF file name carries the TRUE finished size (e.g. `_6.8x7.0in`).
- `features.py` – lettering lines (detached alpha components grouped into lines, OCR via Tesseract),
  ornaments, small high-contrast blobs (eyes, highlights); fidelity gates vs the matrix preview:
  OCR on an actual-size simulation, letter parts and counters, mask IoU, blob contrast, SSIM, ΔE2000, edge F1.
- `convert.py` – exact area integration (linear light), edge-aware and dominant-color variants,
  k-means in Lab snapped to DMC (CIEDE2000, chroma tie-break), near-duplicate thread merge, confetti cleanup.
- `fidelity.py` – design audit of the matrix against the original, cell by cell: extra stitches, missing
  stitches, accidental holes vs intentional negative space (transparent areas, letter counters), filled source
  gaps, stitched islands without source, color bleeding at hard edges, unsupported confetti, uneven flat areas.
  Source-backed repair only: bleeding cells take a thread their own source pixels show; 1–2 cell pinholes in
  painted areas are filled; isolated specks change only if the neighbour thread fits the source cell as well.
- `engine.py` – for each fabric its own grid inside the requested size; area / edge-aware / dominant candidates
  are compared; simplest fabric whose grid passes every gate, then fewest colors within quality tolerance,
  preferring candidates with no review findings.
- `pdf.py` – cover, materials+size table, legend with exact counts, matrix overview, charts with
  global labels, bold 10×10 lines, 2-cell ghost overlaps (grey, never counted), mini-map navigation.
- Fonts: DejaVu bundled in `threadora/data/fonts` (free license) so PDFs match on every OS.
- `qa.py` + `tests/` – rebuilds the matrix from the PDF text layer and asserts equality (overlap reference
  cells checked separately: outside their tile, in the 2-cell band, identical to the master), renders every chart
  cell and compares its color with the master, cover/file-name size truth, tile ownership, page/label/font/
  clipping/legibility checks, independent recomputation of the design audit.
- `tests/test_fidelity.py` – injects each defect class (extra cross, missing stitch, hole, bleeding, wrong size,
  duplicated tile cells, wrong DMC, shifted page, PDF print defects) and asserts it is caught.
- `tests/test_regression_engine.py` (`-m real_engine`, slow) – real engine on 11 synthetic design types
  (`tests/designs.py`) and every preset size + custom sizes, full order tests on every PDF.

V13 additions:
- White stitch vs NO_STITCH: near-white source artwork left unstitched is a hard failure (white is never dropped
  because it matches the fabric); white source cells on a dark thread are flagged. Opaque backgrounds are removed
  edge-aware, so a white shape outlined on white paper stays stitched.
- Broken outlines: line-like source shapes that fall apart in the chart are re-joined along their own skeleton
  where the source paints the line on ≥25 % of the cell; remaining stitchable breaks block VERIFIED. Breaks in
  lines thinner than a quarter stitch are counted (`sub_stitch_outline_breaks`) but cannot be full crosses.
- Near-duplicate DMC threads (CIEDE2000 < 5) are merged in the palette and flagged in the audit.
- PDF: automatic title (personalization → OCR'd lettering → meaningful file name → "Cross Stitch Design", never
  "Pattern"); clean stitched-result page; separate chart-navigation page; floss in meters and skeins calculated per
  color from stitch count, fabric count and strands.
- Size A/B: the cover and the app show A = exact stitched design size and B = the maximum box requested.
- `report.py`: five-gate quality report (technical, physical size, design fidelity, DMC/matrix consistency,
  print readability) built from measured values and the tests that actually ran; shown before download,
  saved as `private/quality_report.{json,md}`. No overall numeric score.
- `tools/benchmark.py`: runs the engine on every synthetic design (+ real images) and writes a comparison table.

Status lines (`pipeline.decide_status`):
- `THREADORA VERIFIED — COMPLETE` – tests pass, gates pass, size kept, practical fabric, no open review point.
- `THREADORA REVIEW_NOT_VERIFIED` – technically correct, but automated confidence is not sufficient (e.g. script
  name not readable by OCR, possible bleeding, light details touching a removed background). Reasons listed.
- `THREADORA SIZE-FIDELITY CONFLICT` – details cannot be kept at this size on a practical fabric.
- `THREADORA TECHNICAL FAILURE` – a test or size invariant failed.

## Known limits
- Type B (existing chart) sources are detected and stopped: no legend-free symbol→DMC guessing.
- No backstitch or fractional stitches are generated yet (none are invented).
- Script lettering that Tesseract cannot read in the source is checked structurally only.
- DMC colors are screen approximations (see `threadora/data/DATA_SOURCES.md`).
