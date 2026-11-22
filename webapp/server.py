"""THREADORA private web application — FastAPI backend.

Run:  python run_web.py            (opens http://127.0.0.1:8765, this computer only)

Environment variables (all optional):
  THREADORA_DATA_DIR       where orders are stored (default: <app>/threadora_orders)
  THREADORA_ACCESS_TOKEN   if set, every API call must send this token (for use beyond localhost)
  THREADORA_TESSERACT_CMD  full path to tesseract.exe if it is not found automatically
"""
from __future__ import annotations

import io
from contextlib import asynccontextmanager
import os
import secrets
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from threadora import __version__ as ENGINE_VERSION
from threadora.dmc import DATA_SHA256, dataset_sha256
from threadora.features import ocr_status
from threadora.sizing import CM_PER_IN, FABRICS

from .jobs import JobStore

APP_ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
PRESET_INCHES = (3, 5, 7, 9, 12, 16, 20)
MIN_IN, MAX_IN = 1.0, 60.0
MAX_UPLOAD = 40 * 1024 * 1024
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}
Image.MAX_IMAGE_PIXELS = 80_000_000


def create_app(data_dir: Path | str | None = None, runner=None, ocr_check=None,
               access_token: str | None = None) -> FastAPI:
    data_dir = Path(data_dir or os.environ.get("THREADORA_DATA_DIR") or APP_ROOT / "threadora_orders")
    token = access_token if access_token is not None else os.environ.get("THREADORA_ACCESS_TOKEN", "")
    store = JobStore(data_dir, runner=runner, ocr_check=ocr_status if ocr_check is None else ocr_check)

    @asynccontextmanager
    async def lifespan(_app):
        yield
        store.shutdown()

    app = FastAPI(title="THREADORA (private)", version=ENGINE_VERSION, docs_url=None, redoc_url=None,
                  openapi_url=None, lifespan=lifespan)
    app.state.store = store

    @app.middleware("http")
    async def private_access(request: Request, call_next):
        if token and request.url.path.startswith("/api/"):
            given = (request.headers.get("x-threadora-token") or request.query_params.get("token")
                     or request.cookies.get("threadora_token") or "")
            if not secrets.compare_digest(given, token):
                return JSONResponse({"detail": "Access token required."}, status_code=401)
        resp = await call_next(request)
        resp.headers["X-Robots-Tag"] = "noindex, nofollow"
        resp.headers["Cache-Control"] = "no-store"
        return resp

    # ------------------------------------------------------------------ info
    @app.get("/api/health")
    def health():
        dmc_ok = dataset_sha256() == DATA_SHA256
        return {"engine_version": ENGINE_VERSION, "dmc_dataset_ok": dmc_ok, "ocr": ocr_status(),
                "ai_services_used": False, "token_required": bool(token)}

    @app.get("/api/options")
    def options():
        return {"presets_in": list(PRESET_INCHES), "min_in": MIN_IN, "max_in": MAX_IN,
                "fabrics": [{"name": f.name, "label": f.label, "stitches_per_inch": f.stitches_per_inch,
                             "practical": f.practical} for f in FABRICS]}

    # ------------------------------------------------------------------ jobs
    @app.post("/api/jobs", status_code=202)
    async def create_job(image: UploadFile = File(...), size: str = Form(...),
                         width: float | None = Form(None), height: float | None = Form(None),
                         units: str = Form("in"), phrase: str = Form(""), name: str = Form(""),
                         year: str = Form(""), display_fabric: str = Form("")):
        raw = await image.read(MAX_UPLOAD + 1)
        if len(raw) > MAX_UPLOAD:
            raise HTTPException(413, "Image is larger than 40 MB.")
        if not raw:
            raise HTTPException(400, "The uploaded file is empty.")
        try:
            with Image.open(io.BytesIO(raw)) as im:
                fmt = im.format
                im.verify()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise HTTPException(400, "This file is not a readable image. Upload a PNG, JPG or WEBP.")
        if fmt not in ALLOWED_FORMATS:
            raise HTTPException(400, f"{fmt} images are not supported. Upload a PNG, JPG or WEBP.")

        if size == "custom":
            if width is None or height is None:
                raise HTTPException(400, "Enter both width and height for a custom size.")
            if units not in ("in", "cm"):
                raise HTTPException(400, "Units must be in or cm.")
            w, h = float(width), float(height)
        else:
            try:
                s = int(size)
            except ValueError:
                raise HTTPException(400, "Unknown size.")
            if s not in PRESET_INCHES:
                raise HTTPException(400, "Unknown size.")
            w = h = float(s)
            units = "in"
        w_in, h_in = (w, h) if units == "in" else (w / CM_PER_IN, h / CM_PER_IN)
        if not (MIN_IN <= w_in <= MAX_IN and MIN_IN <= h_in <= MAX_IN):
            raise HTTPException(400, f"Width and height must each be between {MIN_IN:g} and {MAX_IN:g} in "
                                     f"({MIN_IN * CM_PER_IN:.1f}–{MAX_IN * CM_PER_IN:.0f} cm).")
        if display_fabric and display_fabric not in {f.name for f in FABRICS}:
            raise HTTPException(400, "Unknown fabric.")
        personalization = {"phrase": phrase.strip()[:120], "name": name.strip()[:60], "year": year.strip()[:20]}
        req = {"size": size, "width": w, "height": h, "units": units,
               "personalization": personalization, "display_fabric": display_fabric or None}
        job = store.create(raw, image.filename or "original", req)
        return {"id": job.id, "state": job.state}

    @app.get("/api/jobs")
    def list_jobs():
        return [{"id": j.id, "created": j.created, "state": j.state, "source_name": j.source_name,
                 "request": j.request, "status": (j.result or {}).get("status")} for j in store.list()[:50]]

    def _job(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, "Order not found.")
        return job

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, log_lines: int = 60):
        job = _job(job_id)
        d = job.to_json()
        d["log"] = job.log[-max(1, min(log_lines, 400)):]
        return d

    @app.get("/api/jobs/{job_id}/pdf")
    def download_pdf(job_id: str):
        job = _job(job_id)
        name = (job.result or {}).get("pdf_name")
        if not name or not (job.order_dir / name).exists():
            raise HTTPException(404, "No PDF was produced for this order.")
        return FileResponse(job.order_dir / name, media_type="application/pdf", filename=name)

    _IMAGES = {"preview": "stitched_preview_from_matrix.png", "comparison": "comparison_source_vs_matrix.png"}

    @app.get("/api/jobs/{job_id}/image/{kind}")
    def job_image(job_id: str, kind: str):
        job = _job(job_id)
        if kind == "source":
            return FileResponse(job.source_path)
        if kind not in _IMAGES or not (job.order_dir / "private" / _IMAGES[kind]).exists():
            raise HTTPException(404, "Image not available.")
        return FileResponse(job.order_dir / "private" / _IMAGES[kind], media_type="image/png")

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app

