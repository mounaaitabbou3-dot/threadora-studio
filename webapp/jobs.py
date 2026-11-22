"""Order jobs: save the original upload byte-for-byte, run the tested THREADORA engine
(threadora.pipeline.run_order) in a background worker, and persist every job on disk.

Nothing here converts images or draws charts itself. All pattern work is done by the
engine, so the web app and the command line always produce the same verified result.
No network or AI service is called at any point.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

VERIFIED = "THREADORA VERIFIED — COMPLETE"
CONFLICT = "THREADORA SIZE-FIDELITY CONFLICT"
FAILURE = "THREADORA TECHNICAL FAILURE"

STATES = ("queued", "running", "done", "interrupted")
LOG_KEEP = 400


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_filename(name: str, default: str = "original") -> str:
    stem = Path(name or default).name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or default
    return stem[:80]


@dataclass
class Job:
    id: str
    folder: Path
    created: str
    request: dict
    source_name: str
    source_sha256: str
    state: str = "queued"
    updated: str = ""
    log: list[str] = field(default_factory=list)
    result: dict | None = None

    @property
    def source_path(self) -> Path:
        return self.folder / "source" / self.source_name

    @property
    def order_dir(self) -> Path:
        return self.folder / "order"

    def to_json(self) -> dict:
        return {"id": self.id, "created": self.created, "updated": self.updated, "state": self.state,
                "request": self.request, "source_name": self.source_name,
                "source_sha256": self.source_sha256, "log": self.log[-LOG_KEEP:], "result": self.result}

    @classmethod
    def from_json(cls, folder: Path, d: dict) -> "Job":
        return cls(id=d["id"], folder=folder, created=d["created"], request=d["request"],
                   source_name=d["source_name"], source_sha256=d["source_sha256"], state=d["state"],
                   updated=d.get("updated", ""), log=d.get("log", []), result=d.get("result"))


def summarize(audit: dict, order_dir: Path) -> dict:
    """Customer-facing summary built from the engine's audit and the locked master matrix."""
    out = {k: audit.get(k) for k in ("status", "blocker", "pdf_pages", "dmc_colors", "display_fabric",
                                     "chart_resolution_fabric", "fabric_practical", "within_request",
                                     "alternatives_same_chart", "tests", "seconds")}
    pdf = audit.get("pdf")
    out["pdf_name"] = Path(pdf).name if pdf and Path(pdf).exists() else None
    out["pdf_mb"] = round(audit["pdf_bytes"] / 1e6, 2) if audit.get("pdf_bytes") else None
    master_file = order_dir / "private" / "master_matrix.json"
    if master_file.exists():
        m = json.loads(master_file.read_text())
        out["stitches"] = {"cols": m["cols"], "rows": m["rows"],
                           "total": sum(p["count"] for p in m["palette"])}
        out["fabric"] = m["fabric"]["label"]
        out["finished_in"] = m["physical_in"]
        out["finished_cm"] = m["physical_cm"]
        out["requested_in"] = m["requested_in"]
        out["palette"] = [{k: p[k] for k in ("code", "name", "hex", "symbol", "count", "skeins")}
                          for p in m["palette"]]
    out["has_preview"] = (order_dir / "private" / "stitched_preview_from_matrix.png").exists()
    out["has_comparison"] = (order_dir / "private" / "comparison_source_vs_matrix.png").exists()
    out["verified"] = out.get("status") == VERIFIED
    return out


class JobStore:
    """Jobs live in <data_dir>/<job id>/ : job.json, source/<original bytes>, order/<engine output>."""

    def __init__(self, data_dir: Path, runner: Callable | None = None, ocr_check: Callable | None = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="threadora")  # CPU-heavy: one at a time
        self._runner = runner
        self._ocr_check = ocr_check
        self._load()

    # ---------------------------------------------------------------- persistence
    def _load(self):
        for jf in sorted(self.data_dir.glob("*/job.json")):
            try:
                job = Job.from_json(jf.parent, json.loads(jf.read_text(encoding="utf-8")))
            except Exception:
                continue
            if job.state in ("queued", "running"):       # server stopped mid-run
                job.state = "interrupted"
                job.result = {"status": FAILURE, "blocker": "The app was closed while this order was running. "
                              "Generate it again.", "verified": False}
                self._save(job)
            self._jobs[job.id] = job

    def _save(self, job: Job):
        job.updated = now()
        tmp = job.folder / "job.json.tmp"
        tmp.write_text(json.dumps(job.to_json(), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(job.folder / "job.json")

    # ---------------------------------------------------------------- api
    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)

    def create(self, raw: bytes, filename: str, request: dict) -> Job:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_id = f"{stamp}-{uuid.uuid4().hex[:6]}"
        folder = self.data_dir / job_id
        (folder / "source").mkdir(parents=True)
        job = Job(id=job_id, folder=folder, created=now(), request=request,
                  source_name=safe_filename(filename), source_sha256=hashlib.sha256(raw).hexdigest())
        job.source_path.write_bytes(raw)                 # original bytes, never modified
        job.log.append(f"{now()} received {filename!r} ({len(raw)} bytes) sha256={job.source_sha256}")
        with self._lock:
            self._jobs[job_id] = job
            self._save(job)
        self._pool.submit(self._run, job)
        return job

    def _log(self, job: Job, msg):
        with self._lock:
            for line in str(msg).splitlines() or [""]:
                job.log.append(f"{now()} {line}")
            if len(job.log) > LOG_KEEP * 2:
                job.log[:] = job.log[-LOG_KEEP:]
            self._save(job)

    def _run(self, job: Job):
        with self._lock:
            job.state = "running"
            self._save(job)
        req = job.request
        try:
            if self._ocr_check is not None:
                ocr = self._ocr_check()
                if not ocr["available"]:
                    raise EngineBlocked("Lettering checks need Tesseract OCR, which is not available: "
                                        f"{ocr['detail']} Install it (see INSTALL_WINDOWS.md) and generate again.")
            runner = self._runner or _engine_runner
            audit = runner(job.source_path, req["width"], req["height"], req["units"],
                           req["personalization"], job.order_dir, req.get("display_fabric"),
                           lambda m: self._log(job, m))
            result = summarize(audit, job.order_dir)
            if hashlib.sha256(job.source_path.read_bytes()).hexdigest() != job.source_sha256:
                result.update(status=FAILURE, blocker="Original upload changed during processing.", verified=False)
        except EngineBlocked as e:
            self._log(job, str(e))
            result = {"status": FAILURE, "blocker": str(e), "verified": False}
        except Exception as e:
            self._log(job, traceback.format_exc())
            result = {"status": FAILURE, "blocker": f"Engine error: {type(e).__name__}: {e}", "verified": False}
        with self._lock:
            job.result = result
            job.state = "done"
            self._save(job)
        self._log(job, f"finished: {result.get('status')}")

    def wait_idle(self, timeout: float | None = None):
        """Block until queued work is finished (used by tests)."""
        self._pool.submit(lambda: None).result(timeout=timeout)

    def shutdown(self):
        self._pool.shutdown(wait=False, cancel_futures=True)


class EngineBlocked(RuntimeError):
    pass


def _engine_runner(source, width, height, units, personalization, out_dir, display_fabric, log):
    from threadora.pipeline import run_order
    return run_order(source, width, height, units, personalization, out_dir,
                     display_fabric=display_fabric, log=log, run_tests=True)
