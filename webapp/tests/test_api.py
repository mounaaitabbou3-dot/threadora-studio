"""FastAPI backend tests. Fast tests use a stand-in engine runner; the `real_engine`
test runs the actual THREADORA engine end to end (about 15-30 s)."""
import hashlib
import json
import time
from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient

from webapp.jobs import CONFLICT, FAILURE, VERIFIED, JobStore
from webapp.server import create_app

from .sample import sample_png

OCR_OK = lambda: {"available": True, "detail": "test"}  # noqa: E731


def fake_runner(calls):
    def run(source, width, height, units, personalization, out_dir, display_fabric, log):
        calls.append(dict(source=Path(source), width=width, height=height, units=units,
                          personalization=personalization, display_fabric=display_fabric))
        log("fake engine ran")
        priv = Path(out_dir) / "private"
        priv.mkdir(parents=True, exist_ok=True)
        return {"status": CONFLICT, "blocker": "fake blocker", "pdf_pages": 3, "dmc_colors": 2}
    return run


def wait_done(client, job_id, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["state"] in ("done", "interrupted"):
            return j
        time.sleep(0.2)
    raise AssertionError("job did not finish")


@pytest.fixture
def calls():
    return []


@pytest.fixture
def client(tmp_path, calls):
    app = create_app(data_dir=tmp_path / "orders", runner=fake_runner(calls), ocr_check=OCR_OK, access_token="")
    with TestClient(app) as c:
        yield c


def post(client, data=None, content=None, filename="art.png", **kw):
    files = {"image": (filename, content if content is not None else sample_png(160), "image/png")}
    return client.post("/api/jobs", data=data or {"size": "5"}, files=files, **kw)


def test_health_reports_offline_engine(client):
    h = client.get("/api/health").json()
    assert h["dmc_dataset_ok"] is True
    assert h["ai_services_used"] is False
    assert h["engine_version"]


def test_options_has_all_preset_sizes_and_fabrics(client):
    o = client.get("/api/options").json()
    assert o["presets_in"] == [3, 5, 7, 9, 12, 16, 20]
    assert any(f["name"] == "Aida 14" for f in o["fabrics"])


def test_frontend_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "THREADORA" in r.text and "app.js" in r.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/styles.css").status_code == 200


@pytest.mark.parametrize("size", ["3", "5", "7", "9", "12", "16", "20"])
def test_every_preset_size_is_passed_to_engine_in_inches(client, calls, size):
    r = post(client, {"size": size})
    assert r.status_code == 202
    j = wait_done(client, r.json()["id"])
    assert calls[-1]["width"] == calls[-1]["height"] == float(size)
    assert calls[-1]["units"] == "in"
    assert j["result"]["status"] == CONFLICT


def test_custom_size_cm_and_personalization(client, calls):
    r = post(client, {"size": "custom", "width": "20", "height": "25.4", "units": "cm",
                      "phrase": " In Loving Memory ", "name": "Cecilia", "year": "2026"})
    assert r.status_code == 202
    wait_done(client, r.json()["id"])
    c = calls[-1]
    assert (c["width"], c["height"], c["units"]) == (20.0, 25.4, "cm")
    assert c["personalization"] == {"phrase": "In Loving Memory", "name": "Cecilia", "year": "2026"}


def test_original_bytes_saved_unchanged(client, calls):
    raw = sample_png(200)
    r = post(client, content=raw, filename="../../evil name.png")
    j = wait_done(client, r.json()["id"])
    src = calls[-1]["source"]
    assert src.read_bytes() == raw
    assert j["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert src.parent.name == "source" and ".." not in src.name
    assert client.get(f"/api/jobs/{j['id']}/image/source").content == raw


@pytest.mark.parametrize("data,content,msg", [
    ({"size": "5"}, b"", "empty"),
    ({"size": "5"}, b"not an image at all", "not a readable image"),
    ({"size": "4"}, None, "Unknown size"),
    ({"size": "custom", "width": "5"}, None, "both width and height"),
    ({"size": "custom", "width": "0.5", "height": "5", "units": "in"}, None, "between"),
    ({"size": "custom", "width": "200", "height": "20", "units": "cm"}, None, "between"),
    ({"size": "custom", "width": "5", "height": "5", "units": "mm"}, None, "Units"),
    ({"size": "5", "display_fabric": "Burlap 3"}, None, "Unknown fabric"),
])
def test_bad_requests_rejected(client, calls, data, content, msg):
    r = post(client, data, content=content)
    assert r.status_code == 400 and msg in r.json()["detail"]
    assert calls == []


def test_gif_rejected(client):
    from io import BytesIO
    from PIL import Image
    b = BytesIO(); Image.new("RGB", (10, 10)).save(b, "GIF")
    r = post(client, content=b.getvalue(), filename="a.gif")
    assert r.status_code == 400 and "GIF" in r.json()["detail"]


def test_no_pdf_means_404_and_unknown_job_404(client):
    j = wait_done(client, post(client).json()["id"])
    assert client.get(f"/api/jobs/{j['id']}/pdf").status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get(f"/api/jobs/{j['id']}/image/../../job.json").status_code == 404


def test_engine_crash_is_technical_failure(tmp_path):
    def boom(*a):
        raise ValueError("kaboom")
    with TestClient(create_app(tmp_path, runner=boom, ocr_check=OCR_OK, access_token="")) as c:
        j = wait_done(c, post(c).json()["id"])
    assert j["result"]["status"] == FAILURE and "kaboom" in j["result"]["blocker"]
    assert j["result"]["verified"] is False


def test_missing_ocr_blocks_instead_of_false_verified(tmp_path, calls):
    no_ocr = lambda: {"available": False, "detail": "Tesseract OCR not found."}  # noqa: E731
    with TestClient(create_app(tmp_path, runner=fake_runner(calls), ocr_check=no_ocr, access_token="")) as c:
        j = wait_done(c, post(c).json()["id"])
    assert j["result"]["status"] == FAILURE and "Tesseract" in j["result"]["blocker"]
    assert calls == []


def test_access_token_enforced(tmp_path, calls):
    with TestClient(create_app(tmp_path, runner=fake_runner(calls), ocr_check=OCR_OK, access_token="s3cret")) as c:
        assert c.get("/api/health").status_code == 401
        assert c.get("/api/health", headers={"X-Threadora-Token": "wrong"}).status_code == 401
        assert c.get("/api/health", headers={"X-Threadora-Token": "s3cret"}).status_code == 200
        assert c.get("/api/jobs?token=s3cret").status_code == 200
        assert c.get("/").status_code == 200          # page loads, then asks for the token


def test_jobs_persist_and_running_jobs_become_interrupted(tmp_path, calls):
    store = JobStore(tmp_path, runner=fake_runner(calls), ocr_check=OCR_OK)
    job = store.create(sample_png(120), "a.png", {"size": "3", "width": 3.0, "height": 3.0, "units": "in",
                                                  "personalization": {}, "display_fabric": None})
    store.wait_idle(30)
    store.shutdown()
    # simulate a crash mid-run on a second job
    d = json.loads((job.folder / "job.json").read_text())
    d.update(id="x-running", state="running", result=None)
    (tmp_path / "x-running").mkdir()
    (tmp_path / "x-running" / "job.json").write_text(json.dumps(d))
    again = JobStore(tmp_path, runner=fake_runner(calls), ocr_check=OCR_OK)
    assert again.get(job.id).state == "done"
    assert again.get("x-running").state == "interrupted"
    assert again.get("x-running").result["status"] == FAILURE
    again.shutdown()


@pytest.mark.real_engine
def test_real_engine_end_to_end(tmp_path):
    """Upload -> real THREADORA engine (incl. its own 69 order tests) -> PDF download."""
    with TestClient(create_app(tmp_path, access_token="")) as c:
        r = post(c, {"size": "3", "name": "LOVE"}, content=sample_png())
        j = wait_done(c, r.json()["id"])
        res = j["result"]
        assert res["status"] in (VERIFIED, CONFLICT, FAILURE)
        assert res["status"] == VERIFIED, res
        assert res["tests"]["returncode"] == 0
        assert res["finished_in"][0] <= 3 + 1e-6 and res["finished_in"][1] <= 3 + 1e-6
        assert sum(p["count"] for p in res["palette"]) == res["stitches"]["total"]
        assert len({p["symbol"] for p in res["palette"]}) == len(res["palette"]) == res["dmc_colors"]
        pdf = c.get(f"/api/jobs/{j['id']}/pdf")
        assert pdf.status_code == 200 and pdf.content[:5] == b"%PDF-"
        doc = fitz.open(stream=pdf.content, filetype="pdf")
        assert doc.page_count == res["pdf_pages"]
        assert len(pdf.content) <= 20 * 1024 * 1024
        assert c.get(f"/api/jobs/{j['id']}/image/preview").status_code == 200
        # master matrix saved privately next to the PDF
        master = json.loads((tmp_path / j["id"] / "order" / "private" / "master_matrix.json").read_text())
        assert master["source_sha256"] == j["source_sha256"]
