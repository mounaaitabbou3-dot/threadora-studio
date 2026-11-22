"""Browser test: drive the real page in headless Chromium against a live server running
the real engine: upload, pick 3x3 in, generate, wait, click Download PDF."""
import os
import socket
import threading
import time

import pytest
import uvicorn

from webapp.server import create_app

from .sample import sample_png

playwright = pytest.importorskip("playwright.sync_api")
pytestmark = [pytest.mark.browser, pytest.mark.real_engine]


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


@pytest.fixture
def live_server(tmp_path):
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(tmp_path / "orders", access_token=""),
                                           host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True); t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}/"
    server.should_exit = True
    t.join(10)


def test_upload_generate_download(live_server, tmp_path):
    img = tmp_path / "love.png"
    img.write_bytes(sample_png())
    with playwright.sync_playwright() as p:
        exe = os.environ.get("THREADORA_CHROMIUM")      # optional: use an already installed Chromium
        browser = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
        page = browser.new_page(accept_downloads=True)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(live_server)
        page.wait_for_selector("text=OCR ready")
        assert page.locator(".chip").count() == 8                     # 7 presets + custom
        page.set_input_files("#image", str(img))
        page.click("text=3×3 in")
        page.click("summary:has-text('Personalization')")
        page.fill("#name", "LOVE")
        page.click("#go")
        page.wait_for_selector("#result-card:not([hidden])", timeout=300_000)
        status = page.inner_text("#status")
        assert status == "THREADORA VERIFIED — COMPLETE", page.inner_text("#log2")
        assert page.inner_text("#dl") == "Download PDF"
        assert page.locator("#legend tbody tr").count() >= 2
        with page.expect_download() as dl:
            page.click("#dl")
        path = dl.value.path()
        assert open(path, "rb").read(5) == b"%PDF-"
        assert dl.value.suggested_filename.startswith("THREADORA_")
        page.click("#again")
        page.click("text=Custom")
        assert page.is_visible("#width")
        assert errors == []
        browser.close()
