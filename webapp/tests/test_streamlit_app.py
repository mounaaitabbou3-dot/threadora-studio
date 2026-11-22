"""Tests for the Streamlit Community Cloud app (app.py).

Fast tests use Streamlit's AppTest. The browser test starts `streamlit run app.py` for real,
logs in, uploads an image, generates with the real engine and downloads the PDF."""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from .sample import sample_png

ROOT = Path(__file__).resolve().parents[2]
APP = str(ROOT / "app.py")


def test_refuses_to_run_without_password_configured():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("no password configured" in e.value for e in at.error)
    assert len(at.button) == 0                       # nothing usable


def test_wrong_password_is_rejected_and_right_one_opens_form():
    at = AppTest.from_file(APP, default_timeout=60)
    at.secrets["THREADORA_PASSWORD"] = "correct horse"
    at.run()
    assert len(at.text_input) == 1 and not at.radio      # only the login form
    at.text_input[0].input("nope")
    at.button[0].click().run()
    assert any("Wrong password" in e.value for e in at.error)
    assert not at.radio
    at.text_input[0].input("correct horse")
    at.button[0].click().run()
    assert not at.exception
    sizes = at.radio[0]
    assert list(sizes.options) == ["3×3 in", "5×5 in", "7×7 in", "9×9 in", "12×12 in", "16×16 in", "20×20 in", "Custom"]
    gen = [b for b in at.button if b.label == "Generate PDF"][0]
    assert gen.disabled                                    # no image yet
    sizes.set_value("Custom").run()
    assert [n.label for n in at.number_input] == ["Width", "Height"]
    assert at.selectbox[0].options == ["in", "cm"]


def test_size_and_image_validation_helpers():
    import importlib.util
    # import helpers without executing the page twice: AppTest already covers main()
    src = Path(APP).read_text()
    ns = {"__name__": "td_helpers"}
    exec(compile(src.replace("\nmain()\n", "\n"), APP, "exec"), ns)
    assert ns["validate_size"](5, 5, "in") is None
    assert ns["validate_size"](0.5, 5, "in")
    assert ns["validate_size"](200, 20, "cm")
    assert ns["validate_size"](30, 40, "cm") is None
    assert "empty" in ns["validate_image"](b"")
    assert "not a readable image" in ns["validate_image"](b"hello")
    assert ns["validate_image"](sample_png(64)) is None


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


@pytest.fixture
def streamlit_server():
    secrets = ROOT / ".streamlit" / "secrets.toml"
    created = not secrets.exists()
    if created:
        secrets.write_text('THREADORA_PASSWORD = "e2e-test-pass"\n')
    port = free_port()
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", APP, "--server.headless=true",
                             f"--server.port={port}", "--server.address=127.0.0.1"],
                            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        for _ in range(150):
            try:
                socket.create_connection(("127.0.0.1", port), 0.2).close(); break
            except OSError:
                time.sleep(0.2)
        yield f"http://127.0.0.1:{port}/", secrets.read_text().split('"')[1]
    finally:
        proc.terminate(); proc.wait(15)
        if created:
            secrets.unlink()


@pytest.mark.browser
@pytest.mark.real_engine
def test_streamlit_login_upload_generate_download(streamlit_server, tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    url, password = streamlit_server
    img = tmp_path / "love.png"
    img.write_bytes(sample_png())
    with playwright.sync_playwright() as p:
        exe = os.environ.get("THREADORA_CHROMIUM")
        browser = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
        page = browser.new_page(accept_downloads=True)
        page.goto(url)
        page.get_by_label("Password", exact=True).fill(password)
        page.get_by_role("button", name="Enter").click()
        page.wait_for_selector("text=New pattern", timeout=60_000)
        page.locator("input[type=file]").set_input_files(str(img))
        page.get_by_text("3×3 in", exact=True).click()
        page.get_by_text("3 · Personalization (optional)").click()
        page.get_by_label("Name", exact=True).fill("LOVE")
        page.get_by_label("Name", exact=True).press("Tab")
        page.get_by_role("button", name="Generate PDF").click()
        status = page.locator("[data-testid=td-status]")
        status.wait_for(timeout=300_000)
        assert status.inner_text() == "THREADORA VERIFIED — COMPLETE"
        page.get_by_text("STITCHED RESULT").first.wait_for()
        with page.expect_download() as dl:
            page.get_by_role("button", name="Download PDF").click()
        data = Path(dl.value.path()).read_bytes()
        assert data[:5] == b"%PDF-" and dl.value.suggested_filename == "THREADORA_LOVE_2.6x3.0in.pdf"  # true finished size
        with page.expect_download() as dl2:
            page.get_by_role("button", name="Download private audit files (ZIP)").click()
        import zipfile
        names = zipfile.ZipFile(dl2.value.path()).namelist()
        assert "private/master_matrix.json" in names and "private/test_log.txt" in names
        page.screenshot(path=str(tmp_path / "streamlit_result.png"), full_page=True)
        browser.close()
