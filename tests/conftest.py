import json
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--order-dir", default=None, help="generated order folder to verify")
    parser.addoption("--source", default=None, help="original uploaded image")


@pytest.fixture(scope="session")
def order_dir(request):
    d = request.config.getoption("--order-dir")
    if not d:
        pytest.skip("no --order-dir given")
    return Path(d)


@pytest.fixture(scope="session")
def source_path(request):
    s = request.config.getoption("--source")
    if not s:
        pytest.skip("no --source given")
    return Path(s)


@pytest.fixture(scope="session")
def audit(order_dir):
    return json.loads((order_dir / "private" / "audit.json").read_text()) if (order_dir / "private" / "audit.json").exists() else None


@pytest.fixture(scope="session")
def master(order_dir):
    return json.loads((order_dir / "private" / "master_matrix.json").read_text())


@pytest.fixture(scope="session")
def chart_info(order_dir):
    return json.loads((order_dir / "private" / "chart_info.json").read_text())


@pytest.fixture(scope="session")
def pdf_path(order_dir):
    pdfs = sorted(order_dir.glob("THREADORA_*.pdf"))
    assert len(pdfs) == 1, f"expected exactly one customer PDF, found {pdfs}"
    return pdfs[0]
