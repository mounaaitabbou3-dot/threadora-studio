"""V13 regression tests: white stitches vs NO_STITCH, broken outlines, near-duplicate threads,
edge-aware background removal, automatic titles, calculated floss and the five-gate report."""
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from tests.test_fidelity import WHITE, cs_for, source_from_cells
from threadora import report
from threadora.fidelity import audit, broken_contours, near_duplicate_threads, repair_contours
from threadora.dmc import dmc_by_code, to_lab
from threadora.pipeline import design_title, floss_meters, skeins_for
from threadora.source import load_source

DARK, CREAM = "310", "712"


def ring():
    """16x16: cream panel with a one-stitch dark outline (a frame)."""
    m = np.full((16, 16), -1, np.int32)
    m[2:14, 2:14] = 0
    m[3:13, 3:13] = 1
    return m, [DARK, CREAM]


def test_broken_outline_detected_and_rejoined():
    src, codes = ring()
    rgb, opq = source_from_cells(src, codes)
    cs = cs_for(src, codes, rgb, opq)
    assert audit(src, cs)["broken_contours"] == 0
    bad = src.copy()
    bad[2, 8] = bad[13, 8] = 1                          # two gaps: cream where the outline was
    res = audit(bad, cs)
    assert res["broken_contours"] == 1 and any("broken" in r for r in res["review"])
    assert broken_contours(bad, cs)
    fixed, n = repair_contours(bad, cs)
    assert n == 2 and (fixed == src).all()
    assert audit(fixed, cs)["broken_contours"] == 0


def test_lost_white_highlight_is_hard_failure_and_never_removed_for_matching_fabric():
    """White artwork must stay stitched even though it matches white fabric."""
    src = np.zeros((10, 10), np.int32)
    src[4:6, 4:6] = 1
    codes = ["321", WHITE]
    rgb, opq = source_from_cells(src, codes)
    cs = cs_for(src, codes, rgb, opq)
    assert audit(src, cs)["lost_white_cells"] == 0
    holed = src.copy()
    holed[4:6, 4:6] = -1                                # white removed -> NO_STITCH hole
    res = audit(holed, cs)
    assert res["lost_white_cells"] == 4 and not res["pass"]
    assert any("white artwork" in h for h in res["hard_fail"])


def test_white_source_on_dark_thread_flagged():
    src = np.zeros((10, 10), np.int32)
    src[4:6, 4:6] = 1
    codes = ["321", WHITE, DARK]
    rgb, opq = source_from_cells(src, codes)
    wrong = src.copy()
    wrong[4:6, 4:6] = 2
    res = audit(wrong, cs_for(src, codes, rgb, opq))
    assert res["white_cells_dark_thread"] == 4


def test_intentional_transparent_gap_is_not_white_or_hole():
    src = np.zeros((10, 10), np.int32)
    src[4:6, 4:6] = -1                                  # intentional see-through window
    codes = ["321"]
    rgb, opq = source_from_cells(src, codes)
    res = audit(src, cs_for(src, codes, rgb, opq))
    assert res["lost_white_cells"] == 0 and res["accidental_holes"] == 0 and res["pass"]


def test_near_duplicate_threads_flagged():
    by = dmc_by_code()
    lab = to_lab(np.array([by[c].rgb for c in ("3346", "3346", "321")], np.uint8))
    dups = near_duplicate_threads(lab)
    assert [(i, j) for i, j, _ in dups] == [(0, 1)]


def test_white_artwork_touching_white_paper_is_kept(tmp_path):
    """Edge-aware background: a white snowman outlined in black on white paper stays artwork."""
    im = Image.new("RGB", (300, 300), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.ellipse([60, 60, 240, 240], fill=(255, 255, 255), outline=(20, 20, 20), width=4)
    d.ellipse([130, 120, 145, 135], fill=(20, 20, 20))
    p = tmp_path / "snow.png"
    im.save(p)
    src = load_source(p)
    a = src.opaque
    assert not a[5, 5] and not a[295, 150]              # paper removed
    assert a[150, 150] and a[200, 100]                  # white body inside the outline kept


@pytest.mark.parametrize("pers,ocr,name,expected", [
    ({"phrase": "In Loving Memory", "name": "Rose", "year": "2026"}, [], "x.png",
     "In Loving Memory · Rose · 2026"),
    ({}, ["Merry Christmas"], "IMG_0001.png", "Merry Christmas"),
    ({}, [], "cozy_little_house.png", "Cozy Little House"),
    ({}, [], "83f1fa7a4e2b9c10.png", "Cross Stitch Design"),
    ({}, [], "IMG_2024.jpg", "Cross Stitch Design"),
])
def test_design_title_never_generic(pers, ocr, name, expected):
    class L:
        def __init__(self, t):
            self.source_ocr, self.kind = t, "ocr_text"

    class F:
        text_lines = [L(t) for t in ocr]

    t = design_title(pers, F(), name)
    assert t == expected and t.lower() != "pattern"


def test_floss_is_calculated_not_one_skein_each():
    # about 4.2 m of six-strand floss per 1000 full crosses on 14 count with 2 strands
    assert floss_meters(1000, 14, 2) == pytest.approx(4.2, abs=0.2)
    assert floss_meters(2000, 14, 2) == pytest.approx(2 * floss_meters(1000, 14, 2), rel=0.01)
    assert floss_meters(1000, 11, 2) > floss_meters(1000, 14, 2)        # bigger stitches use more
    assert skeins_for(100, 14, 2) == 1
    assert skeins_for(5000, 14, 2) == 3                                 # ~21 m -> 3 skeins of 8 m


LOG = """
PASSED tests/test_order.py::test_source_hash_unchanged
PASSED tests/test_order.py::test_physical_size_labels_truthful
FAILED tests/test_order.py::test_legend_counts_exact - AssertionError
PASSED tests/test_order.py::test_symbol_legibility_at_100_percent
PASSED tests/test_unit.py::test_cm_conversion[a]
"""


def test_quality_report_has_five_gates_measured_and_no_score():
    audit_ = {"status": "THREADORA TECHNICAL FAILURE", "finished_in": [4.5, 5.0], "requested_in": [5, 5],
              "display_fabric": "Aida 14", "fabric_practical": True, "pdf_bytes": 1e6, "pdf_pages": 8,
              "design_audit": {"missing_stitches": 0, "hard_fail": []}, "dmc_colors": 9,
              "review_reasons": [], "tests": {"summary": "1 failed, 4 passed"}}
    rep = report.build(audit_, LOG)
    assert list(rep["gates"]) == ["technical", "size", "design", "dmc", "print"]
    assert rep["score"] is None and rep["tests_executed"] == 5 and rep["tests_passed"] == 4
    assert rep["gates"]["dmc"]["result"] == "FAIL"
    assert rep["gates"]["size"]["result"] == "PASS"
    assert rep["gates"]["size"]["measured"]["A_stitched_design_in"] == [4.5, 5.0]
    assert "test_cm_conversion[a]" in rep["gates"]["technical"]["tests"]
    md = report.to_markdown(rep)
    assert "## Design fidelity" in md and "/10" not in md


def test_quality_report_without_tests_is_not_a_pass():
    rep = report.build({"status": "X", "design_audit": {}}, None)
    assert rep["gates"]["technical"]["result"] == "NOT TESTED"
    assert rep["tests_executed"] == 0


REAL = os.environ.get("THREADORA_REAL_FIXTURES")


@pytest.mark.real_engine
@pytest.mark.skipif(not REAL, reason="set THREADORA_REAL_FIXTURES to a folder with cecilia.png")
def test_real_cecilia_memorial(tmp_path):
    from threadora.pipeline import CONFLICT, REVIEW, STATUSES, VERIFIED, run_order
    src = Path(REAL) / "cecilia.png"
    if not src.exists():
        pytest.skip("cecilia.png not in the fixture folder")
    a = run_order(src, 9, 9, "in", {"phrase": "In Loving Memory", "name": "Cecilia", "year": "2026"},
                  tmp_path / "o", log=lambda *_: None)
    assert a["status"] in STATUSES and a["status"] != "THREADORA TECHNICAL FAILURE"
    assert a["tests"]["returncode"] == 0
    assert a["finished_in"][0] <= 9 + 1e-6 and a["finished_in"][1] <= 9 + 1e-6
    assert a["design_audit"]["lost_white_cells"] == 0
    assert a["quality_report"]["score"] is None
