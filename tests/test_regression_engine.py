"""End-to-end regression: the REAL engine on every synthetic design type and every
supported size, with the full order test suite run on each generated PDF.

Marked real_engine (minutes, not seconds). Run alone with:
    python -m pytest tests/test_regression_engine.py -m real_engine
"""
import json

import pytest

from tests.designs import DESIGNS, save
from threadora.pipeline import CONFLICT, FAILURE, REVIEW, STATUSES, VERIFIED, run_order
from threadora.sizing import FABRICS, physical_size, size_label, to_inches

pytestmark = pytest.mark.real_engine

PRESETS = [(3, 3, "in"), (5, 5, "in"), (7, 7, "in"), (9, 9, "in"), (12, 12, "in"), (16, 16, "in"),
           (20, 20, "in"), (6, 8, "in"), (20, 30, "cm")]


def _run(tmp_path, design, w, h, units, fabric=None):
    src = save(design, tmp_path)
    out = tmp_path / f"out_{w}x{h}{units}_{fabric or 'auto'}".replace(" ", "_")
    audit = run_order(src, w, h, units, DESIGNS[design][1], out, display_fabric=fabric,
                      log=lambda *_: None, run_tests=True)
    return src, out, audit


def check_order(out, audit, w, h, units):
    """Invariants for every order, whatever its status."""
    assert audit["status"] in STATUSES
    if audit["status"] == FAILURE:
        pytest.fail(f"technical failure: {audit.get('blocker')}")
    if audit["status"] == CONFLICT and not audit.get("pdf"):
        return None
    req_w, req_h = to_inches(w, h, units)
    master = json.loads((out / "private" / "master_matrix.json").read_text())
    fab = next(f for f in FABRICS if f.name == master["fabric"]["name"])
    fw, fh = physical_size(master["cols"], master["rows"], fab)
    # exact physical size, never bigger than ordered, label = truth
    assert fw <= req_w + 1e-6 and fh <= req_h + 1e-6, (fw, fh, req_w, req_h)
    assert audit["finished_in"] == pytest.approx([fw, fh])
    assert audit["pdf"].endswith(f"_{size_label(fw, fh)}.pdf")
    # the generated PDF passed every order test (matrix rebuild, colors, grid, print checks)
    assert audit["tests"]["returncode"] == 0, audit["tests"]
    # design audit: no stitches without source, no lost design cells
    da = audit["design_audit"]
    assert da["extra_stitches"] == 0 and da["missing_stitches"] == 0 and da["spurious_components"] == 0
    # status honesty
    if audit["status"] == VERIFIED:
        assert not audit["review_reasons"] and audit["fabric_practical"] and not audit["size_problems"]
    if audit["review_reasons"]:
        assert audit["status"] in (REVIEW, CONFLICT)
    return master


@pytest.mark.parametrize("design", sorted(DESIGNS))
def test_every_design_type_at_5in(tmp_path, design):
    _src, out, audit = _run(tmp_path, design, 5, 5, "in")
    check_order(out, audit, 5, 5, "in")


@pytest.mark.parametrize("w,h,units", PRESETS)
def test_every_size_memorial_lettering(tmp_path, w, h, units):
    _src, out, audit = _run(tmp_path, "memorial_lettering", w, h, units)
    master = check_order(out, audit, w, h, units)
    if w <= 3 and units == "in":
        # 3 in memorial lettering cannot be read on a practical fabric: never VERIFIED
        assert audit["status"] in (CONFLICT, REVIEW)
    if master is not None:
        assert master["requested_in"] == pytest.approx(list(to_inches(w, h, units)))


@pytest.mark.parametrize("w,h,units", [(7, 7, "in"), (16, 16, "in"), (6, 8, "in")])
def test_every_size_transparent_design(tmp_path, w, h, units):
    _src, out, audit = _run(tmp_path, "transparent_wreath", w, h, units)
    master = check_order(out, audit, w, h, units)
    assert master is not None
    # transparent centre stays NO_STITCH (never filled with an invented color)
    r, c = master["rows"] // 2, master["cols"] // 2
    assert master["cells"][r][c] == "NO_STITCH"
    assert audit["design_audit"]["source_gaps_filled"] == 0


def test_regression_forced_fabric_keeps_requested_size(tmp_path):
    """Bug: choosing Aida 11 for a 7x7 in order reprinted a 25-count chart at 15.45x15.91 in
    and still named it 7x7. The chosen fabric now gets its own grid inside 7x7 in, or the
    order is a SIZE-FIDELITY CONFLICT with alternatives; it is never silently enlarged."""
    _src, out, audit = _run(tmp_path, "teddy_bear", 7, 7, "in", fabric="Aida 11")
    if audit["status"] == CONFLICT and not audit.get("pdf"):
        assert "Aida" in audit["blocker"]
        return
    master = check_order(out, audit, 7, 7, "in")
    assert master["fabric"]["name"] == "Aida 11"
    assert master["cols"] <= 77 and master["rows"] <= 77


def test_unreadable_lettering_is_never_verified(tmp_path):
    """Fine typography at 3x3 in: the letters cannot survive, so no VERIFIED status."""
    _src, out, audit = _run(tmp_path, "fine_typography", 3, 3, "in")
    assert audit["status"] in (CONFLICT, REVIEW), audit["status"]
    if audit.get("pdf"):
        check_order(out, audit, 3, 3, "in")


def test_opaque_white_details_are_kept(tmp_path):
    """Opaque image on white: the white snowman INSIDE the design must stay stitched."""
    _src, out, audit = _run(tmp_path, "opaque_white_background", 7, 7, "in")
    master = check_order(out, audit, 7, 7, "in")
    assert master is not None
    codes = {p["code"]: p for p in master["palette"]}
    whiteish = [c for c, p in codes.items() if min(p["rgb"]) >= 225]
    assert whiteish, "white detail inside the design was dropped"
    r, c = int(master["rows"] * 0.65), master["cols"] // 2      # snowman body
    assert master["cells"][r][c] in whiteish
