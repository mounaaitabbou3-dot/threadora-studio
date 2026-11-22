"""Defect-injection tests: each known error class must be DETECTED by the automated
checks (and, where a source-backed fix exists, REPAIRED). Fast, no full engine run.

1 extra crosses · 2 missing stitches · 3 accidental holes · 4 color bleeding ·
5 wrong physical size · 6 duplicated chart stitches · 7 wrong DMC assignment in the PDF ·
8 broken page boundaries · 9 unreadable lettering (see test_regression_engine) ·
10 PDF print defects. Every bug found becomes a permanent test here.
"""
import json

import numpy as np
import pytest

from threadora.dmc import dmc_by_code, to_lab
from threadora.fidelity import CellSource, audit, repair, repair_confetti
from threadora.pdf import PatternPDF
from threadora.pipeline import CONFLICT, FAILURE, REVIEW, VERIFIED, decide_status
from threadora.qa import (chart_cell_colors, master_array, page_print_problems, rebuild_from_pdf,
                          tile_coverage_problems)
from threadora.sizing import FABRICS, fit_grid, physical_size, size_label, size_problems, size_table
from threadora.symbols import assign_symbols

RED, WHITE, BLUE, PINK = "321", "Blanc", "820", "3326"
CELL = 10   # source pixels per stitch in these synthetic sources


def pal_lab(codes):
    by = dmc_by_code()
    return to_lab(np.array([by[c].rgb for c in codes], np.uint8))


def rgb_of(code):
    return np.array(dmc_by_code()[code].rgb, np.uint8)


def source_from_cells(cells, codes):
    """Paint a source image where each cell is exactly one DMC color (or transparent)."""
    rows, cols = cells.shape
    rgb = np.zeros((rows * CELL, cols * CELL, 3), np.uint8)
    opq = np.zeros((rows * CELL, cols * CELL), bool)
    for r in range(rows):
        for c in range(cols):
            v = cells[r, c]
            if v >= 0:
                rgb[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = rgb_of(codes[v])
                opq[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = True
    return rgb, opq


@pytest.fixture
def square():
    """12x12 cells: red square with a 2-cell transparent margin and a white 2x2 centre."""
    m = np.full((12, 12), -1, np.int32)
    m[2:10, 2:10] = 0
    m[5:7, 5:7] = 1
    codes = [RED, WHITE]
    rgb, opq = source_from_cells(m, codes)
    return m, codes, rgb, opq


def cs_for(m, codes, rgb, opq):
    return CellSource(rgb, opq, m.shape[0], m.shape[1], pal_lab(codes))


def test_faithful_matrix_has_no_findings(square):
    m, codes, rgb, opq = square
    res = audit(m, cs_for(m, codes, rgb, opq))
    assert res["pass"] and not res["review"], res
    assert res["extra_stitches"] == res["missing_stitches"] == res["accidental_holes"] == 0


# 1 extra crosses
def test_detects_extra_cross_outside_design(square):
    m, codes, rgb, opq = square
    bad = m.copy(); bad[0, 0] = 0
    res = audit(bad, cs_for(m, codes, rgb, opq))
    assert res["extra_stitches"] == 1 and res["extra_at"] == [[1, 1]]
    assert res["spurious_components"] == 1
    assert not res["pass"]


# 2 missing stitches
def test_detects_missing_stitch_inside_design(square):
    m, codes, rgb, opq = square
    bad = m.copy(); bad[3, 8] = -1
    res = audit(bad, cs_for(m, codes, rgb, opq))
    assert res["missing_stitches"] == 1 and res["missing_at"] == [[4, 9]]
    assert not res["pass"]


# 3 accidental holes vs intentional negative space
def test_detects_and_repairs_accidental_pinhole(square):
    m, codes, rgb, opq = square
    bad = m.copy(); bad[3, 3] = -1
    cs = cs_for(m, codes, rgb, opq)
    res = audit(bad, cs)
    assert res["accidental_holes"] == 1
    fixed, rep = repair(bad, cs)
    assert rep["pinholes_filled"] == 1 and np.array_equal(fixed, m)


def test_intentional_transparent_hole_is_kept():
    m = np.full((12, 12), -1, np.int32)
    m[1:11, 1:11] = 0
    m[4:8, 4:8] = -1                      # transparent window in the source
    codes = [BLUE]
    rgb, opq = source_from_cells(m, codes)
    cs = cs_for(m, codes, rgb, opq)
    res = audit(m, cs)
    assert res["accidental_holes"] == 0 and res["intentional_holes_kept"] == 1 and res["pass"]
    fixed, rep = repair(m, cs)
    assert rep["pinholes_filled"] == 0 and np.array_equal(fixed, m), "negative space must not be filled"
    # and filling it would be reported as lost negative space
    filled = m.copy(); filled[4:8, 4:8] = 0
    res2 = audit(filled, cs)
    assert res2["source_gaps_filled"] == 1 and res2["review"]


def test_protected_letter_counter_is_not_a_hole(square):
    m, codes, rgb, opq = square
    opened = m.copy(); opened[3, 3] = -1
    po = np.zeros(m.shape, bool); po[3, 3] = True
    cs = cs_for(m, codes, rgb, opq)
    res = audit(opened, cs, po)
    assert res["accidental_holes"] == 0 and res["missing_stitches"] == 0
    fixed, rep = repair(opened, cs, po)
    assert rep["pinholes_filled"] == 0


# 4 color bleeding
def test_detects_and_repairs_color_bleeding_at_boundary():
    src = np.full((10, 10), 0, np.int32)
    src[:, 5:] = 1                                         # red | white, hard edge
    codes = [RED, WHITE, PINK]
    rgb, opq = source_from_cells(src, codes)
    # half-cell offset boundary inside column 5 in the source (mixed cells)
    rgb[:, 5 * CELL: 5 * CELL + CELL // 2] = rgb_of(RED)
    bled = src.copy(); bled[:, 5] = 2                       # pink: a mix neither side has
    cs = cs_for(src, codes, rgb, opq)
    res = audit(bled, cs)
    assert res["color_bleed_cells"] == 10 and any("bleeding" in r for r in res["review"])
    fixed, rep = repair(bled, cs)
    assert rep["bleed_recolored"] == 10
    assert set(np.unique(fixed[:, 5])) <= {0, 1}, "repair must use a thread present in the source cell"
    assert audit(fixed, cs)["color_bleed_cells"] == 0


def test_unsupported_confetti_repaired_but_highlight_kept():
    src = np.zeros((9, 9), np.int32)          # all blue
    src[4, 4] = 1                             # one white highlight: intentional detail
    codes = [BLUE, WHITE, RED]
    rgb, opq = source_from_cells(src, codes)
    noisy = src.copy(); noisy[1, 1] = 2      # red speck the source does not have
    cs = cs_for(src, codes, rgb, opq)
    res = audit(noisy, cs)
    assert res["unsupported_confetti"] == 1 and res["unsupported_confetti_at"] == [[2, 2]]
    fixed, n = repair_confetti(noisy, cs)
    assert n == 1 and fixed[1, 1] == 0 and fixed[4, 4] == 1


def test_uniform_region_breaks_detected():
    src = np.zeros((12, 12), np.int32)
    codes = [BLUE, "3842"]                   # a very different second blue
    rgb, opq = source_from_cells(src, codes)
    striped = src.copy(); striped[:, ::2] = 1
    res = audit(striped, cs_for(src, codes, rgb, opq))
    assert res["uniform_break_rate"] > 0.4 and any("uneven" in r for r in res["review"])


# 5 wrong physical size
def test_regression_15in_result_never_labeled_7in():
    """The reported bug: a 170x175 chart printed on Aida 11 (15.45 x 15.91 in) for a 7x7 in order."""
    aida11 = next(f for f in FABRICS if f.name == "Aida 11")
    w, h = physical_size(170, 175, aida11)
    assert round(w, 2) == 15.45 and round(h, 2) == 15.91
    probs = size_problems(170, 175, aida11, 7, 7)
    assert probs and "larger than the requested" in probs[0]
    assert size_label(w, h) == "15.5x15.9in"
    # the engine instead fits a NEW grid for the chosen fabric inside 7x7 in
    plan = fit_grid(1254, 1254, 7, 7, aida11)
    assert (plan.cols, plan.rows) == (77, 77) and size_problems(77, 77, aida11, 7, 7, 1254, 1254) == []


@pytest.mark.parametrize("w,h", [(3, 3), (5, 5), (7, 7), (9, 9), (12, 12), (16, 16), (20, 20), (6, 8), (11.81, 7.87)])
def test_every_fabric_grid_meets_size_invariants(w, h):
    for src in ((1000, 1000), (1600, 900), (700, 1400)):
        for fab in FABRICS:
            p = fit_grid(src[0], src[1], w, h, fab)
            assert size_problems(p.cols, p.rows, fab, w, h, *src) == []


def test_size_table_has_every_fabric_and_one_chosen_row():
    for fab in FABRICS:
        rows = size_table(100, 80, fab)
        assert sum(r["chosen"] for r in rows) == 1, fab.name
        ch = next(r for r in rows if r["chosen"])
        assert ch["w_in"] == pytest.approx(100 / fab.stitches_per_inch)


def test_status_precedence_never_fakes_verified():
    assert decide_status(True, [], True, True, []) == VERIFIED
    assert decide_status(True, [], True, True, ["x"]) == REVIEW
    assert decide_status(None, [], True, True, []) == REVIEW          # tests not run -> not verified
    assert decide_status(True, [], False, True, []) == CONFLICT
    assert decide_status(True, [], True, False, []) == CONFLICT
    assert decide_status(True, ["too big"], True, True, []) == FAILURE
    assert decide_status(False, [], True, True, []) == FAILURE


# ------------------------------------------------------------------ PDF-level defects
@pytest.fixture(scope="module")
def small_pdf(tmp_path_factory):
    """A real PDF from a synthetic 130x70 matrix (forces several chart tiles + overlaps)."""
    d = tmp_path_factory.mktemp("pdf")
    rng = np.random.default_rng(3)
    rows, cols = 70, 130
    m = rng.integers(-1, 4, size=(rows, cols)).astype(np.int32)
    codes = ["310", "321", "Blanc", "797"]
    by = dmc_by_code()
    syms = assign_symbols(len(codes))
    palette = [{"code": c, "name": by[c].name, "rgb": list(by[c].rgb), "hex": by[c].hex, "symbol": s,
                "count": int((m == i).sum()), "skeins": 1} for i, (c, s) in enumerate(zip(codes, syms))]
    master = {"rows": rows, "cols": cols, "palette_codes": codes, "palette": palette,
              "cells": [[codes[v] if v >= 0 else "NO_STITCH" for v in r] for r in m.tolist()]}
    meta = {"project_name": "Test", "title": "Test", "subtitle": "s", "finished_size": "x",
            "requested_size": "x", "total_stitches": int((m >= 0).sum()), "fabric_label": "14-count Aida",
            "strands": 2, "chart_grid": "", "materials": ["a"], "guidance": ["b"],
            "size_table": size_table(cols, rows, FABRICS[1]), "size_caveats": ["c"]}
    pdf = d / "THREADORA_Test.pdf"
    src = np.full((70, 130, 4), 255, np.uint8)
    info = PatternPDF(pdf, meta, m, palette, src).build()
    return pdf, info, master


def test_pdf_fixture_is_clean(small_pdf):
    pdf, info, master = small_pdf
    assert len(info["tiles"]) > 1
    rebuilt, stats = rebuild_from_pdf(pdf, info, master)
    assert np.array_equal(rebuilt, master_array(master))
    assert stats["ghost_wrong"] == stats["ghost_inside_tile"] == stats["ghost_out_of_band"] == 0
    assert chart_cell_colors(pdf, info, master)["n_wrong"] == 0
    assert tile_coverage_problems(info, master["rows"], master["cols"]) == []
    assert page_print_problems(pdf) == []


# 6 duplicated chart stitches
def test_detects_duplicated_and_missing_tile_cells(small_pdf):
    _pdf, info, master = small_pdf
    dup = json.loads(json.dumps(info))
    dup["tiles"][0]["c1"] += 1                     # first tile also claims its neighbour's column
    assert any("more than one chart" in p for p in tile_coverage_problems(dup, master["rows"], master["cols"]))
    gap = json.loads(json.dumps(info))
    gap["tiles"][0]["r1"] -= 1
    assert any("no chart" in p for p in tile_coverage_problems(gap, master["rows"], master["cols"]))


# 7 wrong DMC assignment (chart colors / symbols vs master)
def test_detects_wrong_dmc_color_and_symbol(small_pdf):
    pdf, info, master = small_pdf
    wrong = json.loads(json.dumps(master))
    wrong["palette"][1]["rgb"] = [0, 255, 0]       # legend says green; chart is printed red
    assert chart_cell_colors(pdf, info, wrong)["n_wrong"] > 0
    swapped = json.loads(json.dumps(master))
    swapped["cells"][0] = ["310" if v == "321" else ("321" if v == "310" else v) for v in swapped["cells"][0]]
    rebuilt, _ = rebuild_from_pdf(pdf, info, swapped)
    assert not np.array_equal(rebuilt, master_array(swapped))


# 8 broken page boundaries (shifted tile, overlap references that disagree)
def test_detects_shifted_tile_and_bad_overlap_reference(small_pdf):
    pdf, info, master = small_pdf
    shifted = json.loads(json.dumps(info))
    shifted["tiles"][1]["gx0"] += shifted["tiles"][1]["cell_pt"]    # one column off
    rebuilt, stats = rebuild_from_pdf(pdf, shifted, master)
    assert not np.array_equal(rebuilt, master_array(master)) or stats["outside"] > 0
    t0 = info["tiles"][0]
    changed = json.loads(json.dumps(master))
    r, c = t0["r0"], t0["c1"]                      # first cell of the right-hand neighbour = ghost on tile 0
    old = changed["cells"][r][c]
    changed["cells"][r][c] = "310" if old != "310" else "321"
    _, stats2 = rebuild_from_pdf(pdf, info, changed)
    assert stats2["ghost_wrong"] >= 1


# 10 PDF print defects
def test_detects_blank_and_non_a4_pages(small_pdf, tmp_path):
    import pymupdf
    pdf, _info, _master = small_pdf
    doc = pymupdf.open(pdf)
    doc.new_page(width=595.2756, height=841.8898)          # blank page
    doc.new_page(width=612, height=792)                      # US Letter
    p2 = tmp_path / "bad.pdf"
    doc.save(p2)
    probs = page_print_problems(p2)
    assert any("blank" in p for p in probs) and any("not A4" in p for p in probs)


def test_lost_small_detail_detected_and_restored():
    """Regression (found by visual inspection): teddy-bear eye catchlights vanished at 5x5 in
    while every gate passed. A one-stitch, high-contrast source detail must be kept."""
    src = np.zeros((12, 12), np.int32)
    src[5, 5] = 1                                  # white catchlight in a black eye
    codes = ["310", WHITE]
    rgb, opq = source_from_cells(src, codes)
    cs = cs_for(src, codes, rgb, opq)
    lost = np.zeros_like(src)                      # all black: the highlight was dropped
    res = audit(lost, cs)
    assert res["lost_details"] == 1 and res["lost_details_at"] == [[6, 6]]
    assert any("small source detail" in r for r in res["review"])
    fixed, rep = repair(lost, cs)
    # restored from the source: either as an unsupported cell color or as a lost detail
    assert fixed[5, 5] == 1 and rep["details_restored"] + rep["bleed_recolored"] == 1
    assert audit(fixed, cs)["lost_details"] == 0


def test_low_contrast_texture_is_not_a_detail():
    src = np.zeros((12, 12), np.int32)
    src[5, 5] = 1
    codes = ["3346", "3347"]                       # two close greens: texture, not a detail
    rgb, opq = source_from_cells(src, codes)
    res = audit(np.zeros_like(src), cs_for(src, codes, rgb, opq))
    assert res["lost_details"] == 0
