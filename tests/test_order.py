"""Verification of one generated order (run with --order-dir and --source)."""
import hashlib
import json

import numpy as np
import pytest

from threadora.dmc import ciede2000_matrix, dmc_by_code, to_lab
from threadora.qa import A4_PT, labels_collide, master_array, rebuild_from_pdf
from threadora.symbols import contrast_ratio, symbol_ink


def test_source_hash_unchanged(source_path, master):
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == master["source_sha256"]


def test_matrix_dimensions_and_physical_size(master):
    rows, cols = master["rows"], master["cols"]
    assert len(master["cells"]) == rows and all(len(r) == cols for r in master["cells"])
    spi = master["fabric"]["stitches_per_inch"]
    assert master["physical_in"][0] == pytest.approx(cols / spi, abs=1e-3)
    assert master["physical_in"][1] == pytest.approx(rows / spi, abs=1e-3)
    # aspect ratio of the content box preserved within one stitch
    x0, y0, x1, y1 = master["content_bbox"]
    assert abs(cols / rows - (x1 - x0) / (y1 - y0)) <= 1 / min(cols, rows) + 1e-9


def test_dmc_codes_names_valid_and_all_used(master):
    by = dmc_by_code()
    m = master_array(master)
    for i, p in enumerate(master["palette"]):
        f = by[p["code"]]
        assert f.name == p["name"] and list(f.rgb) == p["rgb"]
        assert (m == i).any(), f"unused color {p['code']}"
    assert len(set(master["palette_codes"])) == len(master["palette_codes"])


def test_legend_counts_exact(master):
    m = master_array(master)
    for i, p in enumerate(master["palette"]):
        assert int((m == i).sum()) == p["count"]
    assert sum(p["count"] for p in master["palette"]) == int((m >= 0).sum())


def test_symbols_one_to_one_and_contrast(master):
    syms = [p["symbol"] for p in master["palette"]]
    assert len(set(syms)) == len(syms)
    for p in master["palette"]:
        assert contrast_ratio(p["rgb"], symbol_ink(p["rgb"])) >= 4.5


def test_palette_colors_distinguishable(master):
    lab = to_lab(np.array([p["rgb"] for p in master["palette"]], np.uint8))
    d = ciede2000_matrix(lab, lab)
    np.fill_diagonal(d, 99)
    assert d.min() >= 3.0


def test_pdf_rebuilds_master_exactly(pdf_path, chart_info, master):
    rebuilt, stats = rebuild_from_pdf(pdf_path, chart_info, master)
    m = master_array(master)
    assert (rebuilt != -2).all(), "some cells belong to no chart tile"
    assert stats["double_assigned"] == 0 and stats["foreign_symbols"] == 0 and stats["outside"] == 0
    assert np.array_equal(rebuilt, m), f"{int((rebuilt != m).sum())} cells differ"
    assert stats["ghost_symbols"] > 0 or len(chart_info["tiles"]) == 1


def test_tiles_each_cell_exactly_once(chart_info, master):
    cover = np.zeros((master["rows"], master["cols"]), int)
    for t in chart_info["tiles"]:
        cover[t["r0"]:t["r1"], t["c0"]:t["c1"]] += 1
    assert (cover == 1).all()


def test_pages_order_and_titles(pdf_path, chart_info):
    import pymupdf
    doc = pymupdf.open(pdf_path)
    assert doc.page_count == chart_info["pages"]
    t = chart_info["titles"]
    assert t[:4] == ["Pattern cover", "Materials, stitching guide and sizes", "Thread legend (DMC)",
                     "Stitched result overview — reconstructed from the chart"]
    assert t[4:] == [f"Chart {x['id']}" for x in chart_info["tiles"]]
    for i, title in enumerate(t):
        assert title in doc[i].get_text(), f"page {i + 1} missing title {title}"
    assert [x["page"] for x in chart_info["tiles"]] == list(range(5, 5 + len(chart_info["tiles"])))


def test_no_label_collisions_and_labels_correct(chart_info, pdf_path):
    import pymupdf
    doc = pymupdf.open(pdf_path)
    for t in chart_info["tiles"]:
        boxes = chart_info["label_boxes"][str(t["page"])]
        assert not labels_collide(boxes)
        cols = [b[1] for b in boxes if b[0] == "col"]
        rows = [b[1] for b in boxes if b[0] == "row"]
        assert all(t["c0"] < n <= t["c1"] for n in cols) and all(t["r0"] < n <= t["r1"] for n in rows)
        assert all(n % 10 == 0 for n in range(t["c0"] + 1, t["c1"] + 1) if n % 10 == 0 and n in cols)
        assert {n for n in range(t["c0"] + 1, t["c1"] + 1) if n % 10 == 0} <= set(cols)
        assert {n for n in range(t["r0"] + 1, t["r1"] + 1) if n % 10 == 0} <= set(rows)
        text = doc[t["page"] - 1].get_text()
        for n in cols + rows:
            assert str(n) in text


def test_render_every_page_a4_not_blank_not_clipped(pdf_path):
    import pymupdf
    doc = pymupdf.open(pdf_path)
    assert pdf_path.stat().st_size <= 20 * 1024 * 1024
    for i, page in enumerate(doc):
        assert abs(page.rect.width - A4_PT[0]) < 0.5 and abs(page.rect.height - A4_PT[1]) < 0.5
        pix = page.get_pixmap(dpi=100)
        a = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[..., :3]
        assert a.std() > 5, f"page {i + 1} looks blank"
        # nothing drawn in the outer 6 pt band (clipping guard)
        band = int(6 / 72 * 100)
        edges = np.concatenate([a[:band].reshape(-1, 3), a[-band:].reshape(-1, 3),
                                a[:, :band].reshape(-1, 3), a[:, -band:].reshape(-1, 3)])
        assert (edges.min(axis=1) < 245).mean() < 0.005, f"page {i + 1} content touches the page edge"
        for b in page.get_text("dict")["blocks"]:
            x0, y0, x1, y1 = b["bbox"]
            assert x0 >= 0 and y0 >= 0 and x1 <= page.rect.width and y1 <= page.rect.height


def test_fonts_embedded(pdf_path):
    import pymupdf
    doc = pymupdf.open(pdf_path)
    seen = 0
    for i in range(doc.page_count):
        for f in doc.get_page_fonts(i):
            xref, ext = f[0], f[1]
            assert ext not in ("n/a", ""), f"font {f[3]} not embedded"
            name, ext2, _t, content = doc.extract_font(xref)
            assert content and len(content) > 100
            seen += 1
    assert seen > 0


def test_symbol_legibility_at_100_percent(pdf_path, chart_info, master):
    """At 100% print scale the chart symbols must be >= 5.5 pt and cells >= 3.1 mm,
    and rendered at 300 dpi each cell's symbol must contrast with its cell."""
    import pymupdf
    cs = chart_info["cell_pt"]
    assert cs * 25.4 / 72 >= 3.1
    assert cs * chart_info["symbol_scale"] >= 5.5
    doc = pymupdf.open(pdf_path)
    m = master_array(master)
    k = 300 / 72
    checked = 0
    rng = np.random.default_rng(0)
    for t in chart_info["tiles"]:
        page = doc[t["page"] - 1]
        pix = page.get_pixmap(dpi=300)
        a = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[..., :3].astype(float)
        ph = page.rect.height
        cells = [(r, c) for r in range(t["r0"], t["r1"]) for c in range(t["c0"], t["c1"])]
        for i in rng.choice(len(cells), size=min(150, len(cells)), replace=False):
            r, c = cells[i]
            x = (t["gx0"] + (c - t["c0"]) * cs) * k
            ytop = (ph - (t["gy0"] - (r - t["r0"]) * cs)) * k
            lum = a[int(ytop) + 6:int(ytop + cs * k) - 6, int(x) + 6:int(x + cs * k) - 6].mean(axis=2)
            if m[r, c] < 0:     # NO_STITCH: blank, no symbol ink
                assert lum.min() > 200, f"ink in empty cell r{r + 1} c{c + 1}"
                continue
            checked += 1
            assert lum.max() - lum.min() > 80, f"weak symbol contrast r{r + 1} c{c + 1}"
    assert checked > 0


def test_fidelity_gates_recorded(order_dir):
    audit_p = order_dir / "private" / "audit.json"
    if not audit_p.exists():
        pytest.skip("audit written after tests")
    a = json.loads(audit_p.read_text())
    assert "fidelity_metrics" in a
