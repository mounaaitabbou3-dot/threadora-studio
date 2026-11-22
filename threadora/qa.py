"""Independent checks on generated files (used by pytest and the pipeline)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

A4_PT = (595.2756, 841.8898)


def load_master(path) -> dict:
    return json.loads(Path(path).read_text())


def master_array(master: dict) -> np.ndarray:
    codes = master["palette_codes"]
    idx = {c: i for i, c in enumerate(codes)}
    return np.array([[idx[v] if v != "NO_STITCH" else -1 for v in row] for row in master["cells"]],
                    dtype=np.int32)


def rebuild_from_pdf(pdf_path, chart_info: dict, master: dict) -> tuple[np.ndarray, dict]:
    """Read every chart page of the PDF, map each primary (black/white ink) symbol to
    its cell by position, translate symbol -> DMC via the legend, and rebuild the
    matrix. Ghost (grey) overlap symbols are counted separately and must not add stitches."""
    import pymupdf
    sym2idx = {p["symbol"]: i for i, p in enumerate(master["palette"])}
    rows, cols = master["rows"], master["cols"]
    out = np.full((rows, cols), -2, dtype=np.int32)   # -2 = not covered by any tile
    stats = {"ghost_symbols": 0, "double_assigned": 0, "foreign_symbols": 0, "outside": 0,
             "ghost_wrong": 0, "ghost_inside_tile": 0, "ghost_out_of_band": 0}
    ref = master_array(master)
    doc = pymupdf.open(pdf_path)
    for t in chart_info["tiles"]:
        page = doc[t["page"] - 1]
        ph = page.rect.height
        cs = t["cell_pt"]
        out[t["r0"]:t["r1"], t["c0"]:t["c1"]] = -1
        raw = page.get_text("rawdict")
        for b in raw["blocks"]:
            for ln in b.get("lines", []):
                for sp in ln["spans"]:
                    if "TD-Sym" not in sp["font"] and "DejaVuSans-Bold" not in sp["font"]:
                        continue
                    if abs(sp["size"] - cs * chart_info["symbol_scale"]) > 0.05:
                        continue  # labels or legend text, not chart symbols
                    color = sp["color"]
                    rgb = ((color >> 16) & 255, (color >> 8) & 255, color & 255)
                    is_ghost = rgb not in ((0, 0, 0), (255, 255, 255))
                    for ch in sp["chars"]:
                        x0, y0, x1, y1 = ch["bbox"]
                        cx = (x0 + x1) / 2
                        oy = ch["origin"][1]  # baseline, top-down coords
                        base_up = ph - oy     # reportlab coords
                        col = int(np.floor((cx - t["gx0"]) / cs)) + t["c0"]
                        row = int(np.floor((t["gy0"] - base_up) / cs)) + t["r0"]
                        if is_ghost:
                            # overlap reference: outside its own tile, within the band,
                            # and identical to the neighbouring chart's stitch
                            stats["ghost_symbols"] += 1
                            if t["r0"] <= row < t["r1"] and t["c0"] <= col < t["c1"]:
                                stats["ghost_inside_tile"] += 1
                            elif not (t["r0"] - 2 <= row < t["r1"] + 2 and t["c0"] - 2 <= col < t["c1"] + 2
                                      and 0 <= row < rows and 0 <= col < cols):
                                stats["ghost_out_of_band"] += 1
                            elif sym2idx.get(ch["c"], -9) != ref[row, col]:
                                stats["ghost_wrong"] += 1
                            continue
                        if not (t["r0"] <= row < t["r1"] and t["c0"] <= col < t["c1"]):
                            stats["outside"] += 1
                            continue
                        if ch["c"] not in sym2idx:
                            stats["foreign_symbols"] += 1
                            continue
                        if out[row, col] != -1:
                            stats["double_assigned"] += 1
                        out[row, col] = sym2idx[ch["c"]]
    return out, stats


def labels_collide(boxes: list) -> list:
    bad = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if a[0] != b[0]:
                continue
            if a[2] < b[4] - 0.2 and b[2] < a[4] - 0.2 and a[3] < b[5] - 0.2 and b[3] < a[5] - 0.2:
                bad.append((a[:2], b[:2]))
    return bad


def chart_cell_colors(pdf_path, chart_info: dict, master: dict, dpi: int = 200) -> dict:
    """Render every chart page and read each PRIMARY cell's background color near its
    top-left corner (away from the symbol and grid lines). Returns mismatches against
    the master: wrong DMC color, ink in NO_STITCH cells."""
    import pymupdf
    m = master_array(master)
    pal = np.array([p["rgb"] for p in master["palette"]], float)
    doc = pymupdf.open(pdf_path)
    k = dpi / 72
    wrong, checked = [], 0
    for t in chart_info["tiles"]:
        page = doc[t["page"] - 1]
        pix = page.get_pixmap(dpi=dpi)
        a = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[..., :3].astype(float)
        ph = page.rect.height
        cs = t["cell_pt"]
        for r in range(t["r0"], t["r1"]):
            ytop = (ph - (t["gy0"] - (r - t["r0"]) * cs)) * k
            y0, y1 = int(ytop + 0.12 * cs * k), int(ytop + 0.21 * cs * k) + 1
            for c in range(t["c0"], t["c1"]):
                x = (t["gx0"] + (c - t["c0"]) * cs) * k
                x0, x1 = int(x + 0.12 * cs * k), int(x + 0.21 * cs * k) + 1
                got = a[y0:y1, x0:x1].reshape(-1, 3).mean(0)
                want = pal[m[r, c]] if m[r, c] >= 0 else np.array([255.0, 255, 255])
                checked += 1
                if np.abs(got - want).max() > 14:
                    wrong.append({"row": r + 1, "col": c + 1, "want": want.tolist(),
                                  "got": [round(v) for v in got]})
    return {"checked": checked, "wrong": wrong[:50], "n_wrong": len(wrong)}


def cover_size_text(pdf_path) -> str:
    import pymupdf
    return pymupdf.open(pdf_path)[0].get_text()


def tile_coverage_problems(chart_info: dict, rows: int, cols: int) -> list[str]:
    """Every coordinate must belong to exactly one primary chart tile."""
    cover = np.zeros((rows, cols), int)
    for t in chart_info["tiles"]:
        cover[max(0, t["r0"]):t["r1"], max(0, t["c0"]):t["c1"]] += 1
    out = []
    if (cover == 0).any():
        out.append(f"{int((cover == 0).sum())} cells belong to no chart")
    if (cover > 1).any():
        out.append(f"{int((cover > 1).sum())} cells are owned by more than one chart (duplicated stitches)")
    return out


def page_print_problems(pdf_path, max_mb: float = 20.0) -> list[str]:
    """Render every page: A4, not blank, nothing in the outer 6 pt band, text inside the page."""
    import pymupdf
    out = []
    if Path(pdf_path).stat().st_size > max_mb * 1024 * 1024:
        out.append("PDF larger than 20 MB")
    doc = pymupdf.open(pdf_path)
    for i, page in enumerate(doc):
        if abs(page.rect.width - A4_PT[0]) >= 0.5 or abs(page.rect.height - A4_PT[1]) >= 0.5:
            out.append(f"page {i + 1} is not A4")
        pix = page.get_pixmap(dpi=100)
        a = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[..., :3]
        if a.std() <= 5:
            out.append(f"page {i + 1} is blank")
            continue
        band = int(6 / 72 * 100)
        edges = np.concatenate([a[:band].reshape(-1, 3), a[-band:].reshape(-1, 3),
                                a[:, :band].reshape(-1, 3), a[:, -band:].reshape(-1, 3)])
        if (edges.min(axis=1) < 245).mean() >= 0.005:
            out.append(f"page {i + 1} content touches the page edge (clipping)")
        for b in page.get_text("dict")["blocks"]:
            x0, y0, x1, y1 = b["bbox"]
            if x0 < 0 or y0 < 0 or x1 > page.rect.width or y1 > page.rect.height:
                out.append(f"page {i + 1} text outside the page")
                break
    return out
