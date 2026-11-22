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
    stats = {"ghost_symbols": 0, "double_assigned": 0, "foreign_symbols": 0, "outside": 0}
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
                            stats["ghost_symbols"] += 1
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
