"""A4 THREADORA PDF: cover, materials, legend, matrix overview, color+symbol charts.
Everything chart-related is drawn from the master matrix only."""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image
from reportlab.lib.colors import Color, black, white
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from .symbols import FONT_DIR, SYMBOL_FONT_FILE, symbol_ink

PAGE_W, PAGE_H = A4
MARGIN = 28.0
HEADER_H = 46.0
FOOTER_H = 18.0
LABEL_W = 16.0
GHOST = 2                    # overlap reference cells shown from neighbouring charts
MIN_CELL_PT = 9.0            # 3.2 mm: smallest cell we accept for print readability
SYMBOL_SCALE = 0.64
GHOST_INK = (0.55, 0.55, 0.55)

_FONTS_DONE = False


def register_fonts():
    global _FONTS_DONE
    if _FONTS_DONE:
        return
    pdfmetrics.registerFont(TTFont("TD-Sans", str(FONT_DIR / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("TD-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("TD-Sym", str(SYMBOL_FONT_FILE)))
    pdfmetrics.registerFont(TTFont("TD-Serif", str(FONT_DIR / "DejaVuSerif-Bold.ttf")))
    _FONTS_DONE = True


@dataclass
class Tile:
    id: str
    tc: int   # tile column index
    tr: int   # tile row index
    c0: int
    c1: int   # exclusive
    r0: int
    r1: int   # exclusive
    page: int = 0          # 1-based PDF page
    cell_pt: float = 0.0
    gx0: float = 0.0       # PDF x of global column c0 left edge
    gy0: float = 0.0       # PDF y of global row r0 TOP edge (reportlab coords, y up)


def _split(n: int, parts: int) -> list[tuple[int, int]]:
    base, extra = divmod(n, parts)
    out, s = [], 0
    for i in range(parts):
        e = s + base + (1 if i < extra else 0)
        out.append((s, e))
        s = e
    return out


def plan_tiles(cols: int, rows: int) -> tuple[list[Tile], float]:
    """Fewest chart pages whose cells stay >= MIN_CELL_PT; ties -> larger cells."""
    avail_w = PAGE_W - 2 * MARGIN - LABEL_W
    avail_h = PAGE_H - 2 * MARGIN - HEADER_H - FOOTER_H - LABEL_W
    best = None
    for nx in range(1, cols + 1):
        for ny in range(1, rows + 1):
            tw = math.ceil(cols / nx) + (2 * GHOST if nx > 1 else 0)
            th = math.ceil(rows / ny) + (2 * GHOST if ny > 1 else 0)
            cs = min(avail_w / tw, avail_h / th, 14.0)
            if cs < MIN_CELL_PT:
                continue
            key = (nx * ny, -cs)
            if best is None or key < best[0]:
                best = (key, nx, ny, cs)
            break
    _, nx, ny, cs = best
    tiles = []
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for tr, (r0, r1) in enumerate(_split(rows, ny)):
        for tc, (c0, c1) in enumerate(_split(cols, nx)):
            tiles.append(Tile(f"{letters[tr]}{tc + 1}", tc, tr, c0, c1, r0, r1, cell_pt=cs))
    return tiles, cs


def wrap(text: str, font: str, size: float, maxw: float) -> list[str]:
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if pdfmetrics.stringWidth(trial, font, size) <= maxw or not cur:
            cur = trial
        else:
            lines.append(cur); cur = w
    lines.append(cur)
    return lines


def _rgb(c):
    return Color(c[0] / 255, c[1] / 255, c[2] / 255)


class PatternPDF:
    def __init__(self, path, meta: dict, matrix: np.ndarray, palette: list[dict],
                 source_rgba: np.ndarray):
        register_fonts()
        self.path = str(path)
        self.meta = meta
        self.m = matrix
        self.pal = palette               # [{code,name,rgb,symbol,count,skeins}]
        self.src = source_rgba
        self.rows, self.cols = matrix.shape
        self.tiles, self.cell = plan_tiles(self.cols, self.rows)
        self.page_titles: list[str] = []
        self.label_boxes: dict[int, list] = {}

    # ---------------------------------------------------------- helpers
    def _header(self, c, title, sub=""):
        c.setFillColor(Color(0.42, 0.06, 0.10))
        c.roundRect(MARGIN - 10, PAGE_H - 34, PAGE_W - 2 * MARGIN + 20, 24, 5, stroke=0, fill=1)
        c.setFillColor(white)
        c.setFont("TD-Serif", 12)
        c.drawString(MARGIN, PAGE_H - 26, "THREADORA CROSS STITCH")
        c.setFont("TD-Sans", 8.5)
        c.drawRightString(PAGE_W - MARGIN, PAGE_H - 26, self.meta["project_name"])
        if self.meta.get("review_banner"):
            c.setFillColor(Color(1, 0.85, 0.2))
            c.setFont("TD-Bold", 8.5)
            c.drawCentredString(PAGE_W / 2 + 20, PAGE_H - 26, self.meta["review_banner"])
        c.setFillColor(black)
        c.setFont("TD-Bold", 12)
        c.drawString(MARGIN, PAGE_H - 50, title)
        if sub:
            c.setFont("TD-Sans", 8)
            c.drawRightString(PAGE_W - MARGIN, PAGE_H - 50, sub)
        self.page_titles.append(title)

    def _footer(self, c, n, total):
        c.setFont("TD-Sans", 7)
        c.setFillColor(Color(0.35, 0.35, 0.35))
        c.drawString(MARGIN, 16, f"{self.meta['project_name']} · {self.cols} × {self.rows} stitches · "
                                 f"{len(self.pal)} DMC colors")
        c.drawRightString(PAGE_W - MARGIN, 16, f"Page {n} of {total}")
        c.setFillColor(black)

    def _img(self, arr: np.ndarray) -> ImageReader:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG", optimize=True)
        buf.seek(0)
        return ImageReader(buf)

    # ---------------------------------------------------------- pages
    def cover(self, c):
        self._header(c, "Pattern cover")
        mt = self.meta
        c.setFont("TD-Serif", 22)
        c.drawCentredString(PAGE_W / 2, PAGE_H - 92, mt["title"])
        c.setFont("TD-Sans", 10.5)
        c.drawCentredString(PAGE_W / 2, PAGE_H - 110, mt["subtitle"])
        # original image, composited on white, labeled as reference
        rgba = self.src
        a = rgba[..., 3:4] / 255.0
        comp = (rgba[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
        im = Image.fromarray(comp)
        im.thumbnail((900, 900))
        side = 330
        x = (PAGE_W - side) / 2
        y = PAGE_H - 130 - side
        c.drawImage(self._img(np.array(im)), x, y, side, side, preserveAspectRatio=True, anchor="c")
        c.setStrokeColor(Color(0.7, 0.7, 0.7)); c.setLineWidth(0.6)
        c.rect(x - 4, y - 4, side + 8, side + 8)
        c.setFont("TD-Bold", 8)
        c.drawCentredString(PAGE_W / 2, y - 16, "ORIGINAL ARTWORK — reference image only, not the stitched result")
        c.setFont("TD-Sans", 7.5)
        c.drawCentredString(PAGE_W / 2, y - 27, "The stitched result reconstructed from the chart is shown on the overview page.")
        rows = [("Stitched design size", mt["finished_size"] + " — exact, from the stitch count"),
                ("Requested size", mt["requested_size"]),
                ("Stitch count", f"{self.cols} wide × {self.rows} high ({self.cols * self.rows:,} cells, "
                                 f"{mt['total_stitches']:,} full cross stitches)"),
                ("Fabric", mt["fabric_label"]),
                ("Threads", f"{len(self.pal)} DMC six-strand floss colors, {mt['strands']} strand(s)"),
                ("Stitches used", "Full cross stitch only (no backstitch, no fractional stitches)"),
                ("Charts", f"{len(self.tiles)} color + symbol chart pages ({mt['chart_grid']})")]
        ty = y - 56
        vx = MARGIN + 150
        maxw = PAGE_W - MARGIN - vx
        for k, v in rows:
            c.setFont("TD-Bold", 9); c.drawString(MARGIN + 20, ty, k)
            c.setFont("TD-Sans", 9)
            for ln in wrap(v, "TD-Sans", 9, maxw):
                c.drawString(vx, ty, ln)
                ty -= 12
            ty -= 3

    def materials(self, c):
        self._header(c, "Materials, stitching guide and sizes")
        mt = self.meta
        y = PAGE_H - 74
        def para(title, lines):
            nonlocal y
            c.setFont("TD-Bold", 10); c.drawString(MARGIN, y, title); y -= 14
            c.setFont("TD-Sans", 8.6)
            for ln in lines:
                for part in wrap(ln, "TD-Sans", 8.6, PAGE_W - 2 * MARGIN - 8):
                    c.drawString(MARGIN + 8, y, part); y -= 11.5
            y -= 6
        para("Materials", mt["materials"])
        para("Stitching guide", mt["guidance"])
        c.setFont("TD-Bold", 10); c.drawString(MARGIN, y, "Same chart on other fabrics"); y -= 14
        c.setFont("TD-Bold", 8.2)
        c.drawString(MARGIN + 8, y, "Fabric"); c.drawString(MARGIN + 220, y, "Finished size (in)")
        c.drawString(MARGIN + 340, y, "Finished size (cm)"); y -= 11
        c.setFont("TD-Sans", 8.2)
        for r in mt["size_table"]:
            mark = "  ← this pattern" if r.get("chosen") else ""
            c.drawString(MARGIN + 8, y, r["fabric"] + mark)
            c.drawString(MARGIN + 220, y, f"{r['w_in']:.2f} × {r['h_in']:.2f}")
            c.drawString(MARGIN + 340, y, f"{r['w_cm']:.1f} × {r['h_cm']:.1f}")
            y -= 10.5
        y -= 4
        c.setFont("TD-Sans", 7.4)
        for ln in mt["size_caveats"]:
            c.drawString(MARGIN + 8, y, ln); y -= 9.5

    def legend(self, c):
        self._header(c, "Thread legend (DMC)", "Counts are exact full cross stitches from the chart")
        y = PAGE_H - 78
        cs = 16
        c.setFont("TD-Bold", 8.5)
        heads = [(MARGIN, "Symbol"), (MARGIN + 50, "DMC"), (MARGIN + 100, "Color name"),
                 (MARGIN + 290, "Stitches"), (MARGIN + 345, "Floss (m)"), (MARGIN + 400, "Skeins"),
                 (MARGIN + 440, "Swatch")]
        for x, t in heads:
            c.drawString(x, y, t)
        y -= 8
        for p in self.pal:
            y -= cs + 3
            bg = p["rgb"]
            c.setFillColor(_rgb(bg)); c.setStrokeColor(Color(0.3, 0.3, 0.3)); c.setLineWidth(0.5)
            c.rect(MARGIN + 6, y, cs, cs, fill=1, stroke=1)
            c.setFillColor(_rgb(symbol_ink(bg)))
            c.setFont("TD-Sym", cs * SYMBOL_SCALE)
            c.drawCentredString(MARGIN + 6 + cs / 2, y + cs * 0.27, p["symbol"])
            c.setFillColor(black); c.setFont("TD-Bold", 9)
            c.drawString(MARGIN + 50, y + 4.5, p["code"])
            c.setFont("TD-Sans", 9)
            c.drawString(MARGIN + 100, y + 4.5, p["name"])
            c.drawRightString(MARGIN + 330, y + 4.5, f"{p['count']:,}")
            c.drawRightString(MARGIN + 385, y + 4.5, f"{p['meters']:.1f}" if "meters" in p else "—")
            c.drawRightString(MARGIN + 428, y + 4.5, str(p["skeins"]))
            c.setFillColor(_rgb(bg)); c.rect(MARGIN + 440, y, 70, cs, fill=1, stroke=1)
            c.setFillColor(black)
        y -= 20
        c.setFont("TD-Bold", 9)
        tot = sum(p["count"] for p in self.pal)
        c.drawString(MARGIN + 100, y, "Total full cross stitches")
        c.drawRightString(MARGIN + 330, y, f"{tot:,}")
        if all("meters" in p for p in self.pal):
            c.drawRightString(MARGIN + 385, y, f"{sum(p['meters'] for p in self.pal):.1f}")
            c.drawRightString(MARGIN + 428, y, str(sum(p["skeins"] for p in self.pal)))
        y -= 14
        if self.meta.get("floss_note"):
            c.setFont("TD-Sans", 7.5)
            for ln in wrap(self.meta["floss_note"], "TD-Sans", 7.5, PAGE_W - 2 * MARGIN):
                c.drawString(MARGIN, y, ln)
                y -= 10
        c.setFont("TD-Sans", 7.5)
        c.drawString(MARGIN, y, "Blank chart cells = no stitch (fabric shows). Screen and print colors are approximate; "
                                "always stitch by DMC number.")
        y -= 10
        c.drawString(MARGIN, y, "Color data: craft-color-codes by MakeBead (makebead.com), CC BY 4.0.")

    def _matrix_image(self, k: int, fade: float = 0.0) -> np.ndarray:
        pal_rgb = np.array([p["rgb"] for p in self.pal], np.uint8)
        img = np.full((self.rows, self.cols, 3), 255, np.uint8)
        st = self.m >= 0
        img[st] = pal_rgb[self.m[st]]
        if fade:
            img = (255 - (255 - img.astype(float)) * (1 - fade)).astype(np.uint8)
        return np.repeat(np.repeat(img, k, 0), k, 1)

    def overview(self, c):
        """Clean stitched-result preview: only the master matrix, no labels or chart lines."""
        self._header(c, "Stitched result overview — reconstructed from the chart")
        avail_w = PAGE_W - 2 * MARGIN
        avail_h = PAGE_H - 2 * MARGIN - HEADER_H - FOOTER_H - 40
        s = min(avail_w / self.cols, avail_h / self.rows)
        w, h = self.cols * s, self.rows * s
        x0 = (PAGE_W - w) / 2
        ytop = PAGE_H - MARGIN - HEADER_H - 8
        c.drawImage(self._img(self._matrix_image(6)), x0, ytop - h, w, h)
        c.setFont("TD-Sans", 7.5)
        c.drawString(MARGIN, ytop - h - 14,
                     "Every cell above comes from the master stitch matrix (DMC color per stitch, white = no stitch). "
                     "It is not the original artwork.")
        c.drawString(MARGIN, ytop - h - 24, "Chart pages and their order are shown on the next page (chart navigation).")

    def navigation(self, c):
        """Separate chart navigation: which chart page holds which columns and rows."""
        self._header(c, "Chart navigation", f"{len(self.tiles)} chart pages")
        avail_w = PAGE_W - 2 * MARGIN
        avail_h = (PAGE_H - 2 * MARGIN - HEADER_H - FOOTER_H) * 0.55
        s = min(avail_w / self.cols, avail_h / self.rows)
        w, h = self.cols * s, self.rows * s
        x0 = (PAGE_W - w) / 2
        ytop = PAGE_H - MARGIN - HEADER_H - 8
        c.drawImage(self._img(self._matrix_image(4, fade=0.55)), x0, ytop - h, w, h)
        c.setStrokeColor(Color(0.1, 0.1, 0.1)); c.setLineWidth(1.4)
        c.rect(x0, ytop - h, w, h)
        c.setLineWidth(1.0); c.setDash(4, 2)
        for t in self.tiles:
            c.rect(x0 + t.c0 * s, ytop - t.r1 * s, (t.c1 - t.c0) * s, (t.r1 - t.r0) * s)
        c.setDash()
        for t in self.tiles:
            cx = x0 + (t.c0 + t.c1) / 2 * s
            cy = ytop - (t.r0 + t.r1) / 2 * s
            c.setFillColor(Color(1, 1, 1, alpha=0.9))
            c.roundRect(cx - 17, cy - 9, 34, 18, 4, stroke=0, fill=1)
            c.setFillColor(black); c.setFont("TD-Bold", 10)
            c.drawCentredString(cx, cy - 3.5, t.id)
        y = ytop - h - 22
        cols_x = (0, 32, 62, 122)          # Chart, Page, Columns, Rows within one block
        block_w = (PAGE_W - 2 * MARGIN) / 3
        lh = 9.5
        per_col = max(1, int((y - MARGIN - FOOTER_H - 14) // lh) - 1)
        first_chart_page = 6
        if len(self.tiles) > 3 * per_col:
            last = first_chart_page + len(self.tiles) - 1
            c.setFont("TD-Sans", 8.5)
            c.drawString(MARGIN, y, f"Chart pages {first_chart_page}–{last} follow in reading order: chart row A left "
                                    "to right, then row B, and so on. Each page header names its columns and rows.")
            self._nav_footer(c)
            return
        for blk in range(min(3, math.ceil(len(self.tiles) / per_col))):    # only blocks that hold charts
            c.setFont("TD-Bold", 7.5)
            for x, txt in zip(cols_x, ("Chart", "Page", "Columns", "Rows")):
                c.drawString(MARGIN + blk * block_w + x, y, txt)
        c.setFont("TD-Sans", 7.5)
        for i, t in enumerate(self.tiles):
            blk, row = divmod(i, per_col)
            bx, yy = MARGIN + blk * block_w, y - lh * (row + 1)
            c.drawString(bx + cols_x[0], yy, t.id)
            c.drawString(bx + cols_x[1], yy, str(first_chart_page + i))
            c.drawString(bx + cols_x[2], yy, f"{t.c0 + 1}–{t.c1}")
            c.drawString(bx + cols_x[3], yy, f"{t.r0 + 1}–{t.r1}")
        self._nav_footer(c)

    def _nav_footer(self, c):
        c.setFont("TD-Sans", 7.5)
        c.drawString(MARGIN, MARGIN + FOOTER_H, "Dashed boxes = chart pages (letter = chart row, number = chart column). "
                                                "Overlap cells on each chart are references only; stitch them once.")

    def chart(self, c, t: Tile, page_no: int):
        nx = max(tt.tc for tt in self.tiles) + 1
        ny = max(tt.tr for tt in self.tiles) + 1
        self._header(c, f"Chart {t.id}",
                     f"Columns {t.c0 + 1}–{t.c1} · Rows {t.r0 + 1}–{t.r1}")
        cs = self.cell
        gl = GHOST if t.tc > 0 else 0
        gr = GHOST if t.tc < nx - 1 else 0
        gt = GHOST if t.tr > 0 else 0
        gb = GHOST if t.tr < ny - 1 else 0
        vc0, vc1 = t.c0 - gl, t.c1 + gr
        vr0, vr1 = t.r0 - gt, t.r1 + gb
        width = (vc1 - vc0) * cs
        left = MARGIN + LABEL_W + (PAGE_W - 2 * MARGIN - LABEL_W - width) / 2
        top = PAGE_H - MARGIN - HEADER_H - LABEL_W
        X = lambda col: left + (col - vc0) * cs
        Y = lambda row: top - (row - vr0) * cs   # top edge of row
        t.page, t.cell_pt, t.gx0, t.gy0 = page_no, cs, X(t.c0), Y(t.r0)
        fs = cs * SYMBOL_SCALE
        c.setLineWidth(0.25)
        for r in range(vr0, vr1):
            for col in range(vc0, vc1):
                ghost = not (t.r0 <= r < t.r1 and t.c0 <= col < t.c1)
                v = self.m[r, col]
                x, y = X(col), Y(r) - cs
                if v >= 0:
                    rgb = self.pal[v]["rgb"]
                    if ghost:
                        rgb = tuple(int(255 - (255 - ch) * 0.35) for ch in rgb)
                    c.setFillColor(_rgb(rgb))
                else:
                    c.setFillColor(white)
                c.setStrokeColor(Color(0.6, 0.6, 0.6) if not ghost else Color(0.8, 0.8, 0.8))
                c.rect(x, y, cs, cs, fill=1, stroke=1)
                if v >= 0:
                    ink = GHOST_INK if ghost else tuple(ch / 255 for ch in symbol_ink(self.pal[v]["rgb"]))
                    c.setFillColor(Color(*ink))
                    c.setFont("TD-Sym", fs)
                    c.drawCentredString(x + cs / 2, y + cs * 0.27, self.pal[v]["symbol"])
        # bold 10x10 grid lines (global numbering)
        c.setStrokeColor(black)
        for col in range(vc0, vc1 + 1):
            if col % 10 == 0:
                c.setLineWidth(1.1 if t.c0 <= col <= t.c1 else 0.6)
                c.line(X(col), Y(vr0), X(col), Y(vr1))
        for r in range(vr0, vr1 + 1):
            if r % 10 == 0:
                c.setLineWidth(1.1 if t.r0 <= r <= t.r1 else 0.6)
                c.line(X(vc0), Y(r), X(vc1), Y(r))
        # primary tile border (cut line); ghost strips are references only
        c.setLineWidth(1.6); c.setStrokeColor(Color(0.75, 0.05, 0.1))
        c.rect(X(t.c0), Y(t.r1), (t.c1 - t.c0) * cs, (t.r1 - t.r0) * cs)
        # labels: global numbers at multiples of 10 and at the tile edges
        boxes = []
        c.setFillColor(black)
        lf = 6.0
        def col_labels():
            want = {col for col in range(t.c0 + 1, t.c1 + 1) if col % 10 == 0}
            for edge in (t.c0 + 1, t.c1):
                if all(abs(edge - wv) >= 3 for wv in want):
                    want.add(edge)
            return sorted(want)
        def row_labels():
            want = {r for r in range(t.r0 + 1, t.r1 + 1) if r % 10 == 0}
            for edge in (t.r0 + 1, t.r1):
                if all(abs(edge - wv) >= 2 for wv in want):
                    want.add(edge)
            return sorted(want)
        c.setFont("TD-Bold", lf)
        for n in col_labels():                 # n is 1-based column number
            cx = X(n - 1) + cs / 2
            txt = str(n)
            w = pdfmetrics.stringWidth(txt, "TD-Bold", lf)
            c.drawCentredString(cx, top + 4, txt)
            boxes.append(("col", n, cx - w / 2, top + 4, cx + w / 2, top + 4 + lf))
        for n in row_labels():
            cy = Y(n - 1) - cs / 2
            txt = str(n)
            w = pdfmetrics.stringWidth(txt, "TD-Bold", lf)
            xr = X(vc0) - 3
            c.drawRightString(xr, cy - lf * 0.35, txt)
            boxes.append(("row", n, xr - w, cy - lf * 0.35, xr, cy - lf * 0.35 + lf))
        self.label_boxes[page_no] = boxes
        # navigation: neighbours and mini-map
        c.setFont("TD-Sans", 6.5); c.setFillColor(Color(0.3, 0.3, 0.3))
        bottom = Y(vr1)
        nav = []
        for dx, dy, word in ((-1, 0, "left"), (1, 0, "right"), (0, -1, "above"), (0, 1, "below")):
            nb = [tt for tt in self.tiles if tt.tc == t.tc + dx and tt.tr == t.tr + dy]
            if nb:
                nav.append(f"{word}: {nb[0].id}")
        c.drawString(MARGIN, bottom - 11, "Continues " + (", ".join(nav) if nav else "—") +
                     ".  Pale cells outside the red line are overlap references from neighbouring charts; "
                     "stitch them from their own chart.")
        mm = 4.5
        mx = PAGE_W - MARGIN - nx * mm
        my = PAGE_H - 56 + 20 - mm  # top of minimap at PAGE_H-56
        for tt in self.tiles:
            c.setFillColor(Color(0.75, 0.05, 0.1) if tt is t else Color(0.88, 0.88, 0.88))
            c.setStrokeColor(Color(0.5, 0.5, 0.5)); c.setLineWidth(0.3)
            c.rect(mx + tt.tc * mm, PAGE_H - 56 - (tt.tr + 1) * mm, mm, mm, fill=1, stroke=1)
        c.setFillColor(black)

    # ---------------------------------------------------------- build
    def build(self) -> dict:
        c = canvas.Canvas(self.path, pagesize=A4, pageCompression=1, initialFontName="TD-Sans", initialFontSize=9)
        c.setTitle(f"THREADORA Cross Stitch — {self.meta['project_name']}")
        c.setAuthor("THREADORA")
        c.setSubject("Counted cross stitch pattern")
        total = 5 + len(self.tiles)
        n = 1
        for fn in (self.cover, self.materials, self.legend, self.overview, self.navigation):
            fn(c); self._footer(c, n, total); c.showPage(); n += 1
        for t in self.tiles:
            self.chart(c, t, n); self._footer(c, n, total); c.showPage(); n += 1
        c.save()
        return {"pages": total, "titles": self.page_titles, "cell_pt": self.cell,
                "tiles": [asdict(t) for t in self.tiles],
                "label_boxes": {str(k): v for k, v in self.label_boxes.items()},
                "ghost_ink": GHOST_INK, "symbol_scale": SYMBOL_SCALE}
