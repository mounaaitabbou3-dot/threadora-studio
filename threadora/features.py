"""Critical-feature detection in the source and fidelity gates against the
stitched matrix. Metrics are indicators, not proof of identical embroidery."""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np
from skimage.feature import blob_log
from skimage.metrics import structural_similarity

try:
    import pytesseract
    HAVE_OCR = True
except Exception:  # pragma: no cover
    HAVE_OCR = False

WINDOWS_TESSERACT_PATHS = (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                           r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe")


def configure_tesseract() -> str | None:
    """Point pytesseract at the Tesseract binary: THREADORA_TESSERACT_CMD, then PATH,
    then the default Windows install folders. Returns the command used, or None."""
    import os
    import shutil
    if not HAVE_OCR:
        return None
    cmd = os.environ.get("THREADORA_TESSERACT_CMD") or shutil.which("tesseract")
    if not cmd:
        cmd = next((p for p in WINDOWS_TESSERACT_PATHS if os.path.exists(p)), None)
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    return cmd


def ocr_status() -> dict:
    """Whether OCR-based lettering gates can actually run on this machine."""
    cmd = configure_tesseract()
    if not cmd:
        return {"available": False, "detail": "Tesseract OCR not found (install it or set THREADORA_TESSERACT_CMD)."}
    try:
        return {"available": True, "detail": f"Tesseract {pytesseract.get_tesseract_version()} at {cmd}"}
    except Exception as e:  # binary present but broken
        return {"available": False, "detail": f"Tesseract at {cmd} failed: {e}"}


configure_tesseract()


# Size thresholds (fraction of the text-line height squared) for letter parts and
# counters that matter for legibility. Specks of rendered stitch texture are ignored.
COMP_FRAC = 0.02
HOLE_FRAC = 0.006


def source_holes(mask: np.ndarray, bh: int) -> list[np.ndarray]:
    """Significant enclosed background regions (letter counters) as bool masks."""
    inv = (~mask.astype(bool)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    return [lab == i for i in range(1, n)
            if i not in border and stats[i, cv2.CC_STAT_AREA] >= HOLE_FRAC * bh * bh]


def _norm_tokens(s: str) -> list[str]:
    return [t for t in re.split(r"[^0-9A-Za-z]+", s) if t]


def ocr_line(img_rgb: np.ndarray) -> str:
    if not HAVE_OCR:
        return ""
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.copyMakeBorder(g, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    return pytesseract.image_to_string(g, config="--psm 7").strip()


@dataclass
class TextLine:
    box: tuple[int, int, int, int]       # x0,y0,x1,y1 in cropped-source pixels
    n_components: int
    source_ocr: str
    expected_tokens: list[str]           # tokens verified readable in the source
    kind: str                            # "ocr_text" | "script_text"
    src_components: int = 0
    src_holes: int = 0


@dataclass
class Blob:
    x: float
    y: float
    r: float
    contrast: float


@dataclass
class Features:
    text_lines: list[TextLine]
    ornaments: list[tuple[int, int, int, int, int]]   # x0,y0,x1,y1,area of detached non-text components
    blobs: list[Blob]
    body_area: int
    notes: list[str] = field(default_factory=list)
    text_mask: np.ndarray | None = None   # full-res bool: pixels of detected lettering

    def as_dict(self):
        return {"text_lines": [asdict(t) for t in self.text_lines],
                "ornaments": self.ornaments,
                "blobs": [asdict(b) for b in self.blobs],
                "notes": self.notes}


def _components(mask: np.ndarray, min_area: int, conn: int = 8):
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=conn)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return keep, lab, stats


def count_components_and_holes(mask: np.ndarray, min_comp: int, min_hole: int,
                               conn: int = 8) -> tuple[int, int]:
    keep, _, _ = _components(mask, min_comp, conn)
    inv = (~mask.astype(bool)).astype(np.uint8)
    hconn = 4 if conn == 8 else 8
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=hconn)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    holes = [i for i in range(1, n) if i not in border and stats[i, cv2.CC_STAT_AREA] >= min_hole]
    return len(keep), len(holes)


def detect_features(rgb_white: np.ndarray, opaque: np.ndarray,
                    personalization: list[str]) -> Features:
    h, w = opaque.shape
    notes = []
    total = int(opaque.sum())
    n, lab, stats, cent = cv2.connectedComponentsWithStats(opaque.astype(np.uint8), connectivity=8)
    min_px = max(12, int(2e-5 * h * w))
    body = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 0.05 * total]
    small = [i for i in range(1, n) if i not in body and stats[i, cv2.CC_STAT_AREA] >= min_px]
    # group detached components into horizontal lines
    boxes = {i: (stats[i, 0], stats[i, 1], stats[i, 0] + stats[i, 2], stats[i, 1] + stats[i, 3]) for i in small}
    groups: list[list[int]] = []
    for i in sorted(small, key=lambda i: boxes[i][0]):
        x0, y0, x1, y1 = boxes[i]
        placed = False
        for g in groups:
            gx0 = min(boxes[j][0] for j in g); gx1 = max(boxes[j][2] for j in g)
            gy0 = min(boxes[j][1] for j in g); gy1 = max(boxes[j][3] for j in g)
            gh = gy1 - gy0
            ov = min(y1, gy1) - max(y0, gy0)
            if ov > 0.4 * min(y1 - y0, gh) and x0 - gx1 < 1.2 * max(gh, y1 - y0):
                g.append(i); placed = True; break
        if not placed:
            groups.append([i])
    expected_all = [t for p in personalization for t in _norm_tokens(p)]
    lines, ornaments = [], []
    text_mask = np.zeros((h, w), bool)
    for g in groups:
        x0 = min(boxes[j][0] for j in g); x1 = max(boxes[j][2] for j in g)
        y0 = min(boxes[j][1] for j in g); y1 = max(boxes[j][3] for j in g)
        bw, bh = x1 - x0, y1 - y0
        if len(g) >= 3 and bw >= 2.0 * bh:
            pad = int(0.15 * bh)
            crop = rgb_white[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
            txt = ocr_line(crop)
            toks = _norm_tokens(txt)
            exp = []
            for t in toks:
                m = difflib.get_close_matches(t, expected_all, n=1, cutoff=0.85)
                if m and m[0] == t and t not in exp:
                    exp.append(t)
            kind = "ocr_text" if exp else "script_text"
            lines.append(TextLine((x0, y0, x1, y1), len(g), txt, exp, kind))
            text_mask |= np.isin(lab, g)
        else:
            for j in g:
                a = int(stats[j, cv2.CC_STAT_AREA])
                bx = boxes[j]
                ornaments.append((int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3]), a))
    # merge personalization tokens not found by OCR
    found = {t for l in lines for t in l.expected_tokens}
    missing = [t for t in expected_all if t not in found]
    if missing:
        notes.append("Personalization tokens not machine-readable in the SOURCE (script/decorative "
                     f"lettering, checked structurally only): {missing}")
    for l in lines:
        x0, y0, x1, y1 = l.box
        bh = y1 - y0
        m = opaque[y0:y1, x0:x1]
        l.src_components, l.src_holes = count_components_and_holes(
            m, min_comp=int(COMP_FRAC * bh * bh), min_hole=int(HOLE_FRAC * bh * bh))
    # small high-contrast blobs (eyes, highlights, berry centers)
    g = cv2.cvtColor(rgb_white, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255
    half = cv2.resize(g, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    blobs = []
    for sign in (1, -1):
        img = half if sign == 1 else 1 - half
        for y, x, s in blob_log(img, min_sigma=3, max_sigma=10, num_sigma=8, threshold=0.12):
            r = s * np.sqrt(2) * 2   # full-res radius
            cx, cy = x * 2, y * 2
            yy, xx = np.ogrid[:h, :w]
            d2 = (xx - cx) ** 2 + (yy - cy) ** 2
            inner = d2 <= r * r
            ring = (d2 > r * r) & (d2 <= 4 * r * r)
            if not opaque[inner].mean() > 0.9 or not opaque[ring].mean() > 0.9:
                continue
            c = g[inner].mean() - g[ring].mean()
            if abs(c) >= 0.25:
                blobs.append(Blob(float(cx), float(cy), float(r), float(c)))
    return Features(lines, ornaments, blobs, int(sum(stats[i, cv2.CC_STAT_AREA] for i in body)),
                    notes, text_mask)


# ---------------------------------------------------------------- gates

def _matrix_box(box, sx, sy, cols, rows):
    x0, y0, x1, y1 = box
    c0 = max(0, int(np.floor(x0 / sx))); c1 = min(cols, int(np.ceil(x1 / sx)))
    r0 = max(0, int(np.floor(y0 / sy))); r1 = min(rows, int(np.ceil(y1 / sy)))
    return c0, r0, c1, r1


def viewing_render(cells_rgb: np.ndarray, px: int = 12, blur: float = 0.42) -> np.ndarray:
    """Simulate viewing the stitched piece: each stitch blends slightly with neighbours."""
    up = np.repeat(np.repeat(cells_rgb, px, 0), px, 1)
    return cv2.GaussianBlur(up, (0, 0), blur * px)


def evaluate(feat: Features, src_white: np.ndarray, opaque: np.ndarray,
             matrix: np.ndarray, pal_rgb: np.ndarray) -> dict:
    """Compare source vs matrix-reconstructed preview globally and in critical regions."""
    from .convert import render_matrix
    from .dmc import to_lab
    from skimage.color import deltaE_ciede2000, rgb2lab
    h, w = opaque.shape
    rows, cols = matrix.shape
    sx, sy = w / cols, h / rows
    prev = render_matrix(matrix, pal_rgb)
    stitch = matrix >= 0
    # --- global
    k = 4
    src_small = cv2.resize(src_white, (cols * k, rows * k), interpolation=cv2.INTER_AREA)
    prev_up = np.repeat(np.repeat(prev, k, 0), k, 1)
    a = cv2.GaussianBlur(src_small, (0, 0), 1.5)
    b = cv2.GaussianBlur(prev_up, (0, 0), 1.5)
    ssim = structural_similarity(a, b, channel_axis=2, data_range=255)
    src_cells = cv2.resize(src_white, (cols, rows), interpolation=cv2.INTER_AREA)
    de = deltaE_ciede2000(rgb2lab(src_cells / 255.0), rgb2lab(prev / 255.0))
    cov = cv2.resize(opaque.astype(np.float64), (cols, rows), interpolation=cv2.INTER_AREA)
    src_mask = cov >= 0.5
    iou = float((src_mask & stitch).sum() / max(1, (src_mask | stitch).sum()))
    ea = cv2.Canny(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), 40, 100) > 0
    eb = cv2.Canny(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), 40, 100) > 0
    dil_a = cv2.dilate(ea.astype(np.uint8), np.ones((5, 5))) > 0
    dil_b = cv2.dilate(eb.astype(np.uint8), np.ones((5, 5))) > 0
    prec = (eb & dil_a).sum() / max(1, eb.sum()); rec = (ea & dil_b).sum() / max(1, ea.sum())
    edge_f1 = float(2 * prec * rec / max(1e-9, prec + rec))
    res = {"global": {"ssim_blurred": float(ssim), "mean_de2000": float(de[stitch | src_mask].mean()),
                      "p95_de2000": float(np.percentile(de[stitch | src_mask], 95)),
                      "mask_iou": iou, "edge_f1": edge_f1}}
    # --- text lines
    lines_out = []
    view = viewing_render(prev)
    px = 12
    for l in feat.text_lines:
        c0, r0, c1, r1 = _matrix_box(l.box, sx, sy, cols, rows)
        m = stitch[r0:r1, c0:c1]
        bh_cells = (l.box[3] - l.box[1]) / sy
        # same physical thresholds as for the source, expressed in stitch cells
        mc, mh = count_components_and_holes(
            m, min_comp=max(1, int(np.ceil(COMP_FRAC * bh_cells * bh_cells * (sy / sx)))),
            min_hole=max(1, int(np.ceil(HOLE_FRAC * bh_cells * bh_cells * (sy / sx)))))
        # IoU of masks at source resolution inside the box
        x0, y0, x1, y1 = l.box
        up = cv2.resize(stitch.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)[y0:y1, x0:x1] > 0
        sm = opaque[y0:y1, x0:x1]
        box_iou = float((up & sm).sum() / max(1, (up | sm).sum()))
        pad = 1
        crop = view[max(0, r0 - pad) * px:(r1 + pad) * px, max(0, c0 - pad) * px:(c1 + pad) * px]
        txt = ocr_line(crop) if l.expected_tokens else ""
        toks = _norm_tokens(txt)
        ocr_ok = all(t in toks for t in l.expected_tokens) if l.expected_tokens else None
        lssim = structural_similarity(
            cv2.cvtColor(a[r0 * k:r1 * k, c0 * k:c1 * k], cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(b[r0 * k:r1 * k, c0 * k:c1 * k], cv2.COLOR_RGB2GRAY), data_range=255)
        struct_ok = abs(mc - l.src_components) <= 1 and (mh == l.src_holes)
        passed = struct_ok and box_iou >= 0.55 and (ocr_ok is not False)
        lines_out.append({"box": l.box, "kind": l.kind, "expected": l.expected_tokens,
                          "letter_height_stitches": round(bh_cells, 2),
                          "preview_ocr": txt, "ocr_ok": ocr_ok,
                          "components_src": l.src_components, "components_matrix": mc,
                          "holes_src": l.src_holes, "holes_matrix": mh,
                          "mask_iou": round(box_iou, 3), "ssim": round(float(lssim), 3),
                          "pass": bool(passed)})
    res["text_lines"] = lines_out
    # --- ornaments: every detached component >= half a stitch must survive
    orn_fail = []
    cell_area = sx * sy
    for (x0, y0, x1, y1, area) in feat.ornaments:
        if area < 0.5 * cell_area:
            continue
        c0, r0, c1, r1 = _matrix_box((x0, y0, x1, y1), sx, sy, cols, rows)
        if not stitch[r0:r1, c0:c1].any():
            orn_fail.append((x0, y0, x1, y1, area))
    n_orn = sum(1 for o in feat.ornaments if o[4] >= 0.5 * cell_area)
    res["ornaments"] = {"checked": n_orn, "lost": orn_fail, "pass": not orn_fail,
                        "sub_stitch_ignored": len(feat.ornaments) - n_orn}
    # --- small blobs
    gprev = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255
    kept, lost, sub = 0, [], 0
    for bl in feat.blobs:
        rc = bl.r / sx
        if 2 * rc < 1.0:
            sub += 1
            continue
        cc, rr = bl.x / sx, bl.y / sy
        yy, xx = np.ogrid[:rows, :cols]
        d2 = (xx + 0.5 - cc) ** 2 + (yy + 0.5 - rr) ** 2
        inner = d2 <= max(rc, 0.75) ** 2
        ring = (d2 > max(rc, 0.75) ** 2) & (d2 <= (2 * max(rc, 0.75)) ** 2)
        if not inner.any() or not ring.any():
            sub += 1
            continue
        c = gprev[inner].mean() - gprev[ring].mean()
        if np.sign(c) == np.sign(bl.contrast) and abs(c) >= 0.4 * abs(bl.contrast):
            kept += 1
        else:
            lost.append({"x": bl.x, "y": bl.y, "diameter_stitches": round(2 * rc, 2),
                         "src_contrast": round(bl.contrast, 3), "matrix_contrast": round(float(c), 3)})
    checked = kept + len(lost)
    big_lost = [b for b in lost if b["diameter_stitches"] >= 2.0]
    n_big = sum(1 for bl in feat.blobs if 2 * bl.r / sx >= 2.0)
    ratio = kept / checked if checked else 1.0
    big_ratio = 1 - len(big_lost) / n_big if n_big else 1.0
    res["blobs"] = {"checked": checked, "kept": kept, "lost": lost, "sub_stitch": sub,
                    "ratio": ratio, "big_checked": n_big, "big_ratio": big_ratio,
                    "pass": ratio >= 0.9 and big_ratio >= 0.9}
    res["pass"] = bool(all(l["pass"] for l in lines_out) and res["ornaments"]["pass"]
                       and res["blobs"]["pass"])
    return res
