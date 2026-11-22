"""Design-fidelity audit and source-supported repair at stitch-cell level.

The master matrix is compared with the ORIGINAL image cell by cell:

* extra stitches      stitched cells where the source is (almost) empty
* missing stitches    empty cells where the source is (almost) fully painted
* accidental holes    enclosed empty areas the source does not have
* lost negative space enclosed source gaps (>= 1 stitch) that the chart filled in
* spurious components stitched islands with no source paint under them
* color bleeding      a cell colored with a thread that none of its source pixels
                      are closest to, at a boundary between colors
* unsupported confetti isolated single stitches whose color the source does not show
* uniform-region breaks color changes between neighbouring cells that are the same
                      color in the source

Hard defects (extra/missing/spurious) mean an engine bug: the candidate is rejected.
Soft findings above the thresholds mean the result needs a human look:
REVIEW_NOT_VERIFIED. Metrics are indicators, not proof of identical embroidery.
"""
from __future__ import annotations

import cv2
import numpy as np

from .dmc import ciede2000_matrix, to_lab

NO_STITCH = -1

EXTRA_COV = 0.25        # stitched although < 25% of the cell is painted in the source
MISSING_COV = 0.75      # empty although >= 75% of the cell is painted
SUPPORT_MIN = 0.10      # a cell's thread must be nearest for >= 10% of its source pixels ...
SUPPORT_DE = 12.0       # ... unless it is within this CIEDE2000 of the cell's mean color
REPAIR_VOTE = 0.25      # repair only towards a thread covering >= 25% of the cell
HOLE_FILL_MAX = 2       # accidental pinholes up to this many cells are filled
HOLE_FILL_COV = 0.40    # ... and only where the source is nearly half painted
ACCIDENTAL_HOLE_COV = 0.50  # an enclosed empty area is accidental only if the source is mostly
                            # painted there (mean coverage); otherwise it is real negative space
DETAIL_KEEP = 0.35      # a small detail counts as kept if its color covers >= 35% of its area in stitches
UNIFORM_DE = 2.0        # neighbour cells this close in the source are "the same color"

# Review thresholds (fractions of stitched cells / uniform pairs)
REVIEW_BLEED = 0.01
REVIEW_CONFETTI = 0.005
REVIEW_UNIFORM = 0.12


def _cells_lab(rgb: np.ndarray, opaque: np.ndarray, rows: int, cols: int):
    a = opaque.astype(np.float64)
    cov = cv2.resize(a, (cols, rows), interpolation=cv2.INTER_AREA)
    s = cv2.resize(rgb.astype(np.float64) * a[..., None], (cols, rows), interpolation=cv2.INTER_AREA)
    mean = s / np.maximum(cov, 1e-9)[..., None]
    lab = to_lab(np.clip(mean, 0, 255).astype(np.uint8).reshape(-1, 3)).reshape(rows, cols, 3)
    return cov, lab


def pixel_lab(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255.0
    lab[..., 1:] -= 128
    return lab


def _reduce(rgb, opaque, rows, cols, f=8):
    h, w = opaque.shape
    if h > f * rows and w > f * cols:
        # ~8x8 source samples per stitch are plenty for vote fractions; keeps it fast
        rgb = cv2.resize(rgb, (cols * f, rows * f), interpolation=cv2.INTER_AREA)
        opaque = cv2.resize(opaque.astype(np.float32), (cols * f, rows * f), interpolation=cv2.INTER_AREA) >= 0.5
    return rgb, opaque


def pixel_labels(rgb, opaque, pal_lab, chunk: int = 256) -> np.ndarray:
    """Nearest palette index for every source pixel (-1 where not opaque)."""
    h, w = opaque.shape
    lab = pixel_lab(rgb)
    p = np.asarray(pal_lab, np.float32)
    idx = np.empty((h, w), np.int16)
    pp = (p ** 2).sum(1)
    for y in range(0, h, chunk):
        blk = lab[y:y + chunk].reshape(-1, 3)
        idx[y:y + chunk] = (pp[None, :] - 2.0 * (blk @ p.T)).argmin(1).reshape(-1, w)
    idx[~opaque] = -1
    return idx


def palette_votes(rgb: np.ndarray, opaque: np.ndarray, rows: int, cols: int,
                  pal_lab: np.ndarray, chunk: int = 256) -> np.ndarray:
    """votes[r, c, k] = fraction of cell (r, c) whose opaque source pixels are
    nearest to palette color k (Euclidean Lab at pixel level; fast and stable)."""
    rgb, opaque = _reduce(rgb, opaque, rows, cols)
    h, w = opaque.shape
    lab = pixel_lab(rgb)
    p = pal_lab.astype(np.float32)
    idx = np.empty((h, w), np.int16)
    pp = (p ** 2).sum(1)
    for y in range(0, h, chunk):
        blk = lab[y:y + chunk].reshape(-1, 3)
        d = pp[None, :] - 2.0 * (blk @ p.T)          # |x-p|^2 minus the constant |x|^2
        idx[y:y + chunk] = d.argmin(1).reshape(-1, w)
    votes = np.zeros((rows, cols, len(p)))
    for k in range(len(p)):
        m = ((idx == k) & opaque).astype(np.float32)
        votes[..., k] = cv2.resize(m, (cols, rows), interpolation=cv2.INTER_AREA)
    return votes


def source_hole_cells(opaque: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Fraction of each cell covered by ENCLOSED background in the source
    (transparent/background regions not connected to the image border)."""
    h, w = opaque.shape
    inv = (~opaque).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    keep = np.zeros(n, bool)
    keep[1:] = True
    for b in border:
        keep[b] = False
    enclosed = keep[lab]
    return cv2.resize(enclosed.astype(np.float64), (cols, rows), interpolation=cv2.INTER_AREA)


def _enclosed_empty_components(stitch: np.ndarray):
    n, lab, stats, _ = cv2.connectedComponentsWithStats((~stitch).astype(np.uint8), connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    return [(i, lab == i, int(stats[i, cv2.CC_STAT_AREA])) for i in range(1, n) if i not in border]


def _isolated(matrix: np.ndarray) -> np.ndarray:
    rows, cols = matrix.shape
    pad = np.pad(matrix, 1, constant_values=NO_STITCH - 1)
    same = np.zeros((rows, cols), bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            same |= pad[1 + dy:1 + dy + rows, 1 + dx:1 + dx + cols] == matrix
    return (matrix >= 0) & ~same


def _neighbour_majority(matrix: np.ndarray, r: int, c: int) -> int:
    nb = matrix[max(0, r - 1):r + 2, max(0, c - 1):c + 2].reshape(-1)
    nb = nb[nb >= 0]
    if len(nb) == 0:
        return NO_STITCH
    vals, cnt = np.unique(nb, return_counts=True)
    return int(vals[cnt.argmax()])


class CellSource:
    """Per-cell source statistics for one grid + palette (computed once per candidate)."""

    def __init__(self, rgb, opaque, rows, cols, pal_lab, votes=None):
        self.rows, self.cols = rows, cols
        self.cov, self.mean_lab = _cells_lab(rgb, opaque, rows, cols)
        self.pal_lab = np.asarray(pal_lab, np.float64)
        self.votes = votes if votes is not None else palette_votes(rgb, opaque, rows, cols, self.pal_lab)
        self.hole_cov = source_hole_cells(opaque, rows, cols)
        self._nearest_de = None
        self._src = (rgb, opaque)
        self._details = None

    def details(self, max_cells: float = 40.0, min_cells: float = 0.75, min_contrast: float = 25.0):
        """Small source features (eyes, catchlights, mouths, thin outlines, dots): connected
        groups of pixels nearest to one palette thread, between half a stitch and max_cells
        stitches in area, that clearly differ from their surroundings (CIEDE2000 >= 25).
        Calibrated so rendered stitch/bead texture specks (< 0.75 stitch, low contrast) are
        not counted as details, while eyes, catchlights, noses and mouths are. Returns a list of
        (thread k, footprint cell mask, vote-weighted cell list, area in cells)."""
        if self._details is not None:
            return self._details
        rgb, opaque = _reduce(*self._src, self.rows, self.cols, f=6)
        h, w = opaque.shape
        idx = pixel_labels(rgb, opaque, self.pal_lab)
        cell_px = (h / self.rows) * (w / self.cols)
        pd = ciede2000_matrix(self.pal_lab, self.pal_lab)
        out = []
        ring_k = np.ones((5, 5), np.uint8)
        for k in range(len(self.pal_lab)):
            mk = (idx == k).astype(np.uint8)
            if not mk.any():
                continue
            n, lab, stats, _ = cv2.connectedComponentsWithStats(mk, connectivity=8)
            for i in range(1, n):
                area = stats[i, cv2.CC_STAT_AREA] / cell_px
                if area < min_cells or area > max_cells:
                    continue
                x, y, bw, bh = stats[i, :4]
                x0, y0 = max(0, x - 3), max(0, y - 3)
                comp = lab[y0:y + bh + 3, x0:x + bw + 3] == i
                ring = (cv2.dilate(comp.astype(np.uint8), ring_k) > 0) & ~comp
                around = idx[y0:y + bh + 3, x0:x + bw + 3][ring]
                around = around[around >= 0]
                if len(around) == 0:
                    continue
                maj = np.bincount(around).argmax()
                if pd[k, maj] < min_contrast:
                    continue                      # shading variant, not a distinct detail
                full = np.zeros((h, w), np.float32)
                full[y0:y + bh + 3, x0:x + bw + 3] = comp
                foot = cv2.resize(full, (self.cols, self.rows), interpolation=cv2.INTER_AREA)
                out.append((k, foot > 0, foot, float(area)))
        self._details = out
        return out

    def lost_details(self, matrix: np.ndarray, protect: np.ndarray | None = None):
        """Details whose thread (or a thread within dE 8 of it) appears nowhere in their footprint."""
        pd = ciede2000_matrix(self.pal_lab, self.pal_lab)
        lost = []
        for k, foot, frac, area in self.details():
            if protect is not None and (foot & protect).any():
                continue
            vals = matrix[foot]
            vals = vals[vals >= 0]
            kept = int((pd[k, vals] < 8).sum()) if len(vals) else 0
            if kept >= DETAIL_KEEP * area or kept >= foot.sum():
                continue
            lost.append((k, foot, frac, area))
        return lost

    @property
    def hard_edge(self) -> np.ndarray:
        """Cells split between two clearly different threads (>= 20% each, >= 80% together,
        CIEDE2000 between them > 20): a hard color boundary in the source."""
        if self.votes.shape[-1] < 2:
            return np.zeros((self.rows, self.cols), bool)
        v = self.votes / np.maximum(self.cov, 1e-9)[..., None]
        order = np.argsort(-v, axis=-1)
        t1, t2 = order[..., 0], order[..., 1]
        v1 = np.take_along_axis(v, t1[..., None], 2)[..., 0]
        v2 = np.take_along_axis(v, t2[..., None], 2)[..., 0]
        pd = ciede2000_matrix(self.pal_lab, self.pal_lab)
        return (v2 >= 0.2) & (v1 + v2 >= 0.8) & (pd[t1, t2] > 20) & (self.cov > 0)

    @property
    def nearest_de(self) -> np.ndarray:
        """CIEDE2000 from each cell's source mean color to the closest palette thread."""
        if self._nearest_de is None:
            flat = self.mean_lab.reshape(-1, 3)
            out = np.empty(len(flat))
            for s0 in range(0, len(flat), 2048):
                out[s0:s0 + 2048] = ciede2000_matrix(flat[s0:s0 + 2048], self.pal_lab).min(1)
            self._nearest_de = out.reshape(self.rows, self.cols)
        return self._nearest_de

    def support(self, matrix):
        st = matrix >= 0
        idx = np.where(st, matrix, 0)
        sup = np.take_along_axis(self.votes, idx[..., None], 2)[..., 0] / np.maximum(self.cov, 1e-9)
        de = np.zeros(matrix.shape)
        if st.any():
            de[st] = _pair_de_vec(self.mean_lab[st], self.pal_lab[idx[st]])
        return np.where(st, sup, 0.0), de


def _pair_de_vec(a, b):
    # vectorised CIEDE2000 (Sharma 2005) for paired rows
    L1, a1, b1 = a[:, 0], a[:, 1], a[:, 2]
    L2, a2, b2 = b[:, 0], b[:, 1], b[:, 2]
    C1 = np.hypot(a1, b1); C2 = np.hypot(a2, b2)
    Cm = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cm ** 7 / (Cm ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dh = h2p - h1p
    dh = np.where(C1p * C2p == 0, 0, np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh)))
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lpm = (L1 + L2) / 2
    Cpm = (C1p + C2p) / 2
    hs = h1p + h2p
    hpm = np.where(C1p * C2p == 0, hs,
                   np.where(np.abs(h1p - h2p) <= 180, hs / 2, np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2)))
    T = (1 - 0.17 * np.cos(np.radians(hpm - 30)) + 0.24 * np.cos(np.radians(2 * hpm))
         + 0.32 * np.cos(np.radians(3 * hpm + 6)) - 0.20 * np.cos(np.radians(4 * hpm - 63)))
    dth = 30 * np.exp(-(((hpm - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cpm ** 7 / (Cpm ** 7 + 25 ** 7))
    Sl = 1 + 0.015 * (Lpm - 50) ** 2 / np.sqrt(20 + (Lpm - 50) ** 2)
    Sc = 1 + 0.045 * Cpm
    Sh = 1 + 0.015 * Cpm * T
    Rt = -np.sin(np.radians(2 * dth)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


def repair(matrix: np.ndarray, cs: CellSource, protect_open: np.ndarray | None = None,
           protect: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Source-supported corrections only:
    1. color bleeding: a cell whose thread is not supported by its own source pixels takes
       the thread that covers most of the cell (if that covers >= REPAIR_VOTE of it);
    2. accidental pinholes (<= HOLE_FILL_MAX enclosed empty cells, not a source gap and not a
       protected letter counter) where the source has paint are filled with the
       neighbourhood thread.
    Nothing is invented: every change is backed by the source pixels of that cell."""
    m = matrix.copy()
    sup, de = cs.support(m)
    st = m >= 0
    bad = st & (sup < SUPPORT_MIN) & ((de > SUPPORT_DE) | cs.hard_edge)
    if protect is not None:
        bad &= ~protect
    best = cs.votes.argmax(-1)
    bestv = np.take_along_axis(cs.votes, best[..., None], 2)[..., 0] / np.maximum(cs.cov, 1e-9)
    fix = bad & (bestv >= REPAIR_VOTE) & (best != m)
    m[fix] = best[fix]
    # lost small details (eyes, catchlights, mouths, dots): restore them in the stitched
    # cells their own source pixels cover most (>= 30% of the cell), never in empty cells
    restored = 0
    for k, foot, frac, area in cs.lost_details(m, protect):
        share = frac / np.maximum(cs.cov, 1e-9)
        ok = foot & (m >= 0) & (share >= 0.3) & (m != k)
        cand = np.argwhere(ok)
        if not len(cand):
            continue
        have = int((m[foot] == k).sum())
        need = max(1, int(np.ceil(DETAIL_KEEP * area)) - have)
        order = np.argsort(-share[ok])
        for r, c in cand[order][:need]:
            m[r, c] = k
            restored += 1
    filled = 0
    po = protect_open if protect_open is not None else np.zeros(m.shape, bool)
    for _i, comp, area in _enclosed_empty_components(m >= 0):
        if area > HOLE_FILL_MAX or (comp & po).any() or cs.hole_cov[comp].max() >= 0.25 \
                or cs.cov[comp].mean() < ACCIDENTAL_HOLE_COV:
            continue
        for r, c in zip(*np.where(comp)):
            if cs.cov[r, c] >= HOLE_FILL_COV:
                v = _neighbour_majority(m, r, c)
                if v >= 0:
                    m[r, c] = v
                    filled += 1
    return m, {"bleed_recolored": int(fix.sum()), "details_restored": restored, "pinholes_filled": filled}


def repair_confetti(matrix: np.ndarray, cs: CellSource, protect: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Isolated single stitches whose thread the source does not support take the
    neighbourhood thread, but ONLY if that thread matches the cell's own source pixels at
    least as well. Supported isolated stitches (highlights, eyes, sparkles) are kept."""
    m = matrix.copy()
    sup, de = cs.support(m)
    cand = _isolated(m) & (sup < 0.3) & (de > SUPPORT_DE / 2)
    if protect is not None:
        cand &= ~protect
    changed = 0
    cov = np.maximum(cs.cov, 1e-9)
    for r, c in zip(*np.where(cand)):
        v = _neighbour_majority(_mask_self(m, r, c), r, c)
        if v < 0 or v == m[r, c]:
            continue
        de_new = _pair_de_vec(cs.mean_lab[r, c][None], cs.pal_lab[v][None])[0]
        if de_new <= de[r, c] + 2.0 or cs.votes[r, c, v] / cov[r, c] > sup[r, c]:
            m[r, c] = v
            changed += 1
    return m, changed


def _mask_self(m, r, c):
    out = m.copy()
    out[r, c] = NO_STITCH
    return out


def audit(matrix: np.ndarray, cs: CellSource, protect_open: np.ndarray | None = None,
          protect: np.ndarray | None = None) -> dict:
    """Compare the matrix with the source. Returns counts, coordinates (first 50 of each,
    1-based row/col) and a verdict: hard_fail (engine bug), review (needs a human look)."""
    rows, cols = matrix.shape
    st = matrix >= 0
    po = protect_open if protect_open is not None else np.zeros(matrix.shape, bool)
    n_st = max(1, int(st.sum()))
    extra = st & (cs.cov < EXTRA_COV)
    missing = ~st & (cs.cov >= MISSING_COV) & ~po
    # holes
    acc_holes, kept_holes = [], 0
    for _i, comp, area in _enclosed_empty_components(st):
        if (comp & po).any() or cs.hole_cov[comp].max() >= 0.25 or cs.cov[comp].mean() < ACCIDENTAL_HOLE_COV:
            kept_holes += 1
        else:
            acc_holes.append(comp)
    acc_mask = np.zeros(matrix.shape, bool)
    for comp in acc_holes:
        acc_mask |= comp
    # source gaps of >= 1 stitch that the chart filled in completely
    src_gap = cs.hole_cov >= 0.5
    n, lab = cv2.connectedComponents(src_gap.astype(np.uint8), connectivity=4)
    lost_gaps = [i for i in range(1, n) if st[lab == i].all()]
    # spurious stitched islands: no source paint at all under them
    n2, lab2, stats2, _ = cv2.connectedComponentsWithStats(st.astype(np.uint8), connectivity=8)
    spurious = [i for i in range(1, n2) if cs.cov[lab2 == i].max() < EXTRA_COV]
    # color support
    sup, de = cs.support(matrix)
    unsupported = st & (sup < SUPPORT_MIN) & (de > SUPPORT_DE)
    multi = (cs.votes / np.maximum(cs.cov, 1e-9)[..., None] >= 0.2).sum(-1) >= 2
    boundary = multi | (cs.cov < 0.95)
    bleed = (unsupported & boundary) | (st & (sup < SUPPORT_MIN) & cs.hard_edge)
    iso = _isolated(matrix)
    # isolated stitch whose thread neither covers its source pixels nor is the best
    # available match for the cell's color: noise, not detail
    confetti_bad = iso & (sup < 0.3) & (de > SUPPORT_DE / 2) & (de > cs.nearest_de + 2.0)
    # uniform-region breaks
    full = st & (cs.cov >= 0.9)
    pairs = changed = 0
    for a_sl, b_sl in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                       ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
        both = full[a_sl] & full[b_sl]
        if not both.any():
            continue
        d = _pair_de_vec(cs.mean_lab[a_sl][both], cs.mean_lab[b_sl][both])
        uni = d < UNIFORM_DE
        pairs += int(uni.sum())
        changed += int((matrix[a_sl][both][uni] != matrix[b_sl][both][uni]).sum())
    uniform_rate = changed / pairs if pairs else 0.0

    lost = cs.lost_details(matrix, protect)
    lost_mask = np.zeros(matrix.shape, bool)
    for _k, foot, _f, _a in lost:
        lost_mask |= foot

    def coords(mask):
        return [[int(r) + 1, int(c) + 1] for r, c in zip(*np.where(mask))][:50]

    res = {
        "stitched_cells": int(st.sum()),
        "extra_stitches": int(extra.sum()), "extra_at": coords(extra),
        "missing_stitches": int(missing.sum()), "missing_at": coords(missing),
        "accidental_holes": len(acc_holes), "accidental_hole_cells": int(acc_mask.sum()),
        "accidental_holes_at": coords(acc_mask),
        "intentional_holes_kept": kept_holes,
        "source_gaps_filled": len(lost_gaps),
        "spurious_components": len(spurious),
        "color_bleed_cells": int(bleed.sum()), "color_bleed_rate": round(bleed.sum() / n_st, 5),
        "color_bleed_at": coords(bleed),
        "isolated_stitches": int(iso.sum()), "unsupported_confetti": int(confetti_bad.sum()),
        "unsupported_confetti_at": coords(confetti_bad),
        "lost_details": len(lost), "lost_details_at": coords(lost_mask),
        "uniform_pairs": pairs, "uniform_breaks": changed, "uniform_break_rate": round(uniform_rate, 4),
    }
    hard = []
    if res["extra_stitches"]:
        hard.append(f"{res['extra_stitches']} stitches where the source has no design")
    if res["missing_stitches"]:
        hard.append(f"{res['missing_stitches']} design cells left unstitched")
    if res["spurious_components"]:
        hard.append(f"{res['spurious_components']} stitched islands with no source paint")
    review = []
    if res["accidental_holes"]:
        review.append(f"{res['accidental_holes']} enclosed gap(s) in the chart that the source does not have")
    if res["source_gaps_filled"]:
        review.append(f"{res['source_gaps_filled']} enclosed source gap(s) of at least one stitch were filled in")
    if res["color_bleed_rate"] > REVIEW_BLEED:
        review.append(f"{res['color_bleed_cells']} boundary cells use a thread their source pixels do not show "
                      "(possible color bleeding)")
    if res["unsupported_confetti"] > REVIEW_CONFETTI * n_st:
        review.append(f"{res['unsupported_confetti']} isolated single stitches not supported by the source")
    if lost:
        review.append(f"{len(lost)} small source detail(s) (eye, highlight, mouth, dot or thin line) have no "
                      "stitch of their color; they may be too small for full cross stitches at this size")
    if uniform_rate > REVIEW_UNIFORM:
        review.append(f"{uniform_rate:.0%} of same-color neighbour cells change thread (uneven flat areas)")
    res["hard_fail"] = hard
    res["review"] = review
    res["pass"] = not hard
    return res


def defect_overlay(matrix: np.ndarray, pal_rgb: np.ndarray, result: dict, scale: int = 6) -> np.ndarray:
    """Owner preview: matrix image with audit findings outlined
    (red = extra/missing, orange = accidental hole, magenta = bleeding, blue = confetti)."""
    from .convert import render_matrix
    img = render_matrix(matrix, pal_rgb, scale=scale).copy()
    colors = {"extra_at": (230, 0, 0), "missing_at": (230, 0, 0), "accidental_holes_at": (255, 140, 0),
              "color_bleed_at": (220, 0, 220), "unsupported_confetti_at": (0, 90, 255),
              "lost_details_at": (0, 170, 0)}
    for key, col in colors.items():
        for r, c in result.get(key, []):
            y, x = (r - 1) * scale, (c - 1) * scale
            cv2.rectangle(img, (x, y), (x + scale - 1, y + scale - 1), col, 1)
    return img
