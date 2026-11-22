"""Pixel -> stitch-cell conversion, DMC palette optimization, confetti cleanup."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .dmc import ciede2000_matrix, dmc_lab, load_dmc, nearest, nearest_dmc, to_lab

NO_STITCH = -1
METHODS = ("area", "edge_aware", "dominant")


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055) * 255.0


@dataclass
class CellFields:
    cols: int
    rows: int
    coverage: np.ndarray   # rows x cols, fraction of opaque pixels in the cell
    color: np.ndarray      # rows x cols x 3 sRGB float, mean of opaque pixels (linear-light)


def cell_fields(rgb: np.ndarray, opaque: np.ndarray, cols: int, rows: int,
                method: str = "area") -> CellFields:
    """Exact area integration of the source over the stitch grid (cv2 INTER_AREA
    is an exact box filter for downscaling, including non-integer ratios)."""
    src = rgb
    if method == "edge_aware":
        # Remove the rendered stitch texture while keeping real edges.
        src = cv2.bilateralFilter(rgb, 9, 40, 7)
    a = opaque.astype(np.float64)
    lin = srgb_to_linear(src.astype(np.float64)) * a[..., None]
    cov = cv2.resize(a, (cols, rows), interpolation=cv2.INTER_AREA)
    s = cv2.resize(lin, (cols, rows), interpolation=cv2.INTER_AREA)
    col = linear_to_srgb(s / np.maximum(cov, 1e-9)[..., None])
    return CellFields(cols, rows, cov, col)


def dominant_assign(rgb: np.ndarray, opaque: np.ndarray, cols: int, rows: int,
                    pal_lab: np.ndarray) -> np.ndarray:
    """Assign every opaque source pixel to the nearest palette color, then give each
    cell the palette color covering most of its opaque area (crisp strokes)."""
    h, w = opaque.shape
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    # OpenCV 8-bit Lab -> CIE Lab
    lab[..., 0] *= 100 / 255.0
    lab[..., 1:] -= 128
    flat = lab.reshape(-1, 3)
    p = pal_lab.astype(np.float32)
    d = ((flat[:, None, :] - p[None, :, :]) ** 2).sum(-1)
    idx = d.argmin(1).reshape(h, w)
    votes = np.zeros((rows, cols, len(p)))
    for k in range(len(p)):
        m = ((idx == k) & opaque).astype(np.float64)
        votes[..., k] = cv2.resize(m, (cols, rows), interpolation=cv2.INTER_AREA)
    return votes.argmax(-1)


@dataclass
class PaletteResult:
    dmc_idx: list[int]                  # indices into load_dmc()
    matrix: np.ndarray                  # rows x cols, index into dmc_idx or NO_STITCH
    mean_de: float
    p95_de: float
    notes: list[str] = field(default_factory=list)


def _assign(cell_lab: np.ndarray, pal_lab: np.ndarray):
    d = ciede2000_matrix(cell_lab, pal_lab)
    i = d.argmin(1)
    return i, d[np.arange(len(i)), i]


def build_palette(fields: CellFields, stitch: np.ndarray, k: int, weights: np.ndarray,
                  forced_dmc: list[int] | None = None,
                  merge_de: float = 4.0, seed: int = 7) -> PaletteResult:
    """k-means in Lab (critical cells weighted), centroids snapped to real DMC
    colors by CIEDE2000, refined, near-duplicate threads merged."""
    dlab = dmc_lab()
    rows, cols = stitch.shape
    cells = fields.color[stitch].reshape(-1, 3)
    cell_lab = to_lab(np.clip(cells, 0, 255).astype(np.uint8))
    w = weights[stitch].reshape(-1)
    rep = np.repeat(np.arange(len(cell_lab)), np.maximum(1, np.round(w).astype(int)))
    data = cell_lab[rep].astype(np.float32)
    k_eff = min(k, len(np.unique(np.round(cell_lab), axis=0)))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.05)
    cv2.setRNGSeed(seed)
    _, _, centers = cv2.kmeans(data, k_eff, None, crit, 4, cv2.KMEANS_PP_CENTERS)
    sel, _ = nearest_dmc(centers.astype(np.float64))
    sel = list(dict.fromkeys(int(s) for s in sel))
    forced = list(dict.fromkeys(forced_dmc or []))
    # refinement: Lloyd iterations constrained to DMC
    for _ in range(4):
        a, _ = _assign(cell_lab, dlab[sel])
        new = []
        for j in range(len(sel)):
            m = a == j
            if not m.any():
                continue
            mu = np.average(cell_lab[m], axis=0, weights=w[m])
            cand, _ = nearest_dmc(mu[None])
            new.append(int(cand[0]))
        new = list(dict.fromkeys(new))
        if new == sel:
            break
        sel = new
    # lettering colors are forced into the palette (they replace near-duplicates)
    for fi in forced:
        if fi in sel:
            continue
        dd = ciede2000_matrix(dlab[[fi]], dlab[sel])[0] if sel else np.array([])
        near = [sel[j] for j in np.where(dd < merge_de)[0] if sel[j] not in forced]
        sel = [x for x in sel if x not in near] + [fi]
    # merge threads that are perceptually too close to be worth separate skeins
    while len(sel) > 1:
        a, _ = _assign(cell_lab, dlab[sel])
        counts = np.bincount(a, minlength=len(sel))
        pd = ciede2000_matrix(dlab[sel], dlab[sel])
        np.fill_diagonal(pd, np.inf)
        i, j = np.unravel_index(pd.argmin(), pd.shape)
        if pd[i, j] >= merge_de:
            break
        drop = i if counts[i] < counts[j] else j
        if sel[drop] in forced:
            drop = j if drop == i else i
        if sel[drop] in forced:
            break
        sel.pop(drop)
    a, de = _assign(cell_lab, dlab[sel])
    # drop unused colors
    used = np.unique(a)
    sel = [sel[u] for u in used] + [f for f in forced if f not in [sel[u] for u in used]]
    a, de = _assign(cell_lab, dlab[sel])
    matrix = np.full((rows, cols), NO_STITCH, dtype=np.int32)
    matrix[stitch] = a
    return PaletteResult(sel, matrix, float(de.mean()), float(np.percentile(de, 95)))


def remove_confetti(matrix: np.ndarray, pal_lab: np.ndarray, protect: np.ndarray,
                    max_de: float = 12.0) -> tuple[np.ndarray, int, int]:
    """Replace isolated single stitches (no 8-neighbour of the same color) by the
    neighbourhood majority when the color difference is small (texture noise).
    High-contrast isolated stitches and protected cells are kept as intentional detail.
    Returns (new_matrix, removed, kept_isolated)."""
    m = matrix.copy()
    rows, cols = m.shape
    pad = np.pad(m, 1, constant_values=NO_STITCH - 1)
    removed = kept = 0
    pd = ciede2000_matrix(pal_lab, pal_lab)
    for r in range(rows):
        for c in range(cols):
            v = m[r, c]
            if v == NO_STITCH:
                continue
            nb = pad[r:r + 3, c:c + 3].reshape(-1)
            nb = np.delete(nb, 4)
            if (nb == v).any():
                continue
            st = nb[nb >= 0]
            if len(st) < 5:
                kept += 1
                continue
            vals, cnt = np.unique(st, return_counts=True)
            maj = vals[cnt.argmax()]
            if protect[r, c] or pd[v, maj] > max_de:
                kept += 1
                continue
            m[r, c] = maj
            removed += 1
    return m, removed, kept


def isolated_count(matrix: np.ndarray) -> int:
    pad = np.pad(matrix, 1, constant_values=NO_STITCH - 1)
    n = 0
    rows, cols = matrix.shape
    for r in range(rows):
        for c in range(cols):
            v = matrix[r, c]
            if v == NO_STITCH:
                continue
            nb = np.delete(pad[r:r + 3, c:c + 3].reshape(-1), 4)
            if not (nb == v).any():
                n += 1
    return n


def render_matrix(matrix: np.ndarray, pal_rgb: np.ndarray, scale: int = 1,
                  bg=(255, 255, 255)) -> np.ndarray:
    """Stitched-result preview reconstructed ONLY from the master matrix."""
    rows, cols = matrix.shape
    out = np.empty((rows, cols, 3), np.uint8)
    out[:] = bg
    st = matrix >= 0
    out[st] = pal_rgb[matrix[st]]
    if scale > 1:
        out = np.repeat(np.repeat(out, scale, 0), scale, 1)
    return out
