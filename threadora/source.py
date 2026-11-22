"""Source loading, hashing, orientation, alpha handling and source-type detection."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ALPHA_STITCH_THRESHOLD = 128  # alpha >= this is an opaque (stitchable) pixel


@dataclass
class Source:
    path: Path
    sha256: str
    size_bytes: int
    width: int
    height: int
    mode: str
    exif_orientation: int | None
    has_alpha: bool
    rgba: np.ndarray  # uint8 HxWx4 after EXIF orientation
    opaque: np.ndarray  # bool HxW: pixels that are stitch candidates
    background_method: str
    source_type: str  # "A" artwork or "B" existing chart
    type_evidence: dict = field(default_factory=dict)

    def composite_on(self, bg=(255, 255, 255)) -> np.ndarray:
        rgb = self.rgba[..., :3].astype(np.float64)
        a = self.rgba[..., 3:4].astype(np.float64) / 255.0
        return np.clip(rgb * a + np.array(bg) * (1 - a), 0, 255).astype(np.uint8)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _border_background(rgb: np.ndarray, tol: float = 12.0) -> np.ndarray:
    """Conservative background detection for opaque images: flood-fill from the
    border with a near-uniform color. Returns bool mask of background."""
    h, w = rgb.shape[:2]
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]).astype(float)
    med = np.median(border, axis=0)
    if np.mean(np.linalg.norm(border - med, axis=1) < tol) < 0.9:
        return np.zeros((h, w), bool)  # border is not uniform: no background removal
    close = (np.linalg.norm(rgb.astype(float) - med, axis=2) < tol).astype(np.uint8)
    n, lab = cv2.connectedComponents(close, connectivity=4)
    edge_labels = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(edge_labels))
    return bg


def detect_chart_grid(rgb: np.ndarray) -> dict:
    """Detect a printed chart grid: many long, regularly spaced straight lines
    in both directions. Embroidery-textured art has no full-length lines."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(g, 60, 160)
    h, w = g.shape
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=200,
                            minLineLength=int(0.6 * min(h, w)), maxLineGap=4)
    hs, vs = [], []
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            if abs(y1 - y2) <= 2:
                hs.append((y1 + y2) / 2)
            elif abs(x1 - x2) <= 2:
                vs.append((x1 + x2) / 2)

    def regular(pos):
        pos = np.unique(np.round(pos))
        if len(pos) < 8:
            return False, 0.0
        d = np.diff(pos)
        d = d[d > 2]
        if len(d) < 6:
            return False, 0.0
        med = np.median(d)
        return bool(np.mean(np.abs(d - med) < 0.25 * med) > 0.7), float(med)

    rh, sh = regular(hs)
    rv, sv = regular(vs)
    return {"long_horizontal_lines": len(hs), "long_vertical_lines": len(vs),
            "regular_h": rh, "regular_v": rv, "spacing_h": sh, "spacing_v": sv,
            "is_chart": bool(rh and rv)}


def load_source(path: str | Path) -> Source:
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    with Image.open(path) as im:
        mode = im.mode
        exif = im.getexif()
        orient = exif.get(0x0112) if exif else None
        im = ImageOps.exif_transpose(im)
        has_alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
        rgba = np.array(im.convert("RGBA"))
    # Bytes must be unchanged by loading.
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    alpha = rgba[..., 3]
    if has_alpha and (alpha < 255).any():
        opaque = alpha >= ALPHA_STITCH_THRESHOLD
        method = f"PNG alpha channel (alpha >= {ALPHA_STITCH_THRESHOLD} = stitch, else NO_STITCH)"
    else:
        has_alpha = False
        bg = _border_background(rgba[..., :3])
        opaque = ~bg
        method = ("border flood-fill of uniform background color" if bg.any()
                  else "fully opaque: every pixel is stitched (no certain background)")
    grid = detect_chart_grid(rgba[..., :3])
    return Source(path=path, sha256=digest, size_bytes=len(raw), width=rgba.shape[1],
                  height=rgba.shape[0], mode=mode, exif_orientation=orient,
                  has_alpha=has_alpha, rgba=rgba, opaque=opaque, background_method=method,
                  source_type="B" if grid["is_chart"] else "A", type_evidence=grid)


def content_bbox(opaque: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box (x0, y0, x1, y1 exclusive) of stitchable pixels. Only rows and
    columns with no stitchable pixel at all are outside it, so no design content is lost."""
    ys, xs = np.where(opaque)
    if len(xs) == 0:
        raise ValueError("source has no stitchable pixels")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
