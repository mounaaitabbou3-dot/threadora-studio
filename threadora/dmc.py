"""Verified DMC dataset loading and CIEDE2000 matching."""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab

DATA_FILE = Path(__file__).parent / "data" / "dmc_floss.csv"
DATA_SHA256 = "66ebd107ace9f373edc185743d2699fc037e46c36428d72c0e8591d695aa0326"


@dataclass(frozen=True)
class Floss:
    code: str
    name: str
    rgb: tuple[int, int, int]

    @property
    def hex(self) -> str:
        return "#%02X%02X%02X" % self.rgb


class DMCDataError(RuntimeError):
    pass


def dataset_sha256() -> str:
    return hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def load_dmc() -> tuple[Floss, ...]:
    """Load the non-discontinued DMC colors after verifying the dataset hash."""
    if dataset_sha256() != DATA_SHA256:
        raise DMCDataError("DMC dataset hash mismatch: file was modified")
    out = []
    seen = set()
    with DATA_FILE.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["discontinued"].strip().lower() == "true":
                continue
            code = row["code"].strip()
            if code in seen:
                raise DMCDataError(f"duplicate DMC code {code}")
            seen.add(code)
            rgb = (int(row["r"]), int(row["g"]), int(row["b"]))
            if "#%02X%02X%02X" % rgb != row["hex"].strip().upper():
                raise DMCDataError(f"hex/rgb mismatch for {code}")
            out.append(Floss(code, row["name"].strip(), rgb))
    return tuple(out)


@lru_cache(maxsize=1)
def dmc_by_code() -> dict[str, Floss]:
    return {f.code: f for f in load_dmc()}


def to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb_u8, dtype=np.float64).reshape(-1, 1, 3) / 255.0
    return rgb2lab(arr).reshape(-1, 3)


@lru_cache(maxsize=1)
def dmc_lab() -> np.ndarray:
    return to_lab(np.array([f.rgb for f in load_dmc()], dtype=np.uint8))


def ciede2000_matrix(a_lab: np.ndarray, b_lab: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Pairwise CIEDE2000 distances, shape (len(a), len(b))."""
    a_lab = np.asarray(a_lab, dtype=np.float64).reshape(-1, 3)
    b_lab = np.asarray(b_lab, dtype=np.float64).reshape(-1, 3)
    out = np.empty((len(a_lab), len(b_lab)))
    for s in range(0, len(a_lab), chunk):
        blk = a_lab[s:s + chunk]
        out[s:s + chunk] = deltaE_ciede2000(blk[:, None, :], b_lab[None, :, :])
    return out


def nearest(a_lab: np.ndarray, b_lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = ciede2000_matrix(a_lab, b_lab)
    idx = d.argmin(axis=1)
    return idx, d[np.arange(len(idx)), idx]


CHROMA_TIE_DE = 1.5


def nearest_dmc(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CIEDE2000 nearest DMC thread, with a documented near-tie rule: among threads
    within CHROMA_TIE_DE of the best distance, take the one whose chroma (C*) is
    closest to the source, so saturated colors are not replaced by duller,
    browner threads that are only marginally closer."""
    lab = np.asarray(lab, dtype=np.float64).reshape(-1, 3)
    dl = dmc_lab()
    d = ciede2000_matrix(lab, dl)
    best = d.min(axis=1)
    c_src = np.hypot(lab[:, 1], lab[:, 2])
    c_dmc = np.hypot(dl[:, 1], dl[:, 2])
    idx = np.empty(len(lab), int)
    for i in range(len(lab)):
        cand = np.where(d[i] <= best[i] + CHROMA_TIE_DE)[0]
        idx[i] = cand[np.argmin(np.abs(c_dmc[cand] - c_src[i]))]
    return idx, d[np.arange(len(lab)), idx]
