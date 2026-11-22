"""Physical size engine: fabrics, stitch grids, fit without distortion."""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

CM_PER_IN = 2.54


@dataclass(frozen=True)
class Fabric:
    name: str
    count: int          # threads (holes) per inch of the cloth
    over: int           # stitch over 1 or 2 threads
    kind: str           # "Aida" or "Evenweave"
    stitchability: str  # practical difficulty note
    availability: str
    strands: int
    needle: str

    @property
    def practical(self) -> bool:
        """Realistic for a typical customer: up to 28-count over one."""
        return self.stitches_per_inch <= 28

    @property
    def stitches_per_inch(self) -> float:
        return self.count / self.over

    @property
    def label(self) -> str:
        if self.kind == "Aida":
            return f"{self.count}-count Aida"
        return f"{self.count}-count evenweave/linen, stitched over one thread"


# Ordered from simplest to most demanding. Evenweave over two duplicates Aida
# at half the count, so it is listed in the size table, not as a candidate.
FABRICS: tuple[Fabric, ...] = (
    Fabric("Aida 11", 11, 1, "Aida", "beginner", "widely available", 3, "tapestry 22"),
    Fabric("Aida 14", 14, 1, "Aida", "beginner", "widely available", 2, "tapestry 24"),
    Fabric("Aida 16", 16, 1, "Aida", "easy", "widely available", 2, "tapestry 26"),
    Fabric("Aida 18", 18, 1, "Aida", "intermediate", "widely available", 1, "tapestry 26"),
    Fabric("Aida 20", 20, 1, "Aida", "intermediate", "specialist shops", 1, "tapestry 26/28"),
    Fabric("Aida 22", 22, 1, "Aida", "advanced (very fine Aida)", "specialist shops", 1, "tapestry 28"),
    Fabric("Evenweave 25 over one", 25, 1, "Evenweave", "advanced (over one, magnifier helpful)", "widely available (Lugana 25)", 1, "tapestry 28"),
    Fabric("Evenweave 28 over one", 28, 1, "Evenweave", "advanced (over one, magnifier helpful)", "widely available (Cashel/Brittney 28)", 1, "tapestry 28"),
    Fabric("Linen 32 over one", 32, 1, "Evenweave", "expert (over one, magnifier required)", "widely available (Belfast 32)", 1, "tapestry 28/petite"),
    Fabric("Linen 36 over one", 36, 1, "Evenweave", "expert (over one, magnifier required)", "specialist (Edinburgh 36)", 1, "petite 28/30"),
    Fabric("Linen 40 over one", 40, 1, "Evenweave", "extreme (over one, not recommended)", "specialist (Newcastle 40)", 1, "petite 30"),
)


@dataclass(frozen=True)
class GridPlan:
    fabric: Fabric
    cols: int
    rows: int
    req_w_in: float
    req_h_in: float

    @property
    def actual_w_in(self) -> float:
        return self.cols / self.fabric.stitches_per_inch

    @property
    def actual_h_in(self) -> float:
        return self.rows / self.fabric.stitches_per_inch

    @property
    def total_cells(self) -> int:
        return self.cols * self.rows

    def as_dict(self) -> dict:
        return {"fabric": asdict(self.fabric), "cols": self.cols, "rows": self.rows,
                "total_cells": self.total_cells,
                "requested_in": [self.req_w_in, self.req_h_in],
                "actual_in": [round(self.actual_w_in, 3), round(self.actual_h_in, 3)],
                "actual_cm": [round(self.actual_w_in * CM_PER_IN, 2), round(self.actual_h_in * CM_PER_IN, 2)]}


def to_inches(w: float, h: float, units: str) -> tuple[float, float]:
    units = units.lower().strip()
    if units in ("in", "inch", "inches", '"'):
        return float(w), float(h)
    if units in ("cm", "centimeter", "centimeters"):
        return w / CM_PER_IN, h / CM_PER_IN
    raise ValueError(f"unsupported units {units!r}")


def fit_grid(src_w: int, src_h: int, req_w_in: float, req_h_in: float, fabric: Fabric) -> GridPlan:
    """Largest integer stitch grid inside the requested box that keeps the source
    aspect ratio. Never exceeds the requested size; the shorter side gets margin."""
    if req_w_in <= 0 or req_h_in <= 0:
        raise ValueError("requested size must be positive")
    spi = fabric.stitches_per_inch
    max_c = math.floor(req_w_in * spi + 1e-9)
    max_r = math.floor(req_h_in * spi + 1e-9)
    aspect = src_w / src_h
    if max_c / max_r <= aspect:   # width-limited
        cols = max_c
        rows = min(max_r, max(1, round(cols / aspect)))
    else:                          # height-limited
        rows = max_r
        cols = min(max_c, max(1, round(rows * aspect)))
    return GridPlan(fabric, cols, rows, req_w_in, req_h_in)


def size_table(cols: int, rows: int) -> list[dict]:
    """Finished size of the same chart on other fabrics."""
    rows_out = []
    for name, spi in (("11-count Aida", 11), ("14-count Aida", 14), ("16-count Aida", 16),
                      ("18-count Aida", 18), ("20-count Aida", 20), ("22-count Aida", 22),
                      ("28-count evenweave over two", 14), ("32-count linen over two", 16),
                      ("25-count evenweave over one", 25), ("28-count evenweave over one", 28),
                      ("32-count linen over one", 32)):
        rows_out.append({"fabric": name, "w_in": cols / spi, "h_in": rows / spi,
                         "w_cm": cols / spi * CM_PER_IN, "h_cm": rows / spi * CM_PER_IN})
    return rows_out
