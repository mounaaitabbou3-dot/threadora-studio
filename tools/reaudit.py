"""Re-measure existing order folders with the CURRENT design audit, so outputs made by an older
engine version can be compared with new ones on the same checks.

    python tools/reaudit.py ORDER_DIR [ORDER_DIR ...]

Prints one JSON line per order with the V13 counts. The background mask comes from the current
source loader, and lettering protection is not applied, so counts are an approximation of what
the engine itself records.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from threadora.dmc import dmc_by_code, to_lab
from threadora.fidelity import CellSource, audit
from threadora.source import load_source

KEYS = ("extra_stitches", "missing_stitches", "accidental_holes", "source_gaps_filled", "color_bleed_cells",
        "unsupported_confetti", "lost_details", "broken_contours", "sub_stitch_outline_breaks",
        "lost_white_cells", "white_cells_dark_thread", "near_duplicate_threads")


def reaudit(order_dir: Path) -> dict:
    priv = order_dir / "private"
    m = json.loads((priv / "master_matrix.json").read_text())
    a = json.loads((priv / "audit.json").read_text())
    codes = m["palette_codes"]
    idx = {c: i for i, c in enumerate(codes)}
    mat = np.array([[idx.get(v, -1) for v in row] for row in m["cells"]], np.int32)
    by = dmc_by_code()
    pal = to_lab(np.array([by[c].rgb for c in codes], np.uint8))
    src = load_source(a["source"]["path"])
    x0, y0, x1, y1 = m["content_bbox"]
    cs = CellSource(src.rgba[y0:y1, x0:x1, :3], src.opaque[y0:y1, x0:x1], m["rows"], m["cols"], pal)
    res = audit(mat, cs)
    return {k: res.get(k) for k in KEYS}


if __name__ == "__main__":
    for d in sys.argv[1:]:
        try:
            print(json.dumps({"order": d, **reaudit(Path(d))}))
        except Exception as e:  # report, never hide
            print(json.dumps({"order": d, "error": f"{type(e).__name__}: {e}"}))
