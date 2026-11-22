"""Measure the engine on a fixed set of designs and sizes, for before/after comparison.

    python tools/benchmark.py OUT_DIR [--real IMAGE:NAME:W:H[:phrase|name|year] ...] [--quick]

Writes OUT_DIR/benchmark.json and OUT_DIR/benchmark.md. Every number is measured on the
generated master matrix / PDF; nothing is scored by hand. Synthetic designs come from
tests/designs.py so any engine version can be measured on the same inputs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.designs import DESIGNS, save  # noqa: E402
from threadora.pipeline import run_order  # noqa: E402


def measure(out: Path, audit: dict, seconds: float) -> dict:
    row = {"status": audit.get("status"), "seconds": round(seconds, 1)}
    ch = audit.get("chosen") or {}
    row.update({"fabric": ch.get("fabric"), "grid": ch.get("grid"), "dmc_colors": audit.get("dmc_colors"),
                "pdf_pages": audit.get("pdf_pages"), "finished_in": audit.get("finished_in"),
                "ssim": ch.get("ssim"), "mean_de": ch.get("mean_de"), "edge_f1": ch.get("edge_f1"),
                "tests": (audit.get("tests") or {}).get("summary"),
                "review_reasons": audit.get("review_reasons")})
    da = audit.get("design_audit") or {}
    for k in ("extra_stitches", "missing_stitches", "accidental_holes", "source_gaps_filled",
              "color_bleed_cells", "isolated_stitches", "unsupported_confetti", "lost_details",
              "broken_contours", "sub_stitch_outline_breaks", "lost_white_cells", "white_cells_dark_thread",
              "uniform_break_rate", "near_duplicate_threads"):
        row[k] = da.get(k)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--real", nargs="*", default=[])
    ap.add_argument("--designs", nargs="*", default=sorted(DESIGNS))
    ap.add_argument("--size", type=float, default=5.0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for d in a.designs:
        jobs.append((d, save(d, out), a.size, a.size, DESIGNS[d][1]))
    for spec in a.real:
        parts = spec.split(":")
        img, name, w, h = parts[0], parts[1], float(parts[2]), float(parts[3])
        pers = {}
        if len(parts) > 4:
            ph, nm, yr = (parts[4].split("|") + ["", "", ""])[:3]
            pers = {"phrase": ph, "name": nm, "year": yr}
        jobs.append((name, Path(img), w, h, pers))
    results = {}
    for name, img, w, h, pers in jobs:
        t0 = time.time()
        d = out / f"{name}_{w:g}x{h:g}"
        try:
            audit = run_order(img, w, h, "in", pers, d, log=lambda *_: None, run_tests=True)
            results[f"{name} {w:g}x{h:g}"] = measure(d, audit, time.time() - t0)
        except Exception as e:  # record, never hide
            results[f"{name} {w:g}x{h:g}"] = {"status": f"ERROR {type(e).__name__}: {e}"}
        print(name, results[f"{name} {w:g}x{h:g}"].get("status"), flush=True)
    (out / "benchmark.json").write_text(json.dumps(results, indent=2, default=str))
    cols = ["status", "fabric", "grid", "dmc_colors", "pdf_pages", "ssim", "mean_de", "extra_stitches",
            "missing_stitches", "accidental_holes", "color_bleed_cells", "unsupported_confetti",
            "lost_details", "source_gaps_filled", "broken_contours", "lost_white_cells", "near_duplicate_threads", "tests"]
    md = ["| design | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for k, r in results.items():
        md.append(f"| {k} | " + " | ".join(str(r.get(c, "")).replace("THREADORA ", "") for c in cols) + " |")
    (out / "benchmark.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
