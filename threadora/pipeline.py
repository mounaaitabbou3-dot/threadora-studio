"""End-to-end order pipeline: image + size -> one PDF + private audit files."""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from . import __version__
from .convert import render_matrix
from .dmc import DATA_SHA256, dataset_sha256, load_dmc
from .engine import select
from .features import detect_features
from .pdf import PatternPDF
from .sizing import FABRICS, CM_PER_IN, fit_grid, size_table, to_inches
from .source import content_bbox, load_source, sha256_file
from .symbols import assign_symbols

SKEIN_STRAND_MM = 8000 * 6   # one 8 m skein = 48 m of single strand


def skeins_for(count: int, spi: float, strands: int) -> int:
    cell_mm = 25.4 / spi
    per_stitch = cell_mm * (2 * math.sqrt(2) + 2) * 1.25 * strands
    return max(1, math.ceil(count * per_stitch * 1.15 / SKEIN_STRAND_MM))


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_") or "order"


def run_order(source_path, width: float, height: float, units: str, personalization: dict,
              out_dir, display_fabric: str | None = None, log=print, run_tests: bool = True) -> dict:
    t0 = time.time()
    out_dir = Path(out_dir)
    priv = out_dir / "private"
    priv.mkdir(parents=True, exist_ok=True)
    req_w, req_h = to_inches(width, height, units)
    src = load_source(source_path)
    log(f"source sha256={src.sha256} {src.width}x{src.height} {src.mode} type={src.source_type}")
    audit = {"engine_version": __version__, "started": datetime.now(timezone.utc).isoformat(),
             "source": {"path": str(source_path), "sha256": src.sha256, "bytes": src.size_bytes,
                        "size": [src.width, src.height], "mode": src.mode,
                        "exif_orientation": src.exif_orientation, "has_alpha": src.has_alpha,
                        "background": src.background_method, "type": src.source_type,
                        "type_evidence": src.type_evidence},
             "request": {"width": width, "height": height, "units": units,
                         "inches": [req_w, req_h], "personalization": personalization},
             "dmc_dataset_sha256": dataset_sha256()}
    (priv / "source_sha256.txt").write_text(f"{src.sha256}  {Path(source_path).name}\n")
    if src.source_type == "B":
        audit["status"] = "THREADORA TECHNICAL FAILURE"
        audit["blocker"] = ("Source is an existing printed chart. Reconstructing it needs its legend; "
                            "symbol-to-DMC mapping is never guessed.")
        (priv / "audit.json").write_text(json.dumps(audit, indent=2, default=str))
        return audit
    x0, y0, x1, y1 = content_bbox(src.opaque)
    audit["content_bbox"] = [x0, y0, x1, y1]
    rgb = src.rgba[y0:y1, x0:x1, :3].copy()
    opaque = src.opaque[y0:y1, x0:x1]
    white = src.composite_on()[y0:y1, x0:x1]
    texts = [v for v in personalization.values() if v]
    feat = detect_features(white, opaque, texts)
    audit["features"] = feat.as_dict()
    log(f"features: {len(feat.text_lines)} text lines, {len(feat.ornaments)} ornaments, {len(feat.blobs)} blobs")
    sel = select(rgb, opaque, white, feat, req_w, req_h, log=log)
    audit["fabric_trials"] = sel.fabric_trials
    audit["palette_trials"] = sel.palette_trials
    if sel.best is None:
        audit["status"] = "THREADORA SIZE-FIDELITY CONFLICT"
        audit["blocker"] = sel.alternative
        (priv / "audit.json").write_text(json.dumps(audit, indent=2, default=str))
        return audit
    best = sel.best
    cols, rows = best.plan.cols, best.plan.rows
    chart_fabric = best.plan.fabric
    fab = next(f for f in FABRICS if f.name == display_fabric) if display_fabric else chart_fabric
    fin_w, fin_h = cols / fab.stitches_per_inch, rows / fab.stitches_per_inch
    within = fin_w <= req_w + 1e-6 and fin_h <= req_h + 1e-6
    alternatives = []
    for f in FABRICS:
        if f.practical:
            alternatives.append({"fabric": f.label, "w_in": round(cols / f.stitches_per_inch, 2),
                                 "h_in": round(rows / f.stitches_per_inch, 2)})
    # ---- master matrix (single source of truth)
    flosses = load_dmc()
    pal_floss = [flosses[i] for i in best.dmc_idx]
    m = best.matrix
    counts = [int((m == i).sum()) for i in range(len(pal_floss))]
    order = sorted(range(len(pal_floss)), key=lambda i: -counts[i])
    remap = np.full(len(pal_floss), -1)
    for new, old in enumerate(order):
        remap[old] = new
    m = np.where(m >= 0, remap[np.maximum(m, 0)], -1).astype(np.int32)
    pal_floss = [pal_floss[i] for i in order]
    counts = [counts[i] for i in order]
    symbols = assign_symbols(len(pal_floss))
    palette = [{"code": f.code, "name": f.name, "rgb": list(f.rgb), "hex": f.hex, "symbol": s,
                "count": c, "skeins": skeins_for(c, fab.stitches_per_inch, fab.strands)}
               for f, s, c in zip(pal_floss, symbols, counts)]
    codes = [p["code"] for p in palette]
    master = {"format": "threadora-master-1", "rows": rows, "cols": cols,
              "source_sha256": src.sha256, "content_bbox": [x0, y0, x1, y1],
              "fabric": {"name": fab.name, "label": fab.label, "count": fab.count, "over": fab.over,
                         "stitches_per_inch": fab.stitches_per_inch},
              "chart_resolution_fabric": chart_fabric.name,
              "physical_in": [round(fin_w, 4), round(fin_h, 4)],
              "physical_cm": [round(fin_w * CM_PER_IN, 3), round(fin_h * CM_PER_IN, 3)],
              "requested_in": [req_w, req_h],
              "palette_codes": codes, "palette": palette,
              "layers": {"full_cross": "cells", "backstitch": [], "fractional": []},
              "settings": {"method": best.method, "coverage_threshold": best.coverage_t,
                           "kmeans_k": best.k, "confetti_removed": best.confetti_removed,
                           "dmc_dataset_sha256": DATA_SHA256},
              "cells": [[codes[v] if v >= 0 else "NO_STITCH" for v in row] for row in m.tolist()]}
    (priv / "master_matrix.json").write_text(json.dumps(master))
    pal_rgb = np.array([p["rgb"] for p in palette], np.uint8)
    preview = render_matrix(m, pal_rgb, scale=4)
    Image.fromarray(preview).save(priv / "stitched_preview_from_matrix.png")
    srcim = Image.fromarray(white).resize((cols * 4, rows * 4), Image.LANCZOS)
    cmp_img = Image.new("RGB", (cols * 8 + 20, rows * 4), "white")
    cmp_img.paste(srcim, (0, 0)); cmp_img.paste(Image.fromarray(preview), (cols * 4 + 20, 0))
    cmp_img.save(priv / "comparison_source_vs_matrix.png")
    # ---- PDF
    name = personalization.get("name") or "Pattern"
    title = " · ".join(v for v in (personalization.get("phrase"), name, personalization.get("year")) if v)
    meta = {
        "project_name": title,
        "title": title,
        "subtitle": "Counted cross stitch pattern · full color + symbol charts",
        "finished_size": f"{fin_w:.2f} × {fin_h:.2f} in ({fin_w * CM_PER_IN:.1f} × {fin_h * CM_PER_IN:.1f} cm)",
        "requested_size": f"{req_w:g} × {req_h:g} in (design fitted inside, aspect ratio kept)",
        "total_stitches": int((m >= 0).sum()),
        "fabric_label": f"{fab.label} ({fab.stitches_per_inch:g} stitches per inch)",
        "strands": fab.strands,
        "chart_grid": "",
        "materials": [
            f"Fabric: {fab.label}, white or antique white. Cut at least "
            f"{fin_w + 6:.1f} × {fin_h + 6:.1f} in ({(fin_w + 6) * 2.54:.0f} × {(fin_h + 6) * 2.54:.0f} cm) "
            "to leave 3 in on every side.",
            f"Threads: {len(palette)} DMC six-strand cotton floss colors (see legend; one skein each is enough "
            "unless noted).",
            f"Needle: {fab.needle}. Hoop or scroll frame, embroidery scissors, good light"
            + (" and a magnifier." if fab.over == 1 and fab.count >= 25 else "."),
        ],
        "guidance": [
            "Find the center: fold the fabric in half both ways. The chart center is column "
            f"{cols // 2 + 1}, row {rows // 2 + 1}.",
            f"Use {fab.strands} strand(s) of floss for every full cross stitch"
            + (", stitched over ONE fabric thread." if fab.kind == "Evenweave" else "."),
            "Each chart square with a symbol = one full cross stitch in that DMC color. Blank squares are "
            "not stitched.",
            "Work all bottom legs in the same direction (/) and cross them the same way (\\) for an even finish.",
            "Bold lines mark every 10 stitches; the numbers on the charts are global column/row numbers.",
            "Each chart's own stitches sit inside its red border. Pale cells outside it repeat 2 rows/columns of "
            "the neighbouring chart",
            "for alignment only; stitch them from their own chart, never twice.",
            "Wash gently by hand, press face-down on a towel, and frame.",
        ],
        "size_table": [dict(r, chosen=(r["fabric"].startswith(f"{fab.count}-count") and
                                       (("over one" in r["fabric"]) == (fab.over == 1 and fab.kind == "Evenweave"))))
                       for r in size_table(cols, rows)],
        "size_caveats": [
            "Finished size = stitches ÷ stitches per inch. Real fabric counts vary slightly by brand and lot,",
            "so allow a few percent difference. Lettering legibility was verified for this stitch grid; the grid",
            "is identical on every fabric above, only the physical size changes.",
        ],
    }
    pdf_path = out_dir / f"THREADORA_{slug(name)}_{slug(f'{req_w:g}x{req_h:g}in')}.pdf"
    builder = PatternPDF(pdf_path, meta, m, palette, src.rgba[y0:y1, x0:x1])
    nx = max(t.tc for t in builder.tiles) + 1
    ny = max(t.tr for t in builder.tiles) + 1
    meta["chart_grid"] = f"{ny} rows × {nx} columns of charts, about {math.ceil(cols / nx)}×{math.ceil(rows / ny)} stitches each"
    chart_info = builder.build()
    (priv / "chart_info.json").write_text(json.dumps(chart_info, indent=2))
    audit.update({
        "chosen": best.summary(), "fidelity_metrics": best.metrics,
        "chart_resolution_fabric": chart_fabric.name, "display_fabric": fab.name,
        "fabric_practical": fab.practical, "finished_in": [fin_w, fin_h], "within_request": within,
        "alternatives_same_chart": alternatives, "pdf": str(pdf_path),
        "pdf_bytes": pdf_path.stat().st_size, "pdf_pages": chart_info["pages"],
        "dmc_colors": len(palette), "source_sha256_after": sha256_file(source_path),
    })
    # ---- tests on the generated files
    (priv / "audit.json").write_text(json.dumps(audit, indent=2, default=str))
    test_ok = None
    if run_tests:
        env_cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                   str(Path(__file__).resolve().parent.parent / "tests"),
                   f"--order-dir={out_dir}", f"--source={source_path}", "-rA"]
        r = subprocess.run(env_cmd, capture_output=True, text=True)
        (priv / "test_log.txt").write_text(r.stdout + "\n" + r.stderr)
        test_ok = r.returncode == 0
        audit["tests"] = {"returncode": r.returncode, "summary": r.stdout.strip().splitlines()[-1] if r.stdout else ""}
    fidelity_ok = bool(best.metrics["pass"])
    if test_ok is False:
        status = "THREADORA TECHNICAL FAILURE"
        blocker = "Automated tests failed; see private/test_log.txt."
    elif not within or not fidelity_ok:
        status = "THREADORA SIZE-FIDELITY CONFLICT"
        blocker = "Chosen output does not keep the requested size with verified detail."
    elif not fab.practical:
        status = "THREADORA SIZE-FIDELITY CONFLICT"
        blocker = (f"Lettering and details only pass at {cols}×{rows} stitches, which fits "
                   f"{req_w:g}×{req_h:g} in only on {fab.label} ({fab.stitchability}). "
                   "That is not a realistic fabric for most stitchers.")
    else:
        status = "THREADORA VERIFIED — COMPLETE" if test_ok else "UNTESTED"
        blocker = ""
    audit["status"], audit["blocker"] = status, blocker
    audit["seconds"] = round(time.time() - t0, 1)
    (priv / "audit.json").write_text(json.dumps(audit, indent=2, default=str))
    log(f"STATUS: {status} {blocker}")
    return audit
