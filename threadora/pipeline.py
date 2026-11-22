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
from .dmc import DATA_SHA256, dataset_sha256, dmc_lab, load_dmc
from .engine import select, text_cells
from .features import detect_features
from .fidelity import CellSource, audit as design_audit, defect_overlay
from .pdf import PatternPDF
from .sizing import (FABRICS, CM_PER_IN, physical_size, size_label, size_problems, size_table,
                     to_inches)
from .source import content_bbox, load_source, sha256_file
from .symbols import assign_symbols

VERIFIED = "THREADORA VERIFIED — COMPLETE"
REVIEW = "THREADORA REVIEW_NOT_VERIFIED"
CONFLICT = "THREADORA SIZE-FIDELITY CONFLICT"
FAILURE = "THREADORA TECHNICAL FAILURE"
STATUSES = (VERIFIED, REVIEW, CONFLICT, FAILURE)

SKEIN_STRAND_MM = 8000 * 6   # one 8 m skein = 48 m of single strand


def skeins_for(count: int, spi: float, strands: int) -> int:
    cell_mm = 25.4 / spi
    per_stitch = cell_mm * (2 * math.sqrt(2) + 2) * 1.25 * strands
    return max(1, math.ceil(count * per_stitch * 1.15 / SKEIN_STRAND_MM))


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_") or "order"


def fabric_by_name(name: str):
    for f in FABRICS:
        if f.name == name:
            return f
    raise ValueError(f"unknown fabric {name!r}; choose one of: {', '.join(f.name for f in FABRICS)}")


def background_risk(src) -> str | None:
    """Opaque images: the background is removed by flood fill from the border. If many
    design pixels along that cut line are nearly the background color, light details
    touching the background may have been removed with it -> needs a human look."""
    if src.has_alpha or src.opaque.all():
        return None
    import cv2
    bg = ~src.opaque
    rgb = src.rgba[..., :3].astype(float)
    bgc = np.median(rgb[bg], axis=0)
    edge = src.opaque & (cv2.dilate(bg.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    if not edge.any():
        return None
    near = float((np.linalg.norm(rgb[edge] - bgc, axis=1) < 40).mean())
    if near > 0.25:
        return (f"{near:.0%} of the design outline is almost the background color, so light details touching "
                "the background may have been removed with it. A PNG with a transparent background makes "
                "this certain.")
    return None


def decide_status(test_ok, size_errs, fidelity_ok, fabric_practical, review_reasons) -> str:
    """Status precedence. VERIFIED only when every mandatory gate passed and nothing
    is uncertain."""
    if test_ok is False or size_errs:
        return FAILURE
    if not fidelity_ok or not fabric_practical:
        return CONFLICT
    if review_reasons or test_ok is None:
        return REVIEW
    return VERIFIED


def _write(priv: Path, audit: dict):
    (priv / "audit.json").write_text(json.dumps(audit, indent=2, default=str))


def run_order(source_path, width: float, height: float, units: str, personalization: dict,
              out_dir, display_fabric: str | None = None, log=print, run_tests: bool = True) -> dict:
    """display_fabric: optional fabric chosen by the owner (a FABRICS name). The stitch grid
    is then fitted to THAT fabric at the requested size; the size is never changed."""
    t0 = time.time()
    out_dir = Path(out_dir)
    priv = out_dir / "private"
    priv.mkdir(parents=True, exist_ok=True)
    req_w, req_h = to_inches(width, height, units)
    if req_w <= 0 or req_h <= 0:
        raise ValueError("requested size must be positive")
    forced = fabric_by_name(display_fabric) if display_fabric else None
    src = load_source(source_path)
    log(f"source sha256={src.sha256} {src.width}x{src.height} {src.mode} type={src.source_type}")
    audit = {"engine_version": __version__, "started": datetime.now(timezone.utc).isoformat(),
             "source": {"path": str(source_path), "sha256": src.sha256, "bytes": src.size_bytes,
                        "size": [src.width, src.height], "mode": src.mode,
                        "exif_orientation": src.exif_orientation, "has_alpha": src.has_alpha,
                        "background": src.background_method, "type": src.source_type,
                        "type_evidence": src.type_evidence},
             "request": {"width": width, "height": height, "units": units,
                         "inches": [req_w, req_h], "personalization": personalization,
                         "fabric": forced.name if forced else "automatic"},
             "dmc_dataset_sha256": dataset_sha256()}
    (priv / "source_sha256.txt").write_text(f"{src.sha256}  {Path(source_path).name}\n")
    if src.source_type == "B":
        audit["status"] = FAILURE
        audit["blocker"] = ("Source is an existing printed chart. Reconstructing it needs its legend; "
                            "symbol-to-DMC mapping is never guessed.")
        _write(priv, audit)
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
    sel = select(rgb, opaque, white, feat, req_w, req_h, log=log,
                 fabrics=(forced,) if forced else FABRICS)
    audit["fabric_trials"] = sel.fabric_trials
    audit["palette_trials"] = sel.palette_trials
    if sel.best is None:
        audit["status"] = CONFLICT
        audit["blocker"] = sel.alternative
        if forced:
            log("chosen fabric failed; looking for the size that would work on it")
            auto = select(rgb, opaque, white, feat, req_w, req_h, log=lambda *_: None, k_values=(26,))
            if auto.best is not None:
                g = auto.best.plan
                need = max(g.cols / forced.stitches_per_inch / req_w, g.rows / forced.stitches_per_inch / req_h)
                aw, ah = req_w * need, req_h * need
                audit["blocker"] = (f"On {forced.label} the lettering and details are not kept at "
                                    f"{req_w:g} × {req_h:g} in. Alternatives without changing your order: "
                                    f"Automatic fabric ({g.fabric.label}) at {req_w:g} × {req_h:g} in, or "
                                    f"{forced.label} at about {math.ceil(aw * 2) / 2:g} × "
                                    f"{math.ceil(ah * 2) / 2:g} in.")
        _write(priv, audit)
        return audit
    best = sel.best
    cols, rows = best.plan.cols, best.plan.rows
    fab = best.plan.fabric
    fin_w, fin_h = physical_size(cols, rows, fab)
    size_errs = size_problems(cols, rows, fab, req_w, req_h, x1 - x0, y1 - y0)
    alternatives = [{"fabric": f.label, "w_in": round(cols / f.stitches_per_inch, 2),
                     "h_in": round(rows / f.stitches_per_inch, 2)} for f in FABRICS if f.practical]
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
    dmc_order = [best.dmc_idx[i] for i in order]
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
              "chart_resolution_fabric": fab.name,
              "physical_in": [round(fin_w, 4), round(fin_h, 4)],
              "physical_cm": [round(fin_w * CM_PER_IN, 3), round(fin_h * CM_PER_IN, 3)],
              "requested_in": [req_w, req_h],
              "palette_codes": codes, "palette": palette,
              "layers": {"full_cross": "cells", "backstitch": [], "fractional": []},
              "settings": {"method": best.method, "coverage_threshold": best.coverage_t,
                           "kmeans_k": best.k, "confetti_removed": best.confetti_removed,
                           "repairs": best.metrics.get("design", {}).get("repairs"),
                           "dmc_dataset_sha256": DATA_SHA256},
              "cells": [[codes[v] if v >= 0 else "NO_STITCH" for v in row] for row in m.tolist()]}
    (priv / "master_matrix.json").write_text(json.dumps(master))
    pal_rgb = np.array([p["rgb"] for p in palette], np.uint8)
    # ---- design-fidelity audit of the LOCKED master against the original
    cs = CellSource(rgb, opaque, rows, cols, dmc_lab()[dmc_order])
    final_design = design_audit(m, cs, best.opened_mask, text_cells(feat, rows, cols))
    final_design["protected_open_cells"] = [[int(r), int(c)] for r, c in zip(*np.where(best.opened_mask))] \
        if best.opened_mask is not None else []
    (priv / "design_audit.json").write_text(json.dumps(final_design, indent=2))
    Image.fromarray(defect_overlay(m, pal_rgb, final_design)).save(priv / "design_audit_overlay.png")
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
        "requested_size": (f"{width:g} × {height:g} {units}" +
                           (f" ({req_w:.2f} × {req_h:.2f} in)" if units != "in" else "") +
                           " — design fitted inside, aspect ratio kept"),
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
        "size_table": size_table(cols, rows, fab),
        "size_caveats": [
            "Finished size = stitches ÷ stitches per inch. Real fabric counts vary slightly by brand and lot,",
            "so allow a few percent difference. The stitch grid is identical on every fabric above; only the",
            f"finished size changes. This pattern was sized for {fab.label} at the requested size.",
        ],
    }
    pdf_path = out_dir / f"THREADORA_{slug(name)}_{size_label(fin_w, fin_h)}.pdf"
    builder = PatternPDF(pdf_path, meta, m, palette, src.rgba[y0:y1, x0:x1])
    nx = max(t.tc for t in builder.tiles) + 1
    ny = max(t.tr for t in builder.tiles) + 1
    meta["chart_grid"] = f"{ny} rows × {nx} columns of charts, about {math.ceil(cols / nx)}×{math.ceil(rows / ny)} stitches each"
    chart_info = builder.build()
    (priv / "chart_info.json").write_text(json.dumps(chart_info, indent=2))
    # ---- review reasons: where automated confidence is not sufficient
    review = list(final_design["review"])
    missing_tokens = [n for n in feat.notes if n.startswith("Personalization tokens not machine-readable")]
    if missing_tokens:
        review.append("Some personalization text could not be read by OCR in the original (script or "
                      "decorative lettering); it was checked by shape only. " + missing_tokens[0].split(":", 1)[1].strip())
    bg_note = background_risk(src)
    if bg_note:
        review.append(bg_note)
    audit.update({
        "chosen": best.summary(), "fidelity_metrics": best.metrics,
        "design_audit": final_design,
        "chart_resolution_fabric": fab.name, "display_fabric": fab.name,
        "fabric_practical": fab.practical, "finished_in": [fin_w, fin_h], "within_request": not size_errs,
        "size_problems": size_errs,
        "alternatives_same_chart": alternatives, "pdf": str(pdf_path),
        "pdf_bytes": pdf_path.stat().st_size, "pdf_pages": chart_info["pages"],
        "dmc_colors": len(palette), "source_sha256_after": sha256_file(source_path),
        "review_reasons": review,
    })
    # ---- tests on the generated files
    _write(priv, audit)
    test_ok = None
    if run_tests:
        env_cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                   str(Path(__file__).resolve().parent.parent / "tests" / "test_order.py"),
                   str(Path(__file__).resolve().parent.parent / "tests" / "test_unit.py"),
                   f"--order-dir={out_dir}", f"--source={source_path}", "-rA"]
        r = subprocess.run(env_cmd, capture_output=True, text=True)
        (priv / "test_log.txt").write_text(r.stdout + "\n" + r.stderr)
        test_ok = r.returncode == 0
        audit["tests"] = {"returncode": r.returncode, "summary": r.stdout.strip().splitlines()[-1] if r.stdout else ""}
    fidelity_ok = bool(best.metrics["pass"]) and final_design["pass"]
    status = decide_status(test_ok, size_errs, fidelity_ok, fab.practical, review)
    if status == FAILURE:
        blocker = ("Automated tests failed; see private/test_log.txt." if test_ok is False
                   else "Size engine error: " + "; ".join(size_errs))
    elif status == CONFLICT and not fidelity_ok:
        blocker = "Chosen output does not keep the critical details: " + \
                  "; ".join(final_design["hard_fail"] or ["fidelity gates failed"])
    elif status == CONFLICT:
        blocker = (f"Lettering and details only pass at {cols}×{rows} stitches, which fits "
                   f"{req_w:g}×{req_h:g} in only on {fab.label} ({fab.stitchability}). "
                   "That is not a realistic fabric for most stitchers.")
    elif status == REVIEW:
        blocker = ("Technical checks passed, but automated confidence is not sufficient. Look at these "
                   "points before sending: " + " | ".join(review if test_ok is not None
                                                          else review + ["automated tests were not run"]))
    else:
        blocker = ""
    audit["status"], audit["blocker"] = status, blocker
    audit["seconds"] = round(time.time() - t0, 1)
    _write(priv, audit)
    log(f"STATUS: {status} {blocker}")
    return audit
