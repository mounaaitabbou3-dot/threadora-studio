"""Candidate search: fabric -> conversion method -> palette size, gated by fidelity."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .convert import (METHODS, NO_STITCH, build_palette, cell_fields, dominant_assign,
                      isolated_count, remove_confetti)
from .dmc import ciede2000_matrix, dmc_lab, load_dmc, nearest_dmc, to_lab
from .features import Features, evaluate
from .fidelity import (CellSource, audit as design_audit, palette_votes, repair as design_repair,
                       repair_confetti, repair_contours)
from .sizing import FABRICS, GridPlan, fit_grid

STITCH_COVERAGE = (0.5, 0.42)  # coverage thresholds tried (0.42 protects thin strokes)
COUNTER_MIN_COVER = 0.25       # a letter counter covering >= 25% of a cell keeps that cell open


def protect_counters(stitch: np.ndarray, opaque: np.ndarray, feat: Features) -> tuple[np.ndarray, int]:
    """Contrast-aware negative-space protection: every significant counter (enclosed
    gap) of the source lettering keeps at least its best-covered cell unstitched."""
    import cv2
    from .features import source_holes
    h, w = opaque.shape
    rows, cols = stitch.shape
    out = stitch.copy()
    opened = 0
    for l in feat.text_lines:
        x0, y0, x1, y1 = l.box
        for hole in source_holes(opaque[y0:y1, x0:x1], y1 - y0):
            full = np.zeros((h, w), np.float64)
            full[y0:y1, x0:x1] = hole
            cov = cv2.resize(full, (cols, rows), interpolation=cv2.INTER_AREA)
            r, c = np.unravel_index(cov.argmax(), cov.shape)
            if cov[r, c] >= COUNTER_MIN_COVER and out[r, c]:
                out[r, c] = False
                opened += 1
    return out, opened


@dataclass
class Candidate:
    plan: GridPlan
    method: str
    coverage_t: float
    k: int
    dmc_idx: list[int]
    matrix: np.ndarray
    confetti_removed: int
    isolated_kept: int
    metrics: dict
    palette_de_mean: float
    opened_mask: np.ndarray | None = None

    @property
    def needs_review(self) -> bool:
        return bool(self.metrics.get("review"))

    @property
    def passed(self) -> bool:
        return bool(self.metrics["pass"])

    @property
    def n_colors(self) -> int:
        return len(self.dmc_idx)

    def summary(self) -> dict:
        g = self.metrics["global"]
        return {"fabric": self.plan.fabric.name, "grid": [self.plan.cols, self.plan.rows],
                "method": self.method, "coverage_t": self.coverage_t, "k": self.k,
                "colors": self.n_colors, "pass": self.passed,
                "ssim": round(g["ssim_blurred"], 4), "mean_de": round(g["mean_de2000"], 2),
                "edge_f1": round(g["edge_f1"], 3), "mask_iou": round(g["mask_iou"], 3),
                "text": [(t["expected"] or t["kind"], t["pass"], t["preview_ocr"],
                          t["components_src"], t["components_matrix"], t["holes_src"],
                          t["holes_matrix"], t["mask_iou"]) for t in self.metrics["text_lines"]],
                "ornaments_pass": self.metrics["ornaments"]["pass"],
                "blobs": [self.metrics["blobs"]["kept"], self.metrics["blobs"]["checked"]],
                "confetti_removed": self.confetti_removed,
                "design_hard_fail": self.metrics.get("design", {}).get("hard_fail", []),
                "design_review": self.metrics.get("review", [])}


def protect_mask(feat: Features, shape_src, rows, cols) -> np.ndarray:
    h, w = shape_src
    sx, sy = w / cols, h / rows
    m = np.zeros((rows, cols), bool)
    for l in feat.text_lines:
        x0, y0, x1, y1 = l.box
        m[int(y0 / sy):int(np.ceil(y1 / sy)), int(x0 / sx):int(np.ceil(x1 / sx))] = True
    for (x0, y0, x1, y1, _a) in feat.ornaments:
        m[int(y0 / sy):int(np.ceil(y1 / sy)), int(x0 / sx):int(np.ceil(x1 / sx))] = True
    for b in feat.blobs:
        r0, r1 = int((b.y - b.r) / sy), int(np.ceil((b.y + b.r) / sy))
        c0, c1 = int((b.x - b.r) / sx), int(np.ceil((b.x + b.r) / sx))
        m[max(r0, 0):r1, max(c0, 0):c1] = True
    return m


def lettering_threads(rgb: np.ndarray, feat: Features) -> list[int]:
    """One DMC thread per text line: nearest_dmc of the median color of its letters."""
    out = []
    for l in feat.text_lines:
        x0, y0, x1, y1 = l.box
        m = feat.text_mask[y0:y1, x0:x1]
        px = rgb[y0:y1, x0:x1][m]
        if len(px) == 0:
            out.append(-1)
            continue
        med = np.median(px, axis=0).astype(np.uint8)
        i, _ = nearest_dmc(to_lab(med[None]))
        out.append(int(i[0]))
    return out


def unify_lettering(matrix: np.ndarray, feat: Features, shape_src, pal_lab, targets=None,
                    max_de: float = 25.0):
    """Lettering in the source is one thread color per word with rendered stitch
    shading. Cells of a text line close to that line's majority color take it, so
    letters read as one clean color (fewest practical colors, no lost letters)."""
    import cv2
    if feat.text_mask is None:
        return matrix
    rows, cols = matrix.shape
    h, w = shape_src
    tm = cv2.resize(feat.text_mask.astype(np.float64), (cols, rows), interpolation=cv2.INTER_AREA) >= 0.5
    pd = ciede2000_matrix(pal_lab, pal_lab)
    out = matrix.copy()
    sx, sy = w / cols, h / rows
    for l in feat.text_lines:
        x0, y0, x1, y1 = l.box
        sl = (slice(int(y0 / sy), int(np.ceil(y1 / sy))), slice(int(x0 / sx), int(np.ceil(x1 / sx))))
        sub = out[sl]
        m = tm[sl] & (sub >= 0)
        if not m.any():
            continue
        vals, cnt = np.unique(sub[m], return_counts=True)
        maj = vals[cnt.argmax()]
        li = feat.text_lines.index(l)
        if targets is not None and targets[li] >= 0:
            maj = targets[li]
        close = m & (pd[maj][np.where(sub >= 0, sub, 0)] < max_de)
        sub[close] = maj
    return out


def text_cells(feat: Features, rows: int, cols: int) -> np.ndarray:
    """Stitch cells touched by detected lettering (letters have their own gates)."""
    import cv2
    if feat.text_mask is None:
        return np.zeros((rows, cols), bool)
    return cv2.resize(feat.text_mask.astype(np.float64), (cols, rows), interpolation=cv2.INTER_AREA) > 0


def make_candidate(rgb, opaque, src_white, feat, plan, method, cov_t, k, log, counters=True) -> Candidate:
    rows, cols = plan.rows, plan.cols
    f = cell_fields(rgb, opaque, cols, rows, "edge_aware" if method == "edge_aware" else "area")
    base_stitch = f.coverage >= cov_t
    stitch, opened = base_stitch, 0
    if counters:
        stitch, opened = protect_counters(base_stitch, opaque, feat)
    opened_mask = base_stitch & ~stitch
    prot = protect_mask(feat, opaque.shape, rows, cols)
    weights = np.where(prot, 4.0, 1.0)
    letter_dmc = lettering_threads(rgb, feat)
    pr = build_palette(f, stitch, k, weights, forced_dmc=[d for d in letter_dmc if d >= 0])
    dl = dmc_lab()
    pal_lab = dl[pr.dmc_idx]
    matrix = pr.matrix
    votes = palette_votes(rgb, opaque, rows, cols, pal_lab)
    if method == "dominant":
        matrix = np.where(stitch, votes.argmax(-1), NO_STITCH).astype(np.int32)
    targets = [pr.dmc_idx.index(d) if d in pr.dmc_idx else -1 for d in letter_dmc]
    matrix = unify_lettering(matrix, feat, opaque.shape, pal_lab, targets)
    # source-supported repair: color bleeding at boundaries, accidental pinholes, lost details,
    # unsupported confetti, broken source lines
    tcells = text_cells(feat, rows, cols)
    cs_full = CellSource(rgb, opaque, rows, cols, pal_lab, votes)
    plain, removed0, kept0 = remove_confetti(matrix, pal_lab, prot)
    fixed, repaired = design_repair(matrix, cs_full, opened_mask, tcells)
    fixed, removed, kept = remove_confetti(fixed, pal_lab, prot)
    fixed, unsup_fixed = repair_confetti(fixed, cs_full, tcells)
    repaired["unsupported_confetti_recolored"] = unsup_fixed
    fixed, joined = repair_contours(fixed, cs_full, tcells)
    repaired["contour_cells_joined"] = joined

    def finish(mat):
        used = sorted(set(np.unique(mat[mat >= 0]).tolist()))
        m2 = np.full(mat.shape, NO_STITCH, np.int32)
        for i, u in enumerate(used):
            m2[mat == u] = i
        dmc_idx = [pr.dmc_idx[u] for u in used]
        pal_rgb = np.array([load_dmc()[i].rgb for i in dmc_idx], dtype=np.uint8)
        metrics = evaluate(feat, src_white, opaque, m2, pal_rgb)
        cs = CellSource(rgb, opaque, rows, cols, pal_lab[used], votes[..., used])
        da = design_audit(m2, cs, opened_mask, tcells)
        metrics["design"] = da
        metrics["review"] = list(da["review"])
        metrics["critical_pass"] = bool(metrics["pass"])          # lettering, ornaments, blobs
        metrics["pass"] = bool(metrics["pass"] and da["pass"])
        return m2, dmc_idx, metrics

    m2, dmc_idx, metrics = finish(fixed)
    if not metrics["critical_pass"]:
        # never accept a repair that damages a critical region (lettering, ornaments, eyes):
        # if the unrepaired matrix keeps them, use it instead
        m_plain, idx_plain, met_plain = finish(plain)
        if met_plain["critical_pass"]:
            m2, dmc_idx, metrics = m_plain, idx_plain, met_plain
            removed, kept = removed0, kept0
            repaired = {"reverted": "repairs damaged a critical region; unrepaired matrix kept"}
    metrics["design"]["repairs"] = repaired
    c = Candidate(plan, method, cov_t, k, dmc_idx, m2, removed, kept, metrics, pr.mean_de)
    c.opened_mask = opened_mask
    log(f"  cand {plan.fabric.name} {cols}x{rows} {method} cov={cov_t} k={k} -> {c.n_colors} colors "
        f"pass={c.passed} review={len(metrics['review'])} ssim={metrics['global']['ssim_blurred']:.3f} dE={metrics['global']['mean_de2000']:.2f}")
    return c


@dataclass
class Selection:
    status: str                      # "OK" | "SIZE_FIDELITY_CONFLICT"
    best: Candidate | None
    fabric_trials: list[dict] = field(default_factory=list)
    palette_trials: list[dict] = field(default_factory=list)
    alternative: str = ""
    review_best: "Candidate | None" = None   # best failing candidate, for an owner review copy


def select(rgb, opaque, src_white, feat, req_w_in, req_h_in, log=print,
           k_probe: int = 26, k_values=(10, 12, 14, 16, 18, 20, 22, 24, 26, 30),
           fabrics=FABRICS) -> Selection:
    """Fabric and stitch grid are chosen JOINTLY for the requested size: each fabric
    gets its own grid fitted inside the requested box (fit_grid), so the finished
    size is always stitches / stitches-per-inch <= requested size."""
    h, w = opaque.shape
    sel = Selection("SIZE_FIDELITY_CONFLICT", None)
    chosen_plan = None
    probe_best: dict = {}
    # Stage 1: the simplest fabric whose grid passes every critical gate.
    for fab in fabrics:
        plan = fit_grid(w, h, req_w_in, req_h_in, fab)
        log(f"fabric {fab.name}: grid {plan.cols}x{plan.rows}")
        best = None
        for method in METHODS:
            for cov_t in STITCH_COVERAGE:
                c = make_candidate(rgb, opaque, src_white, feat, plan, method, cov_t, k_probe, log)
                sel.fabric_trials.append(c.summary())
                if sel.review_best is None or (c.metrics["global"]["ssim_blurred"]
                                               > sel.review_best.metrics["global"]["ssim_blurred"]):
                    sel.review_best = c
                if c.passed and (best is None or (c.needs_review, -c.metrics["global"]["ssim_blurred"])
                                 < (best.needs_review, -best.metrics["global"]["ssim_blurred"])):
                    best = c
        sel.fabric_trials.append({"fabric": fab.name, "grid": [plan.cols, plan.rows], "fabric_pass": best is not None})
        if best is not None:
            chosen_plan = plan
            probe_best = {"method": best.method, "cov_t": best.coverage_t}
            break
    if chosen_plan is None:
        # find what WOULD work, to propose an alternative without changing the order
        names = ", ".join(f.name for f in fabrics)
        sel.alternative = (f"No tested fabric ({names}) keeps the lettering and details at the "
                           "requested size. Choose a larger finished size.")
        return sel
    # Stage 2: fewest colors that still pass and stay close to the best quality.
    cands = []
    for k in k_values:
        for method in dict.fromkeys([probe_best["method"], *METHODS]):
            c = make_candidate(rgb, opaque, src_white, feat, chosen_plan, method,
                               probe_best["cov_t"], k, log)
            sel.palette_trials.append(c.summary())
            if c.passed:
                cands.append(c)
    best_ssim = max(c.metrics["global"]["ssim_blurred"] for c in cands)
    best_de = min(c.metrics["global"]["mean_de2000"] for c in cands)
    ok = [c for c in cands if c.metrics["global"]["ssim_blurred"] >= best_ssim - 0.01
          and c.metrics["global"]["mean_de2000"] <= best_de + 0.75]
    ok.sort(key=lambda c: (c.needs_review, c.n_colors, -c.metrics["global"]["ssim_blurred"]))
    sel.best = ok[0]
    sel.status = "OK"
    return sel
