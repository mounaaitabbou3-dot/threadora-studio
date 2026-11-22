"""Five-gate quality report shown before download.

Every line is a measured value or a test that actually ran on this order. There is
deliberately no overall numeric score: a single number would hide which gate failed
and has no independently evaluated rubric behind it.
"""
from __future__ import annotations

import re

GATES = [
    ("technical", "Technical", "Files build, fonts embed, every page renders on A4, source unchanged."),
    ("size", "Physical size", "Stitch count, fabric and printed size labels agree with the request."),
    ("design", "Design fidelity", "Stitches follow the source: no lost details, holes, broken outlines or invented stitches."),
    ("dmc", "DMC and matrix consistency", "Legend, symbols, chart colors and preview all rebuild the same master matrix."),
    ("print", "Print readability", "Symbols readable at 100 %, no label collisions, every cell on exactly one chart."),
]

# order test name -> gate
TEST_GATE = {
    "test_source_hash_unchanged": "technical", "test_fonts_embedded": "technical",
    "test_render_every_page_a4_not_blank_not_clipped": "technical",
    "test_pages_order_and_titles": "technical", "test_review_copy_is_marked_on_every_page": "technical",
    "test_matrix_dimensions_and_physical_size": "size", "test_physical_size_labels_truthful": "size",
    "test_size_table_marks_exactly_the_chosen_fabric": "size",
    "test_fidelity_gates_recorded": "design", "test_design_audit_recomputed_from_source": "design",
    "test_dmc_codes_names_valid_and_all_used": "dmc", "test_legend_counts_exact": "dmc",
    "test_palette_colors_distinguishable": "dmc", "test_pdf_rebuilds_master_exactly": "dmc",
    "test_chart_cell_colors_match_master": "dmc", "test_floss_amounts_calculated": "dmc",
    "test_symbols_one_to_one_and_contrast": "print", "test_tiles_each_cell_exactly_once": "print",
    "test_no_label_collisions_and_labels_correct": "print", "test_symbol_legibility_at_100_percent": "print",
    "test_text_does_not_overlap_on_text_pages": "print",
    "test_overview_is_clean_and_navigation_complete": "print",
}


def parse_test_log(text: str) -> dict[str, str]:
    """pytest -rA short summary lines: 'PASSED tests/test_order.py::test_x' -> {test_x: PASSED}."""
    out = {}
    for m in re.finditer(r"^(PASSED|FAILED|ERROR|SKIPPED)\s+\S*::(test_\w+(?:\[[^\]\s]*\])?)", text or "", re.M):
        out[m.group(2)] = m.group(1)
    return out


def build(audit: dict, test_log: str | None) -> dict:
    results = parse_test_log(test_log) if test_log is not None else {}
    da = audit.get("design_audit", {}) or {}
    fm = audit.get("fidelity_metrics", {}) or {}
    gates = {}
    for key, label, what in GATES:
        tests = {t: s for t, s in results.items() if TEST_GATE.get(t.split("[")[0]) == key}
        gates[key] = {"label": label, "checks": what, "tests": tests, "measured": {}, "risks": []}
    # unit tests on the engine itself are technical
    for t, s in results.items():
        if t.split("[")[0] not in TEST_GATE:
            gates["technical"]["tests"][t] = s

    g = gates["technical"]["measured"]
    g.update({"pdf_pages": audit.get("pdf_pages"), "pdf_megabytes": round((audit.get("pdf_bytes") or 0) / 1e6, 2),
              "tests_summary": (audit.get("tests") or {}).get("summary", "not run")})
    if (audit.get("pdf_bytes") or 0) > 20e6:
        gates["technical"]["risks"].append("PDF is larger than 20 MB.")

    g = gates["size"]["measured"]
    fin, req = audit.get("finished_in"), audit.get("requested_in")
    g.update({"A_stitched_design_in": fin, "B_maximum_box_in": req,
              "fabric": audit.get("display_fabric"), "fabric_practical": audit.get("fabric_practical")})
    gates["size"]["risks"] += list(audit.get("size_problems") or [])
    if audit.get("fabric_practical") is False:
        gates["size"]["risks"].append(f"{audit.get('display_fabric')} is not a practical fabric for most stitchers.")

    g = gates["design"]["measured"]
    for k in ("missing_stitches", "extra_stitches", "spurious_components", "lost_details", "accidental_holes",
              "source_gaps_filled", "broken_contours", "lost_white_cells", "white_cells_dark_thread",
              "bleed_cells", "confetti_cells"):
        if k in da:
            g[k] = da[k]
    g.update({"ssim": fm.get("ssim"), "mean_delta_e": fm.get("mean_de")})
    tl = fm.get("text_lines") or []
    if tl:
        g["lettering_lines_readable"] = f"{sum(1 for t in tl if t.get('pass'))} of {len(tl)}"
    gates["design"]["risks"] += list(da.get("hard_fail") or [])

    g = gates["dmc"]["measured"]
    g.update({"dmc_colors": audit.get("dmc_colors"), "near_duplicate_threads": da.get("near_duplicate_threads")})

    review = list(audit.get("review_reasons") or [])
    for r in review:
        low = r.lower()
        key = ("print" if any(w in low for w in ("symbol", "print", "legib")) else
               "dmc" if any(w in low for w in ("thread", "dmc", "color", "colour")) else
               "size" if any(w in low for w in ("fabric", "size", "inch")) else "design")
        if r not in gates[key]["risks"]:
            gates[key]["risks"].append(r)

    for key, gate in gates.items():
        failed = [t for t, s in gate["tests"].items() if s in ("FAILED", "ERROR")]
        if failed:
            gate["result"] = "FAIL"
            gate["risks"].insert(0, "Failed tests: " + ", ".join(failed))
        elif not gate["tests"] and key != "design":
            gate["result"] = "NOT TESTED"
        elif key == "design" and da.get("hard_fail"):
            gate["result"] = "FAIL"
        elif gate["risks"]:
            gate["result"] = "REVIEW"
        else:
            gate["result"] = "PASS"
    return {"status": audit.get("status"), "gates": gates,
            "tests_executed": len(results),
            "tests_passed": sum(1 for s in results.values() if s == "PASSED"),
            "score": None,
            "note": "No overall score is given. Each gate shows its own measured values and tests."}


def to_markdown(rep: dict) -> str:
    lines = [f"# Quality report — {rep['status']}", "",
             f"Tests executed on this order: {rep['tests_executed']} ({rep['tests_passed']} passed). "
             + rep["note"], ""]
    for gate in rep["gates"].values():
        lines += [f"## {gate['label']}: {gate['result']}", gate["checks"], ""]
        for k, v in gate["measured"].items():
            if v is not None:
                lines.append(f"- {k.replace('_', ' ')}: {v}")
        if gate["tests"]:
            ok = sum(1 for s in gate["tests"].values() if s == "PASSED")
            lines.append(f"- tests: {ok} of {len(gate['tests'])} passed")
        for r in gate["risks"]:
            lines.append(f"- RISK: {r}")
        lines.append("")
    return "\n".join(lines)
