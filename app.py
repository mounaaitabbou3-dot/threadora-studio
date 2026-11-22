"""THREADORA Studio — private Streamlit web app (Streamlit Community Cloud entry point).

Upload the original image, choose the finished size, click Generate, download the PDF.
All pattern work is done by the tested THREADORA engine (threadora.pipeline.run_order):
one master stitch matrix, DMC matching, color + symbol PDF, and the automated order tests.
No Claude or other AI service is called.

Privacy: the app refuses to run until a password is configured in Streamlit secrets
(THREADORA_PASSWORD) and entered by the visitor. Use it together with Community Cloud's
private-app viewer list.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import shutil
import tempfile
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from threadora import __version__ as ENGINE_VERSION
from threadora.dmc import DATA_SHA256, dataset_sha256
from threadora.features import ocr_status
from threadora.pipeline import run_order
from threadora.sizing import CM_PER_IN, FABRICS
from webapp.jobs import CONFLICT, FAILURE, REVIEW, VERIFIED, safe_filename, summarize

PRESETS_IN = (3, 5, 7, 9, 12, 16, 20)
MIN_IN, MAX_IN = 1.0, 60.0
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}
Image.MAX_IMAGE_PIXELS = 80_000_000

st.set_page_config(page_title="THREADORA Studio", page_icon="✕", layout="centered",
                   initial_sidebar_state="collapsed")

CSS = """
<style>
#MainMenu, footer, [data-testid="stToolbar"] {visibility: hidden;}
.block-container {max-width: 860px; padding-top: 2rem;}
.td-brand {font-weight: 700; letter-spacing: .14em; font-size: 1.15rem;}
.td-brand span {color: #8c2f39;}
.td-sub {color: #6f655d; font-size: .9rem; margin-top: -4px;}
.td-status {font-weight: 700; letter-spacing: .04em; font-size: 1.25rem; padding: 12px 16px;
            border-radius: 10px; margin: 6px 0 10px;}
.td-ok {background: #e3f3ea; color: #1f6b45;}
.td-warn {background: #fdf0d5; color: #8a5a00;}
.td-bad {background: #fbe3e3; color: #9b1c1c;}
.td-sw {display:inline-block; width:34px; height:22px; border-radius:4px; text-align:center;
        border:1px solid rgba(0,0,0,.25); font-weight:700; line-height:22px;}
table.td-legend {border-collapse: collapse; width: 100%; font-size: .9rem;}
table.td-legend th, table.td-legend td {padding: 5px 8px; border-bottom: 1px solid #e4d9cc; text-align: left;}
table.td-legend td.n, table.td-legend th.n {text-align: right;}
</style>
"""


# ---------------------------------------------------------------- shared resources
@st.cache_resource
def engine_lock() -> threading.Lock:
    """One generation at a time: the engine is CPU and memory heavy."""
    return threading.Lock()


@st.cache_resource
def work_root() -> Path:
    return Path(tempfile.mkdtemp(prefix="threadora_orders_"))


@st.cache_data(ttl=600)
def health() -> dict:
    return {"dmc_ok": dataset_sha256() == DATA_SHA256, "ocr": ocr_status()}


# ---------------------------------------------------------------- access control
def configured_password() -> str:
    try:
        return str(st.secrets.get("THREADORA_PASSWORD", "") or "")
    except Exception:          # no secrets file at all
        return ""


def require_login() -> None:
    pw = configured_password()
    if not pw:
        st.error("This private app has no password configured yet. In Streamlit Community Cloud open "
                 "**App settings → Secrets** and add:  `THREADORA_PASSWORD = \"your-long-password\"`")
        st.stop()
    if st.session_state.get("authed"):
        return
    with st.form("login"):
        st.markdown("#### Private studio")
        given = st.text_input("Password", type="password")
        ok = st.form_submit_button("Enter")
    if ok:
        if hmac.compare_digest(given.encode(), pw.encode()):
            st.session_state.authed = True
            st.rerun()
        time.sleep(1.5)        # slow down guessing
        st.error("Wrong password.")
    st.stop()


# ---------------------------------------------------------------- helpers
def validate_image(raw: bytes) -> str | None:
    if not raw:
        return "The uploaded file is empty."
    try:
        with Image.open(io.BytesIO(raw)) as im:
            fmt = im.format
            im.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return "This file is not a readable image. Upload a PNG, JPG or WEBP."
    if fmt not in ALLOWED_FORMATS:
        return f"{fmt} images are not supported. Upload a PNG, JPG or WEBP."
    return None


def validate_size(w: float, h: float, units: str) -> str | None:
    w_in, h_in = (w, h) if units == "in" else (w / CM_PER_IN, h / CM_PER_IN)
    if not (MIN_IN <= w_in <= MAX_IN and MIN_IN <= h_in <= MAX_IN):
        return (f"Width and height must each be between {MIN_IN:g} and {MAX_IN:g} in "
                f"({MIN_IN * CM_PER_IN:.1f}–{MAX_IN * CM_PER_IN:.0f} cm).")
    return None


def private_bundle(order_dir: Path, source: Path) -> bytes:
    """Owner-only ZIP: master matrix, audit, test log, previews, source hash and original."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted((order_dir / "private").glob("*")):
            z.write(f, f"private/{f.name}")
        z.write(source, f"source/{source.name}")
    return buf.getvalue()


def generate(raw: bytes, filename: str, w: float, h: float, units: str, pers: dict,
             display_fabric: str | None, log) -> dict:
    job_dir = work_root() / f"{datetime.now():%Y%m%d-%H%M%S}-{hashlib.sha256(raw).hexdigest()[:8]}"
    src = job_dir / "source" / safe_filename(filename)
    src.parent.mkdir(parents=True)
    src.write_bytes(raw)                                   # original bytes, unchanged
    sha = hashlib.sha256(raw).hexdigest()
    order_dir = job_dir / "order"
    try:
        audit = run_order(src, w, h, units, pers, order_dir, display_fabric=display_fabric,
                          log=log, run_tests=True)
        res = summarize(audit, order_dir)
        if hashlib.sha256(src.read_bytes()).hexdigest() != sha:
            res.update(status=FAILURE, blocker="Original upload changed during processing.", verified=False)
    except Exception as e:                                 # report, never hide
        log(f"ERROR {type(e).__name__}: {e}")
        res = {"status": FAILURE, "blocker": f"Engine error: {type(e).__name__}: {e}", "verified": False}
    res["source_sha256"] = sha
    res["request"] = {"width": w, "height": h, "units": units}
    pdf = order_dir / res["pdf_name"] if res.get("pdf_name") else None
    res["pdf_bytes"] = pdf.read_bytes() if pdf and pdf.exists() else None
    prev = order_dir / "private" / "stitched_preview_from_matrix.png"
    res["preview_png"] = prev.read_bytes() if prev.exists() else None
    res["bundle_zip"] = private_bundle(order_dir, src) if (order_dir / "private").exists() else None
    res["bundle_name"] = f"THREADORA_private_{job_dir.name}.zip"
    shutil.rmtree(job_dir, ignore_errors=True)             # cloud disk is temporary anyway
    return res


def show_result(res: dict, raw: bytes) -> None:
    status = res.get("status") or FAILURE
    cls = "td-ok" if status == VERIFIED else "td-warn" if status in (CONFLICT, REVIEW) else "td-bad"
    st.markdown(f'<div class="td-status {cls}" data-testid="td-status">{status}</div>', unsafe_allow_html=True)
    if status == REVIEW and res.get("review_reasons"):
        st.write("Technical checks passed, but these points need your eye before the PDF goes to a customer:")
        st.markdown("\n".join(f"- {r}" for r in res["review_reasons"]))
    else:
        st.write(res.get("blocker") or ("All automated tests and detail checks passed." if res.get("verified") else ""))

    if res.get("pdf_bytes"):
        if res.get("verified"):
            st.download_button("⬇ Download PDF", res["pdf_bytes"], file_name=res["pdf_name"],
                               mime="application/pdf", type="primary", key="dl_pdf")
            st.caption(f"{res['pdf_name']} · {res.get('pdf_mb')} MB")
        else:
            st.download_button("Download review copy (not verified)", res["pdf_bytes"],
                               file_name=res["pdf_name"].replace(".pdf", "_NOT_VERIFIED.pdf"),
                               mime="application/pdf", key="dl_review")
            st.caption("Do not send this copy to a customer as final.")
    else:
        st.caption("No PDF was produced.")

    if res.get("stitches"):
        req = res["request"]
        c = st.columns(4)
        c[0].metric("Requested", f"{req['width']:g} × {req['height']:g} {req['units']}")
        fi, fc = res["finished_in"], res["finished_cm"]
        c[1].metric("Finished", f"{fi[0]:.2f} × {fi[1]:.2f} in", f"{fc[0]:.1f} × {fc[1]:.1f} cm", delta_color="off")
        c[2].metric("Stitch grid", f"{res['stitches']['cols']} × {res['stitches']['rows']}",
                    f"{res['stitches']['total']:,} stitches", delta_color="off")
        c[3].metric("DMC colors", res.get("dmc_colors"))
        c = st.columns(3)
        c[0].markdown(f"**Fabric**  \n{res.get('fabric')}" + (" (expert only)" if res.get("fabric_practical") is False else ""))
        c[1].markdown(f"**PDF pages**  \n{res.get('pdf_pages')}")
        t = res.get("tests") or {}
        c[2].markdown(f"**Automated tests**  \n{t.get('summary') or 'not run'}")

    c1, c2 = st.columns(2)
    c1.image(raw, caption="Original (reference)", use_container_width=True)
    if res.get("preview_png"):
        c2.image(res["preview_png"], caption="STITCHED RESULT · rebuilt from the master stitch matrix",
                 use_container_width=True)

    alts = res.get("alternatives_same_chart") or []
    if alts and not res.get("verified"):
        st.markdown("##### Same chart on other fabrics")
        st.caption("The stitch grid stays the same; only the finished size changes. Nothing changes without your decision.")
        st.markdown("\n".join(f"- {a['fabric']}: {a['w_in']} × {a['h_in']} in" for a in alts))

    if res.get("palette"):
        st.markdown("##### DMC legend")
        rows = []
        for p in res["palette"]:
            r, g, b = (int(p["hex"][i:i + 2], 16) for i in (1, 3, 5))
            ink = "#000" if 0.299 * r + 0.587 * g + 0.114 * b > 140 else "#fff"
            rows.append(f'<tr><td><span class="td-sw" style="background:{p["hex"]};color:{ink}">{p["symbol"]}</span></td>'
                        f'<td>{p["code"]}</td><td>{p["name"]}</td><td class="n">{p["count"]:,}</td>'
                        f'<td class="n">{p["skeins"]}</td></tr>')
        st.markdown('<table class="td-legend"><tr><th>Symbol</th><th>DMC</th><th>Name</th><th class="n">Stitches</th>'
                    '<th class="n">Skeins</th></tr>' + "".join(rows) + "</table>", unsafe_allow_html=True)

    if res.get("bundle_zip"):
        st.download_button("Download private audit files (ZIP)", res["bundle_zip"], file_name=res["bundle_name"],
                           mime="application/zip", key="dl_bundle",
                           help="Master stitch matrix JSON, audit, test log, previews and source hash. "
                                "For your records only; do not send to customers.")
    st.caption(f"Source SHA-256: {res.get('source_sha256')}")


# ---------------------------------------------------------------- page
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="td-brand"><span>✕</span> THREADORA</div>'
                '<div class="td-sub">Cross-stitch studio · private</div>', unsafe_allow_html=True)
    require_login()

    hl = health()
    if not hl["dmc_ok"]:
        st.error("The DMC color dataset failed its integrity check. Generation is disabled.")
        st.stop()
    if not hl["ocr"]["available"]:
        st.error("Tesseract OCR is not available, so lettering cannot be checked. Generation is disabled. "
                 f"({hl['ocr']['detail']}) On Streamlit Cloud, make sure packages.txt is in the repository root.")
        st.stop()

    st.markdown("### New pattern")
    up = st.file_uploader("1 · Original image (PNG, JPG or WEBP)", type=["png", "jpg", "jpeg", "webp"],
                          help="Transparent PNG areas stay unstitched. The file is used exactly as uploaded.")
    labels = [f"{n}×{n} in" for n in PRESETS_IN] + ["Custom"]
    choice = st.radio("2 · Finished size", labels, index=1, horizontal=True)
    if choice == "Custom":
        c1, c2, c3 = st.columns([2, 2, 1.4])
        w = c1.number_input("Width", min_value=0.1, max_value=200.0, value=6.0, step=0.5)
        h = c2.number_input("Height", min_value=0.1, max_value=200.0, value=8.0, step=0.5)
        units = c3.selectbox("Units", ["in", "cm"])
    else:
        w = h = float(PRESETS_IN[labels.index(choice)])
        units = "in"
    st.caption("The design is fitted inside this size with its proportions kept. It is never stretched or cropped.")

    with st.expander("3 · Personalization (optional)"):
        st.caption("Type the text exactly as it already appears in the artwork. The engine uses it to find the "
                   "lettering and check it stays readable. It does not rewrite text in the artwork.")
        c1, c2, c3 = st.columns(3)
        phrase = c1.text_input("Phrase", max_chars=120, placeholder="In Loving Memory")
        name = c2.text_input("Name", max_chars=60, placeholder="Cecilia")
        year = c3.text_input("Year / date", max_chars=20, placeholder="2026")
    with st.expander("4 · Fabric (automatic)"):
        st.caption("Leave on automatic. If you choose a fabric, the chart is sized for that fabric at the size "
                   "you asked for. Your size never changes; if the details cannot be kept on that fabric you get "
                   "SIZE-FIDELITY CONFLICT with alternatives.")
        fab_opts = ["Automatic (recommended)"] + [f.name for f in FABRICS]
        fab = st.selectbox("Fabric", fab_opts)
    display_fabric = None if fab == fab_opts[0] else fab

    if st.button("Generate PDF", type="primary", disabled=up is None):
        raw = up.getvalue()
        err = validate_image(raw) or validate_size(w, h, units)
        if err:
            st.error(err)
            st.stop()
        lock = engine_lock()
        if not lock.acquire(blocking=False):
            st.warning("Another pattern is being generated right now. Try again in a few minutes.")
            st.stop()
        try:
            st.session_state.pop("result", None)
            with st.status("Generating your pattern… detailed designs take a few minutes.", expanded=False) as box:
                lines: list[str] = []
                area = st.empty()

                def log(msg):
                    lines.extend(str(msg).splitlines())
                    area.code("\n".join(lines[-12:]), language=None)

                res = generate(raw, up.name, w, h, units,
                               {"phrase": phrase.strip(), "name": name.strip(), "year": year.strip()},
                               display_fabric, log)
                box.update(label=res.get("status") or FAILURE, state="complete" if res.get("verified") else "error")
            st.session_state.result = res
            st.session_state.result_raw = raw
        finally:
            lock.release()

    if st.session_state.get("result"):
        st.divider()
        show_result(st.session_state.result, st.session_state.result_raw)

    st.divider()
    st.caption(f"Engine v{ENGINE_VERSION} · DMC data verified · OCR: {hl['ocr']['detail']} · "
               "No AI service is used to make patterns. DMC colors on screen are approximations. "
               f"Generated {datetime.now(timezone.utc):%Y-%m-%d} UTC.")


main()
