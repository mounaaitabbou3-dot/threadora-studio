"use strict";
// THREADORA private studio — talks only to the local FastAPI backend.
const $ = (id) => document.getElementById(id);
const VERIFIED = "THREADORA VERIFIED — COMPLETE";
const CONFLICT = "THREADORA SIZE-FIDELITY CONFLICT";
const REVIEW = "THREADORA REVIEW_NOT_VERIFIED";
let token = "";
try { token = localStorage.getItem("threadora_token") || ""; } catch (e) { /* storage blocked */ }
let options = null;
let size = "5";
let pollTimer = null;

function withToken(url) {
  return token ? url + (url.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(token) : url;
}

async function api(path, init = {}) {
  init.headers = Object.assign({}, init.headers || {}, token ? { "X-Threadora-Token": token } : {});
  const r = await fetch(path, init);
  if (r.status === 401) {
    const t = window.prompt("This THREADORA studio is private. Enter the access token:");
    if (t) { token = t.trim(); try { localStorage.setItem("threadora_token", token); } catch (e) {} return api(path, init); }
    throw new Error("Access token required.");
  }
  let body = null;
  try { body = await r.json(); } catch (e) { /* not json */ }
  if (!r.ok) throw new Error((body && body.detail) || `Request failed (${r.status}).`);
  return body;
}

function statusClass(status) {
  return status === VERIFIED ? "ok" : (status === CONFLICT || status === REVIEW) ? "warn" : "bad";
}
function fmt(n, d = 2) { return Number(n).toFixed(d).replace(/\.?0+$/, ""); }
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ------------------------------------------------------------------ setup
async function init() {
  try {
    const h = await api("/api/health");
    const problems = [];
    if (!h.dmc_dataset_ok) problems.push("DMC dataset failed its integrity check");
    if (!h.ocr.available) problems.push(h.ocr.detail);
    $("health").textContent = problems.length ? "⚠ " + problems.join(" · ")
      : `Engine v${h.engine_version} ready · DMC data verified · OCR ready · offline`;
    $("health").classList.toggle("bad", problems.length > 0);
  } catch (e) { $("health").textContent = "⚠ " + e.message; $("health").classList.add("bad"); }

  options = await api("/api/options");
  const chips = $("sizes");
  [...options.presets_in.map(String), "custom"].forEach((v) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "chip"; b.setAttribute("role", "radio"); b.dataset.size = v;
    b.textContent = v === "custom" ? "Custom" : `${v}×${v} in`;
    b.addEventListener("click", () => pickSize(v));
    chips.appendChild(b);
  });
  options.fabrics.forEach((f) => {
    const o = document.createElement("option");
    o.value = f.name;
    o.textContent = `${f.label} · ${f.stitches_per_inch} st/in${f.practical ? "" : " (expert only)"}`;
    $("fabric").appendChild(o);
  });
  pickSize("5");
  ["width", "height", "units"].forEach((id) => $(id).addEventListener("input", sizeHint));
  loadHistory();
}

function pickSize(v) {
  size = v;
  document.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-checked", String(c.dataset.size === v)));
  $("custom").hidden = v !== "custom";
  sizeHint();
}

function sizeHint() {
  if (size !== "custom") {
    const n = Number(size);
    $("size-hint").textContent = `${n} × ${n} in (${fmt(n * 2.54, 1)} × ${fmt(n * 2.54, 1)} cm). ` +
      "The design is fitted inside this area with its proportions kept; it is never stretched or cropped.";
  } else {
    const u = $("units").value;
    $("size-hint").textContent = `Custom ${$("width").value || "?"} × ${$("height").value || "?"} ${u}. ` +
      "The design is fitted inside this rectangle with its proportions kept.";
  }
}

// ------------------------------------------------------------------ upload
const fileInput = $("image");
function showThumb(file) {
  if (!file) return;
  const url = URL.createObjectURL(file);
  $("thumb").src = url; $("thumb").hidden = false;
  $("drop-text").innerHTML = `<strong>${esc(file.name)}</strong><br><small>${fmt(file.size / 1024 / 1024, 2)} MB · click to change</small>`;
}
fileInput.addEventListener("change", () => showThumb(fileInput.files[0]));
const drop = $("drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) { fileInput.files = e.dataTransfer.files; showThumb(fileInput.files[0]); }
});

$("order-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("form-error").textContent = "";
  const file = fileInput.files[0];
  if (!file) { $("form-error").textContent = "Choose the original image first."; return; }
  const fd = new FormData();
  fd.append("image", file);
  fd.append("size", size);
  if (size === "custom") {
    fd.append("width", $("width").value); fd.append("height", $("height").value); fd.append("units", $("units").value);
  }
  ["phrase", "name", "year"].forEach((k) => fd.append(k, $(k).value));
  fd.append("display_fabric", $("fabric").value);
  $("go").disabled = true;
  try {
    const job = await api("/api/jobs", { method: "POST", body: fd });
    track(job.id);
  } catch (err) {
    $("form-error").textContent = err.message;
  } finally {
    $("go").disabled = false;
  }
});

// ------------------------------------------------------------------ progress
function track(id) {
  clearTimeout(pollTimer);
  $("order-card").hidden = true; $("result-card").hidden = true; $("progress-card").hidden = false;
  $("progress-title").textContent = "Generating your pattern…";
  $("log").textContent = "";
  const tick = async () => {
    try {
      const j = await api(`/api/jobs/${encodeURIComponent(id)}`);
      $("log").textContent = j.log.join("\n");
      $("log").scrollTop = $("log").scrollHeight;
      if (j.state === "done" || j.state === "interrupted") { showResult(j); loadHistory(); return; }
      $("progress-title").textContent = j.state === "queued" ? "Waiting for the previous order to finish…" : "Generating your pattern…";
    } catch (err) { $("log").textContent += "\n" + err.message; }
    pollTimer = setTimeout(tick, 2000);
  };
  tick();
}

function showResult(j) {
  const r = j.result || {};
  $("progress-card").hidden = true; $("result-card").hidden = false; $("order-card").hidden = true;
  const st = $("status");
  st.textContent = r.status || "THREADORA TECHNICAL FAILURE";
  st.className = "status " + statusClass(r.status);
  $("blocker").textContent = r.blocker || (r.verified ? "All automated tests and detail checks passed." : "");

  const q = r.quality_report;
  $("quality").innerHTML = q ? `<h3>${esc(r.title || "")}</h3><h3>Quality report</h3>
    <p class="hint">${q.tests_executed} automated tests ran on this order, ${q.tests_passed} passed. No overall score is given: each gate shows its own measurements.</p>
    <ul>${Object.values(q.gates).map((g) => `<li><b>${esc(g.label)}: ${esc(g.result)}</b>${
      g.risks.length ? "<br>" + g.risks.map(esc).join("<br>") : ""}</li>`).join("")}</ul>` : "";

  const dl = $("dl");
  if (r.pdf_name) {
    dl.hidden = false;
    dl.href = withToken(`/api/jobs/${encodeURIComponent(j.id)}/pdf`);
    dl.setAttribute("download", r.pdf_name);
    dl.textContent = r.verified ? "Download PDF" : "Download review copy (not verified)";
    dl.classList.toggle("review", !r.verified);
    $("dl-note").textContent = r.verified ? `${r.pdf_name} · ${r.pdf_mb} MB`
      : "Do not send this copy to a customer as final.";
  } else { dl.hidden = true; $("dl-note").textContent = "No PDF was produced."; }

  const stats = [];
  const req = j.request;
  stats.push(["B · Maximum box requested", req.size === "custom" ? `${fmt(req.width)} × ${fmt(req.height)} ${req.units}` : `${req.width} × ${req.height} in`]);
  if (r.finished_in) stats.push(["A · Stitched design (exact)", `${fmt(r.finished_in[0])} × ${fmt(r.finished_in[1])} in (${fmt(r.finished_cm[0], 1)} × ${fmt(r.finished_cm[1], 1)} cm)`]);
  if (r.stitches) stats.push(["Stitch grid", `${r.stitches.cols} × ${r.stitches.rows} (${r.stitches.total.toLocaleString("en-US")} stitches)`]);
  if (r.fabric) stats.push(["Fabric", r.fabric + (r.fabric_practical === false ? " (expert only)" : "")]);
  if (r.dmc_colors != null) stats.push(["DMC colors", r.dmc_colors]);
  if (r.pdf_pages != null) stats.push(["PDF pages", r.pdf_pages]);
  if (r.tests) stats.push(["Automated tests", r.tests.summary || (r.tests.returncode === 0 ? "passed" : "failed")]);
  stats.push(["Source SHA-256", j.source_sha256.slice(0, 16) + "…"]);
  $("stats").innerHTML = stats.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("");

  $("src-img").src = withToken(`/api/jobs/${encodeURIComponent(j.id)}/image/source`);
  if (r.has_preview) { $("prev-img").src = withToken(`/api/jobs/${encodeURIComponent(j.id)}/image/preview`); $("prev-img").closest("figure").hidden = false; }
  else $("prev-img").closest("figure").hidden = true;

  const alts = r.alternatives_same_chart || [];
  $("alts").innerHTML = alts.length && !r.verified ? `<h3>Same verified chart on other fabrics</h3>
    <p class="hint">The stitch grid stays the same; only the finished size changes. Nothing is changed without your decision.</p>
    <ul>${alts.map((a) => `<li>${esc(a.fabric)}: ${a.w_in} × ${a.h_in} in</li>`).join("")}</ul>` : "";

  const tb = $("legend").querySelector("tbody");
  tb.innerHTML = (r.palette || []).map((p) => {
    const rgb = p.hex.match(/\w\w/g).map((h) => parseInt(h, 16));
    const lum = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]);
    return `<tr><td><span class="swatch" style="background:${esc(p.hex)};color:${lum > 140 ? "#000" : "#fff"}">${esc(p.symbol)}</span></td>
      <td>${esc(p.symbol)}</td><td>${esc(p.code)}</td><td>${esc(p.name)}</td>
      <td class="n">${p.count.toLocaleString("en-US")}</td><td class="n">${p.meters ?? ""}</td><td class="n">${p.skeins}</td></tr>`;
  }).join("");
  $("legend").closest(".table-wrap").previousElementSibling.hidden = !(r.palette || []).length;
  $("log2").textContent = (j.log || []).join("\n");
}

$("again").addEventListener("click", () => {
  $("result-card").hidden = true; $("order-card").hidden = false; window.scrollTo(0, 0);
});

// ------------------------------------------------------------------ history
async function loadHistory() {
  try {
    const jobs = await api("/api/jobs");
    if (!jobs.length) return;
    $("history").innerHTML = jobs.map((j) => {
      const req = j.request;
      const sz = req.size === "custom" ? `${fmt(req.width)}×${fmt(req.height)} ${req.units}` : `${req.width}×${req.height} in`;
      const st = j.status ? `<span class="tag ${statusClass(j.status)}">${esc(j.status.replace("THREADORA ", ""))}</span>`
        : `<span class="tag">${esc(j.state)}</span>`;
      return `<li><span><a data-id="${esc(j.id)}">${esc(j.source_name)}</a> · ${esc(sz)} · <small>${esc(j.created.replace("T", " ").slice(0, 16))} UTC</small></span>${st}</li>`;
    }).join("");
    $("history").querySelectorAll("a[data-id]").forEach((a) => a.addEventListener("click", async () => {
      const j = await api(`/api/jobs/${encodeURIComponent(a.dataset.id)}?log_lines=400`);
      if (j.state === "done" || j.state === "interrupted") showResult(j); else track(j.id);
      window.scrollTo(0, 0);
    }));
  } catch (e) { /* history is optional */ }
}

init();
