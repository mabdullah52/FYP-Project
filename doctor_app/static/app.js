/* LungScan AI - clinician workstation front-end (plain JS, no build step). */
const $ = (s) => document.querySelector(s);
const esc = (x) => String(x ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const state = { patients: [], patient: null, study: null, z: 0, win: "lung", sel: null, poll: null, imgCache: new Map(),
  camCache: new Map(), user: null, editing: null };

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (r.status === 401 && !path.startsWith("/api/auth/")) { showLogin(false); throw new Error("Please log in."); }
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return r.headers.get("content-type")?.includes("json") ? r.json() : r;
}
const jsonBody = (method, body) => ({ method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const doctor = () => state.user?.name || null;

/* ---------------------------------------------------------------- login */
function showLogin(setup) {
  $("#appLayout").hidden = true; $("#loginView").hidden = false;
  $("#loginTitle").textContent = setup ? "Create the administrator account" : "Sign in";
  $("#loginHint").textContent = setup ? "First start: this account can add doctors and change AI settings. Keep the password safe."
    : "Clinician workstation - authorised staff only.";
  $("#lblFullName").hidden = !setup;
  $("#loginBtn").textContent = setup ? "Create account" : "Sign in";
  $("#loginForm").dataset.setup = setup ? "1" : "";
  $("#userBox").textContent = ""; $("#btnAdmin").hidden = true; $("#btnLogout").hidden = true;
}
function startApp(user) {
  state.user = user;
  $("#loginView").hidden = true; $("#appLayout").hidden = false;
  $("#userBox").textContent = `${user.name} · ${user.role}`;
  $("#btnAdmin").hidden = user.role !== "admin"; $("#btnLogout").hidden = false;
  loadPatients(); refreshHealth();
}
$("#loginForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = Object.fromEntries(new FormData($("#loginForm")));
  const setup = !!$("#loginForm").dataset.setup;
  try {
    const r = await api(setup ? "/api/auth/setup" : "/api/auth/login", jsonBody("POST", setup ? f : { username: f.username, password: f.password }));
    $("#loginMsg").hidden = true; $("#loginForm").reset(); startApp(r.user);
  } catch (err) { $("#loginMsg").hidden = false; $("#loginMsg").textContent = err.message; }
});
$("#btnLogout").onclick = async () => { await api("/api/auth/logout", { method: "POST" }); location.reload(); };
(async () => {
  const st = await api("/api/auth/state");
  if (st.setup_needed) showLogin(true); else if (!st.user) showLogin(false); else startApp(st.user);
})();

async function refreshHealth() {
  try {
    const h = await api("/api/health");
    const p = $("#modelPill");
    if (h.error) { p.textContent = "model error"; p.className = "pill bad"; p.title = h.error; }
    else if (!h.model) { p.textContent = "no trained model"; p.className = "pill bad"; }
    else { p.textContent = `model ${h.state}`; p.className = "pill" + (h.state.startsWith("ready") ? " ok" : ""); p.title = h.model; }
  } catch (_) {}
}
setInterval(() => { if (state.user) refreshHealth(); }, 10000);

/* ---------------------------------------------------------------- views */
function show(view) {
  for (const v of ["#emptyView", "#patientView", "#studyView", "#adminView"]) $(v).hidden = v !== view;
  if (view !== "#studyView" && state.poll) { clearInterval(state.poll); state.poll = null; }
}

/* ---------------------------------------------------------------- patients */
async function loadPatients() {
  state.patients = await api("/api/patients?q=" + encodeURIComponent($("#search").value));
  $("#patientList").innerHTML = state.patients.map((p) => `
    <li data-id="${p.id}" class="${state.patient?.id === p.id ? "on" : ""}">
      <div class="nm">${esc(p.name)}${p.n_open ? ` <span class="newbadge" title="open patient messages">${p.n_open}</span>` : ""}</div>
      <div class="sub">${p.mrn ? "MRN " + esc(p.mrn) + " · " : ""}${p.age ?? "?"} y ${esc(p.sex || "")} · ${p.n_studies} stud${p.n_studies === 1 ? "y" : "ies"}</div>
    </li>`).join("") || `<li class="muted">No patients yet</li>`;
}
$("#patientList").addEventListener("click", (e) => { const li = e.target.closest("li[data-id]"); if (li) openPatient(+li.dataset.id); });
$("#search").addEventListener("input", () => loadPatients());

function patientDialog(p) {
  state.editing = p ? p.id : null;
  const form = $("#formPatient"); form.reset();
  $("#dlgTitle").textContent = p ? "Edit patient" : "New patient";
  $("#dlgOk").textContent = p ? "Save" : "Create";
  if (p) for (const el of form.elements) {
    if (!el.name || p[el.name] == null) continue;
    if (el.type === "checkbox") el.checked = !!p[el.name]; else el.value = p[el.name];
  }
  $("#dlgPatient").showModal();
}
$("#btnNewPatient").onclick = () => patientDialog(null);
$("#btnEditPatient").onclick = () => patientDialog(state.patient);
$("#dlgPatient").addEventListener("close", async () => {
  if ($("#dlgPatient").returnValue !== "ok") return;
  const form = $("#formPatient");
  const f = Object.fromEntries(new FormData(form));
  for (const k of Object.keys(f)) if (f[k] === "") f[k] = null;
  f.age = f.age != null ? +f.age : null;
  f.pack_years = f.pack_years != null ? +f.pack_years : null;
  f.family_history = form.elements.family_history.checked;
  f.emphysema = form.elements.emphysema.checked;
  try {
    const p = state.editing ? await api("/api/patients/" + state.editing, jsonBody("PUT", f)) : await api("/api/patients", jsonBody("POST", f));
    await loadPatients(); openPatient(p.id);
  } catch (e) { alert(e.message); }
});

async function openPatient(id) {
  state.patient = await api("/api/patients/" + id);
  const p = state.patient;
  $("#pName").textContent = p.name;
  $("#pMeta").textContent = [p.mrn && "MRN " + p.mrn, p.age != null && p.age + " years", p.sex, p.phone, p.notes].filter(Boolean).join(" · ");
  $("#pCode").textContent = p.access_code;
  $("#pLife").textContent = "Smoking: " + (p.smoking_status || "unknown") + (p.pack_years ? `, ${p.pack_years} pack-years` : "")
    + " · Family history: " + (p.family_history == null ? "unknown" : p.family_history ? "yes" : "no")
    + " · Emphysema: " + (p.emphysema == null ? "unknown" : p.emphysema ? "yes" : "no");
  $("#uploadMsg").hidden = true;
  renderStudies();
  loadRequests();
  show("#patientView");
  loadPatients();
}

const KIND = { appointment: "Appointment request", question: "Question", symptom: "Symptom report" };
async function loadRequests() {
  const reqs = await api(`/api/patients/${state.patient.id}/requests`);
  $("#reqCard").hidden = !reqs.length;
  $("#reqList").innerHTML = reqs.map((r) => `
    <div class="req ${r.status} ${r.urgency}">
      <div class="top"><b>${KIND[r.kind] || r.kind}</b>
        <span class="muted small">${esc(r.created.replace("T", " "))} · ${r.urgency === "urgent" ? "⚠ URGENT" : esc(r.urgency)}</span></div>
      <p>${esc(r.text)}</p>
      ${r.reply ? `<p class="reply">↳ ${esc(r.reply)} <span class="muted small">— ${esc(r.replied_by || "")}</span></p>`
        : `<div class="row"><input placeholder="Reply to the patient…" data-reply="${r.id}"><button class="btn sm primary" data-send="${r.id}">Send</button></div>`}
    </div>`).join("");
}
$("#reqList").addEventListener("click", async (e) => {
  const b = e.target.closest("[data-send]"); if (!b) return;
  const text = $(`[data-reply="${b.dataset.send}"]`).value.trim(); if (!text) return;
  try { await api(`/api/requests/${b.dataset.send}`, jsonBody("PUT", { reply: text, doctor: doctor() })); loadRequests(); loadPatients(); }
  catch (err) { alert(err.message); }
});

function statusChip(s) { return `<span class="status ${s}">${s}</span>`; }
function renderStudies() {
  const rows = state.patient.studies.map((s) => {
    const sm = s.report?.summary || {};
    return `<tr data-id="${s.id}"><td>${s.id}</td><td>${esc(s.created.replace("T", " "))}</td><td>${statusChip(s.status)}</td>
      <td>${s.status === "done" ? sm.n_nodules_detected : "–"}</td><td>${s.status === "done" && sm.largest_nodule_mm ? sm.largest_nodule_mm.toFixed(1) + " mm" : "–"}</td>
      <td>${s.impression ? "✓ " + esc(s.reviewed_by || "") : "<span class='muted'>pending</span>"}</td><td>${s.released ? "✓" : "–"}</td></tr>`;
  }).join("");
  $("#studyTable tbody").innerHTML = rows || `<tr><td colspan="7" class="muted">No studies yet - upload a CT above.</td></tr>`;
}
$("#studyTable").addEventListener("click", (e) => { const tr = e.target.closest("tr[data-id]"); if (tr) openStudy(+tr.dataset.id); });

/* ---------------------------------------------------------------- upload */
const drop = $("#drop");
$("#btnBrowse").onclick = () => $("#fileInput").click();
$("#fileInput").onchange = (e) => uploadFiles([...e.target.files]);
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); uploadFiles([...e.dataTransfer.files]); });

function uploadMsg(text, err = false) { const m = $("#uploadMsg"); m.hidden = false; m.textContent = text; m.className = "msg" + (err ? " err" : ""); }

function uploadFiles(files) {
  if (!files.length) return;
  const names = files.map((f) => f.name.toLowerCase());
  if (names.some((n) => n.endsWith(".mhd")) && !names.some((n) => n.endsWith(".raw") || n.endsWith(".zraw")))
    return uploadMsg("Select the .mhd AND its .raw file together (hold Ctrl while clicking).", true);
  const fd = new FormData();
  fd.append("patient_id", state.patient.id);
  files.forEach((f) => fd.append("files", f, f.name));
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/studies/upload");
  xhr.upload.onprogress = (e) => e.lengthComputable && uploadMsg(`Uploading… ${Math.round((e.loaded / e.total) * 100)}%`);
  xhr.onload = () => {
    let r = {}; try { r = JSON.parse(xhr.responseText); } catch (_) {}
    if (xhr.status >= 300) return uploadMsg(r.detail || "Upload failed", true);
    openStudy(r.id);
  };
  xhr.onerror = () => uploadMsg("Upload failed (server not reachable)", true);
  uploadMsg("Uploading…");
  xhr.send(fd);
}
$("#btnPath").onclick = async () => {
  const path = $("#pathInput").value.trim();
  if (!path) return;
  try { const r = await api("/api/studies/from_path", jsonBody("POST", { patient_id: state.patient.id, path })); openStudy(r.id); }
  catch (e) { uploadMsg(e.message, true); }
};

/* ---------------------------------------------------------------- study */
$("#btnBack").onclick = () => openPatient(state.study.patient_id);

async function openStudy(id) {
  state.sel = null; state.imgCache.clear();
  show("#studyView");
  await refreshStudy(id, true);
  if (state.poll) clearInterval(state.poll);
  state.poll = setInterval(() => {
    if (state.study && ["queued", "processing", "uploading"].includes(state.study.status)) refreshStudy(id, false);
  }, 2500);
}

async function refreshStudy(id, first) {
  const s = await api("/api/studies/" + id);
  const wasDone = state.study?.id === id && state.study.status === "done";
  state.study = s;
  $("#sTitle").textContent = `${s.patient.name} · CT study #${s.id}`;
  $("#sMeta").textContent = `${s.created.replace("T", " ")} · ${s.source || ""}`;
  $("#btnPrint").href = "/report/" + s.id;
  const busy = ["queued", "processing", "uploading"].includes(s.status);
  $("#processing").hidden = !busy;
  $("#errorBox").hidden = s.status !== "error";
  $("#studyBody").hidden = s.status !== "done";
  $("#btnRelease").disabled = s.status !== "done";
  $("#btnRelease").textContent = s.released ? "Released ✓ (undo)" : "Release to patient";
  if (busy) {
    $("#procTitle").textContent = s.status === "processing" ? "Analysing the scan…" : "Waiting in queue…";
    $("#procText").textContent = s.status === "processing"
      ? "Finding candidates in the lungs and scoring them with the AI model (usually 30–90 s)."
      : `${s.queue_position} study(ies) ahead.`;
  }
  if (s.status === "error") $("#errorBox").innerHTML = `<b>Analysis failed.</b> ${esc(s.error)} <button class="btn sm" id="btnRetry">Retry</button>`;
  if (s.status === "done" && (first || !wasDone)) renderStudy();
}
document.addEventListener("click", async (e) => {
  if (e.target.id === "btnRetry") { await api(`/api/studies/${state.study.id}/reanalyze`, { method: "POST" }); refreshStudy(state.study.id, true); }
});

function radsClass(cat) { return cat.startsWith("2") ? "g" : cat.startsWith("3") ? "w" : cat.startsWith("4A") ? "s" : "c"; }
function radsIcon(cat) { return cat.startsWith("2") ? "●" : cat.startsWith("3") ? "▲" : "◆"; }

function renderStudy() {
  const s = state.study, rep = s.report, sm = rep.summary, g = rep.geometry;
  const nFlag = rep.findings.filter((f) => f.needs_review).length;
  $("#tiles").innerHTML = [
    ["Suspected nodules", sm.n_nodules_detected],
    ["Largest nodule", sm.largest_nodule_mm ? `${sm.largest_nodule_mm.toFixed(1)} <small>mm</small>` : "–"],
    ["Flagged for review", nFlag],
    ["Candidates screened", sm.n_candidates_screened],
    ["Analysis time", `${sm.runtime_s} <small>s</small>`],
  ].map(([k, v]) => `<div class="tile"${k === "Analysis time" && sm.timing ? ` title="${esc(Object.entries(sm.timing).map(([a, b]) => `${a.replace(/_s$/, "").replace(/_/g, " ")}: ${b} s`).join("\n"))}"` : ""}><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
  $("#statement").textContent = sm.statement + " " + sm.disclaimer;
  $("#impression").value = s.impression || "";
  $("#impMsg").textContent = s.reviewed_at ? `saved by ${s.reviewed_by || "?"} at ${s.reviewed_at.replace("T", " ")}` : "";
  renderFindings();
  state.risk = sliceRisk();
  $("#zSlider").max = g.slices - 1;
  const first = rep.findings[0];
  setSlice(first ? first.voxel_zyx[0] : Math.floor(g.slices / 2));
  if (first) state.sel = first.id;
  renderFindings();
}

function renderFindings() {
  const s = state.study, rep = s.report;
  if (!rep.findings.length) { $("#findings").innerHTML = `<p class="muted">No suspected nodules above the threshold. Please still review the scan.</p>`; return; }
  $("#findings").innerHTML = rep.findings.map((f) => {
    const rv = s.reviews?.[f.id] || {};
    const btn = (d, label) => `<button class="btn sm ${d} ${rv.decision === d ? "on" : ""}" data-decide="${d}" data-f="${f.id}">${label}</button>`;
    return `<div class="finding ${state.sel === f.id ? "on" : ""}" data-f="${f.id}">
      <div class="top"><span class="ttl">Finding ${f.id}${f.above_threshold ? "" : " <span class='muted small'>(below threshold)</span>"}</span>
        <span class="badge ${radsClass(f.lung_rads_size_category)}">${radsIcon(f.lung_rads_size_category)} Lung-RADS ${esc(f.lung_rads_size_category)}</span></div>
      <div class="facts">
        <span>Nodule probability <b>${(f.probability * 100).toFixed(1)}%</b><span class="bar"><i style="width:${(f.probability * 100).toFixed(0)}%"></i></span></span>
        <span>Size <b>${f.diameter_mm.toFixed(1)} mm</b></span><span>Slice <b>${f.voxel_zyx[0]}</b></span>
        <span>Uncertainty <b>±${(f.uncertainty * 100).toFixed(1)}%</b></span>
        ${f.cancer_risk != null ? `<span title="Image-based AI estimate (model trained on LUNA25 / NLST screening). Not a diagnosis.">AI cancer risk <b class="risk ${esc(f.cancer_risk_band)}">${(f.cancer_risk * 100).toFixed(1)}% · ${esc(f.cancer_risk_band)}</b></span>` : ""}
        ${f.brock_risk != null ? `<span title="Brock (PanCan) clinical model: age, sex, family history, emphysema + nodule size, type, location, count, spiculation">Brock clinical risk <b class="risk ${esc(f.brock_band)}">${(f.brock_risk * 100).toFixed(1)}% · ${esc(f.brock_band)}</b></span>` : `<span class="muted small">Brock risk: enter the patient's age</span>`}
        ${f.needs_review ? `<span class="flag">⚠ needs your review</span>` : ""}</div>
      <div class="attrs" data-attrs="${f.id}">
        <select data-a="nodule_type">${["solid", "part-solid", "nonsolid"].map((t) => `<option ${f.attrs?.nodule_type === t ? "selected" : ""}>${t}</option>`).join("")}</select>
        <label><input type="checkbox" data-a="upper_lobe" ${f.attrs?.upper_lobe ? "checked" : ""}> upper lobe</label>
        <label><input type="checkbox" data-a="spiculated" ${f.attrs?.spiculated ? "checked" : ""}> spiculated</label>
      </div>
      <div class="decide">${btn("confirm", "✓ Confirm nodule")}${btn("reject", "✕ False positive")}${btn("unsure", "? Unsure")}
        <input placeholder="note (optional)" data-note="${f.id}" value="${esc(rv.note || "")}"></div>
    </div>`;
  }).join("");
}

$("#findings").addEventListener("click", async (e) => {
  const b = e.target.closest("[data-decide]");
  if (b) {
    const fid = +b.dataset.f, note = $(`[data-note="${fid}"]`).value || null;
    try {
      await api(`/api/studies/${state.study.id}/findings/${fid}/review`, jsonBody("PUT", { decision: b.dataset.decide, note, doctor: doctor() }));
      state.study.reviews = state.study.reviews || {};
      state.study.reviews[fid] = { decision: b.dataset.decide, note };
      state.risk = sliceRisk(); drawRiskBar();
      renderFindings();
    } catch (err) { alert(err.message); }
    return;
  }
  if (e.target.closest("input") || e.target.closest("select")) return;
  const card = e.target.closest(".finding");
  if (card) { const f = state.study.report.findings.find((x) => x.id === +card.dataset.f); state.sel = f.id; setSlice(f.voxel_zyx[0]); renderFindings(); }
});

$("#findings").addEventListener("change", async (e) => {
  const box = e.target.closest("[data-attrs]"); if (!box) return;
  const v = (a) => box.querySelector(`[data-a="${a}"]`);
  try {
    const s = await api(`/api/studies/${state.study.id}/findings/${box.dataset.attrs}/attrs`, jsonBody("PUT",
      { nodule_type: v("nodule_type").value, upper_lobe: v("upper_lobe").checked, spiculated: v("spiculated").checked }));
    state.study.report = s.report; renderFindings();
  } catch (err) { alert(err.message); }
});

$("#btnSaveImp").onclick = async () => {
  try {
    await api(`/api/studies/${state.study.id}/impression`, jsonBody("PUT", { impression: $("#impression").value, doctor: doctor() }));
    state.study.impression = $("#impression").value;
    $("#impMsg").textContent = "saved ✓";
  } catch (e) { alert(e.message); }
};
$("#btnReanalyze").onclick = async () => {
  if (!state.study) return;
  await api(`/api/studies/${state.study.id}/reanalyze`, { method: "POST" });
  refreshStudy(state.study.id, true);
};

$("#btnRelease").onclick = async () => {
  const s = state.study;
  try {
    await api(`/api/studies/${s.id}/release?released=${!s.released}`, { method: "POST" });
    await refreshStudy(s.id, false);
    if (state.study.released) alert(`Released. The patient can now see this study in the patient app with code ${s.patient.access_code}.`);
  } catch (e) { alert(e.message); }
};

/* ---------------------------------------------------------------- slice viewer */
const cv = $("#cv"), ctx = cv.getContext("2d");
function sliceUrl(z) { return `/api/studies/${state.study.id}/slice?z=${z}&window=${state.win}`; }
function loadImg(z) {
  const key = state.win + ":" + z;
  if (!state.imgCache.has(key)) {
    const im = new Image(); im.src = sliceUrl(z); state.imgCache.set(key, im);
    if (state.imgCache.size > 120) state.imgCache.delete(state.imgCache.keys().next().value);
  }
  return state.imgCache.get(key);
}
/* slice-level risk: each slice gets the highest nodule probability of the (non-rejected) findings it
   passes through; green < yellow threshold <= yellow < red threshold <= red (proposal: 30% / 70%) */
function sliceRisk() {
  const rep = state.study.report, g = rep.geometry, sz = g.spacing_xyz_mm[2];
  const risk = new Float32Array(g.slices);
  for (const f of rep.findings) {
    if (state.study.reviews?.[f.id]?.decision === "reject") continue;
    const rz = Math.max(1, Math.ceil((Math.max(3, f.diameter_mm / 2)) / sz));
    for (let z = f.voxel_zyx[0] - rz; z <= f.voxel_zyx[0] + rz; z++)
      if (z >= 0 && z < g.slices) risk[z] = Math.max(risk[z], f.probability);
  }
  return risk;
}
function drawRiskBar() {
  const c = $("#riskBar"), g = state.study.report.geometry, st = state.study.settings || { slice_yellow: 0.3, slice_red: 0.7 };
  c.width = c.clientWidth || 600;
  const x = c.getContext("2d"), risk = state.risk || sliceRisk(), w = c.width / g.slices;
  x.clearRect(0, 0, c.width, c.height);
  for (let z = 0; z < g.slices; z++) {
    x.fillStyle = risk[z] >= st.slice_red ? "#d03b3a" : risk[z] >= st.slice_yellow ? "#e0a800" : "#3aa655";
    x.fillRect(z * w, 0, Math.ceil(w), c.height);
  }
  x.fillStyle = "#111"; x.fillRect(state.z * w + w / 2 - 1, 0, 2, c.height);
  $("#riskLegendTxt").textContent = `(yellow ≥ ${Math.round(st.slice_yellow * 100)}%, red ≥ ${Math.round(st.slice_red * 100)}% nodule probability)`;
}
$("#riskBar").addEventListener("click", (e) => {
  const c = $("#riskBar"), g = state.study.report.geometry;
  setSlice(Math.floor((e.offsetX / c.clientWidth) * g.slices));
});
function camImg(fid, z) {
  const key = `${state.study.id}:${fid}:${z}`;
  if (!state.camCache.has(key)) {
    const im = new Image(); im.onload = () => { if (state.z === z) setSlice(state.z); };
    im.src = `/api/studies/${state.study.id}/cam/${fid}?z=${z}`; state.camCache.set(key, im);
    if (state.camCache.size > 300) state.camCache.delete(state.camCache.keys().next().value);
  }
  return state.camCache.get(key);
}
$("#camToggle").addEventListener("change", () => setSlice(state.z));

function setSlice(z) {
  const g = state.study.report.geometry;
  state.z = Math.max(0, Math.min(g.slices - 1, z));
  $("#zSlider").value = state.z;
  drawRiskBar();
  $("#sliceLabel").textContent = `slice ${state.z + 1} / ${g.slices}`;
  const im = loadImg(state.z);
  if (im.complete && im.naturalWidth) draw(im); else im.onload = () => { if (+$("#zSlider").value === state.z) draw(im); };
  for (const d of [1, 2, 3, -1, -2]) if (state.z + d >= 0 && state.z + d < g.slices) loadImg(state.z + d);
}
function draw(im) {
  const g = state.study.report.geometry, [sx, , sz] = g.spacing_xyz_mm;
  cv.width = g.cols; cv.height = g.rows;
  ctx.drawImage(im, 0, 0);
  if ($("#camToggle").checked) {
    const [sx, sy] = g.spacing_xyz_mm;
    for (const f of state.study.report.findings) {
      if (!f.cam || Math.abs(f.voxel_zyx[0] - state.z) * sz > 16) continue;
      const ci = camImg(f.id, state.z);
      if (ci.complete && ci.naturalWidth) ctx.drawImage(ci, f.voxel_zyx[2] - 16 / sx, f.voxel_zyx[1] - 16 / sy, 32 / sx, 32 / sy);
    }
  }
  ctx.lineWidth = Math.max(1.5, g.cols / 300);
  ctx.font = `${Math.round(g.cols / 34)}px Segoe UI, sans-serif`;
  for (const f of state.study.report.findings) {
    const [fz, fy, fx] = f.voxel_zyx, rMm = Math.max(3, f.diameter_mm / 2) + 3;
    const dzMm = Math.abs(fz - state.z) * sz;
    if (dzMm > rMm) continue;
    const r = Math.sqrt(Math.max(rMm * rMm - dzMm * dzMm, 4)) / sx;
    const rej = state.study.reviews?.[f.id]?.decision === "reject";
    ctx.strokeStyle = rej ? "#8a8a85" : f.id === state.sel ? "#ff9b54" : "#f2c14e";
    ctx.setLineDash(rej ? [6, 5] : []);
    ctx.beginPath(); ctx.arc(fx, fy, r, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = ctx.strokeStyle; ctx.fillText(String(f.id), fx + r + 4, fy - r);
  }
  ctx.setLineDash([]);
}
$("#zSlider").addEventListener("input", (e) => setSlice(+e.target.value));
cv.addEventListener("wheel", (e) => { e.preventDefault(); setSlice(state.z + (e.deltaY > 0 ? 1 : -1)); }, { passive: false });
document.addEventListener("keydown", (e) => {
  if ($("#studyView").hidden || !state.study?.report || ["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  if (e.key === "ArrowUp") { e.preventDefault(); setSlice(state.z - 1); }
  if (e.key === "ArrowDown") { e.preventDefault(); setSlice(state.z + 1); }
});
$("#winSeg").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  state.win = b.dataset.w;
  document.querySelectorAll("#winSeg button").forEach((x) => x.classList.toggle("on", x === b));
  setSlice(state.z);
});

/* ---------------------------------------------------------------- administration */
$("#btnAdmin").onclick = () => openAdmin();
async function openAdmin() {
  show("#adminView");
  const [users, st, logs] = await Promise.all([api("/api/admin/users"), api("/api/settings"), api("/api/admin/audit")]);
  $("#userTable tbody").innerHTML = users.map((u) => `<tr data-uid="${u.id}"><td>${esc(u.username)}</td><td>${esc(u.name)}</td>
    <td><select data-role>${["doctor", "admin"].map((r) => `<option ${u.role === r ? "selected" : ""}>${r}</option>`).join("")}</select></td>
    <td><input type="checkbox" data-active ${u.active ? "checked" : ""}></td><td class="small">${esc((u.last_login || "").replace("T", " "))}</td>
    <td><button class="btn sm" data-pw>Reset password</button></td></tr>`).join("");
  const form = $("#settingsForm");
  for (const [k, v] of Object.entries(st)) if (form.elements[k]) form.elements[k].value = v;
  $("#auditTable tbody").innerHTML = logs.map((l) => `<tr><td>${esc(l.at.replace("T", " "))}</td><td>${esc(l.user)}</td><td>${esc(l.action)}</td><td>${esc(l.detail)}</td></tr>`).join("");
}
$("#userTable").addEventListener("change", async (e) => {
  const tr = e.target.closest("tr[data-uid]"); if (!tr) return;
  const body = e.target.matches("[data-role]") ? { role: e.target.value } : { active: e.target.checked };
  try { await api(`/api/admin/users/${tr.dataset.uid}`, jsonBody("PUT", body)); } catch (err) { alert(err.message); }
  openAdmin();
});
$("#userTable").addEventListener("click", async (e) => {
  if (!e.target.matches("[data-pw]")) return;
  const pw = prompt("New password (at least 8 characters):"); if (!pw) return;
  try { await api(`/api/admin/users/${e.target.closest("tr").dataset.uid}`, jsonBody("PUT", { password: pw })); alert("Password changed."); }
  catch (err) { alert(err.message); }
});
$("#addUserForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  try { await api("/api/admin/users", jsonBody("POST", Object.fromEntries(new FormData(e.target)))); e.target.reset(); openAdmin(); }
  catch (err) { alert(err.message); }
});
$("#settingsForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  try { await api("/api/admin/settings", jsonBody("PUT", Object.fromEntries(new FormData(e.target)))); $("#settingsMsg").textContent = "saved ✓"; openAdmin(); }
  catch (err) { $("#settingsMsg").textContent = err.message; }
});
