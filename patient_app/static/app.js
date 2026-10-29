/* LungCare patient app (plain JS). */
const $ = (s) => document.querySelector(s);
const esc = (x) => String(x ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const md = (t) => esc(t).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/\n/g, "<br>");
const niceDate = (iso) => { try { return new Date(iso.slice(0, 10) + "T00:00").toLocaleDateString(undefined, { day: "numeric", month: "long", year: "numeric" }); } catch (_) { return iso; } };

let CODE = null, HOME = null;
try { CODE = localStorage.getItem("lungcare_code"); } catch (_) {}

const BASE = location.pathname.startsWith("/patient") ? "/patient" : "";   // works locally and online
async function api(path, opts = {}) {
  opts.headers = Object.assign({ "Content-Type": "application/json", "X-Code": CODE || "" }, opts.headers || {});
  const r = await fetch(BASE + path, opts);
  let data = null; try { data = await r.json(); } catch (_) {}
  if (!r.ok) throw new Error((data && data.detail) || r.statusText);
  return data;
}

/* ------------------------------------------------------------- login */
$("#loginForm").onsubmit = async (e) => {
  e.preventDefault();
  $("#loginErr").hidden = true;
  try {
    const r = await api("/api/login", { method: "POST", body: JSON.stringify({ code: $("#code").value.trim() }) });
    CODE = r.code; try { localStorage.setItem("lungcare_code", CODE); } catch (_) {}
    start();
  } catch (err) { $("#loginErr").textContent = err.message === "unknown access code" ? "That code was not found. Please check it with your clinic." : err.message; $("#loginErr").hidden = false; }
};
$("#forgotForm").onsubmit = async (e) => {
  e.preventDefault();
  const m = $("#forgotMsg"); m.hidden = false; m.className = "muted"; m.textContent = "Checking...";
  try {
    const r = await api("/api/recover", { method: "POST", body: JSON.stringify({ phone: $("#fPhone").value, mrn: $("#fMrn").value }) });
    $("#code").value = r.code;
    m.textContent = `Your code is ${r.code} - it has been filled in above.`;
  } catch (err) { m.className = "err"; m.textContent = err.message; }
};
$("#logout").onclick = () => { try { localStorage.removeItem("lungcare_code"); } catch (_) {} CODE = null; location.reload(); };

async function start() {
  try { await loadHome(); } catch (e) { $("#login").hidden = false; $("#app").hidden = true; if (CODE) { $("#loginErr").textContent = e.message; $("#loginErr").hidden = false; } return; }
  $("#login").hidden = true; $("#app").hidden = false;
  loadChat();
}

/* ------------------------------------------------------------- tabs */
document.querySelectorAll(".tabs button").forEach((b) => b.onclick = () => go(b.dataset.go));
function go(tab) {
  document.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("on", x.dataset.go === tab));
  document.querySelectorAll("section.tab").forEach((s) => s.hidden = s.dataset.tab !== tab);
  if (tab === "chat") setTimeout(() => window.scrollTo(0, document.body.scrollHeight), 50);
}

/* ------------------------------------------------------------- home */
function cat(c) { const k = c.split(" ")[0]; return k === "2" || k === "1" ? ["g", "●"] : k === "3" ? ["w", "▲"] : ["b", "◆"]; }

async function loadHome() {
  HOME = await api("/api/home");
  const p = HOME.patient;
  $("#hello").textContent = `Hello, ${p.name.split(" ")[0]}`;
  $("#modeLine").textContent = HOME.assistant === "claude" ? "Assistant: Claude AI" : "Assistant: offline mode";
  $("#summary").innerHTML = md(HOME.summary);

  const plan = HOME.plan;
  if (plan.category_by_size) {
    const has = HOME.reminders.some((r) => r.due_date === plan.suggested_next_scan_date && !r.done);
    $("#nextCard").innerHTML = `<h3>What happens next</h3><div class="muted small">Suggested next scan · based on ${esc(plan.date_source)}</div>
      <div class="date">${niceDate(plan.suggested_next_scan_date)}</div><p>${esc(plan.general_guidance)}</p>
      <p class="muted small">${esc(plan.important)}</p>
      ${has ? `<p><b>✅ Reminder is set.</b></p>` : `<button class="btn primary" id="btnRemind">Remind me</button>`}`;
    const b = $("#btnRemind");
    if (b) b.onclick = async () => { await api("/api/reminders", { method: "POST", body: JSON.stringify({ title: "Follow-up CT scan", due_date: plan.suggested_next_scan_date }) }); loadHome(); };
  } else { $("#nextCard").innerHTML = `<h3>What happens next</h3><p>${esc(plan.plan)}</p>`; }

  const s = HOME.studies[0];
  $("#nodules").innerHTML = s && s.findings.length ? `<div class="card"><h3>Spots your doctor reported</h3>` + s.findings.map((f, i) => {
    const [cl, ic] = cat(f.lung_rads_size_category);
    return `<div class="nod" style="margin:10px 0"><div class="size">${f.diameter_mm.toFixed(1)} mm</div>
      <div><div>Spot ${i + 1} <span class="pill ${cl}">${ic} Category ${esc(f.lung_rads_size_category.split(" ")[0])} (by size)</span></div>
      ${f.doctor_note ? `<div class="muted small">Doctor: ${esc(f.doctor_note)}</div>` : ""}</div></div>`;
  }).join("") + `<p class="muted small">10 mm = 1 cm. Categories are estimated from size only; your doctor's plan always comes first.</p></div>` : "";
  renderReminders(); renderSymptoms(); renderRequests();
}

/* ------------------------------------------------------------- chat */
function bubble(role, text, actions = []) {
  const d = document.createElement("div");
  d.className = "bub " + role;
  d.innerHTML = md(text) + (actions.length ? `<div class="acts">${actions.map((a) => `<span class="act">✓ ${esc(a)}</span>`).join("")}</div>` : "");
  $("#chatLog").appendChild(d);
  window.scrollTo(0, document.body.scrollHeight);
  return d;
}
async function loadChat() {
  const h = await api("/api/chat");
  $("#chatLog").innerHTML = "";
  if (!h.length) bubble("assistant", "Hi! I'm your LungCare assistant. I can explain your results, set reminders, keep a symptom diary and send messages to your doctor. What would you like to do?");
  h.forEach((m) => bubble(m.role, m.content, m.actions));
}
async function send(text) {
  if (!text.trim()) return;
  bubble("user", text);
  const t = bubble("assistant", "…thinking"); t.classList.add("typing");
  try {
    const r = await api("/api/chat", { method: "POST", body: JSON.stringify({ message: text }) });
    t.remove(); bubble("assistant", r.reply, r.actions);
    if (r.actions.length) loadHome();
  } catch (e) { t.textContent = "Sorry - " + e.message; }
}
$("#chatForm").onsubmit = (e) => { e.preventDefault(); const v = $("#chatInput").value; $("#chatInput").value = ""; send(v); };
$("#chips").onclick = (e) => { if (e.target.tagName === "BUTTON") send(e.target.textContent); };

/* ------------------------------------------------------------- reminders */
function renderReminders() {
  const today = HOME.today;
  $("#remList").innerHTML = HOME.reminders.length ? HOME.reminders.map((r) => {
    const days = Math.round((new Date(r.due_date) - new Date(today)) / 864e5);
    const cls = r.done ? "" : days < 0 ? "late" : days <= 14 ? "soon" : "";
    const when = r.done ? "done" : days < 0 ? `${-days} days overdue` : days === 0 ? "today" : `in ${days} days`;
    return `<div class="card item ${r.done ? "done" : ""}"><div><div class="t"><b>${esc(r.title)}</b></div>
      <div class="due ${cls}">${niceDate(r.due_date)} · ${when}</div></div>
      <div><button class="btn sm" data-done="${r.id}" data-v="${r.done ? 0 : 1}">${r.done ? "Undo" : "Done ✓"}</button>
      <button class="btn sm" data-del="${r.id}">✕</button></div></div>`;
  }).join("") : `<p class="muted">No reminders yet. The assistant can set one for your next scan.</p>`;
}
$("#remList").onclick = async (e) => {
  const d = e.target.closest("[data-done]"), x = e.target.closest("[data-del]");
  if (d) await api(`/api/reminders/${d.dataset.done}/done?done=${d.dataset.v === "1"}`, { method: "PUT" });
  if (x) await api(`/api/reminders/${x.dataset.del}`, { method: "DELETE" });
  if (d || x) loadHome();
};
$("#remForm").onsubmit = async (e) => {
  e.preventDefault();
  const f = Object.fromEntries(new FormData(e.target));
  try { await api("/api/reminders", { method: "POST", body: JSON.stringify(f) }); e.target.reset(); loadHome(); } catch (err) { alert(err.message); }
};

/* ------------------------------------------------------------- diary */
$("#sev").oninput = (e) => $("#sevVal").textContent = e.target.value;
$("#symForm").onsubmit = async (e) => {
  e.preventDefault();
  const f = Object.fromEntries(new FormData(e.target)); f.severity = +f.severity;
  const r = await api("/api/symptoms", { method: "POST", body: JSON.stringify(f) });
  const m = $("#symMsg"); m.hidden = false;
  m.className = "msg" + (r.urgent ? " bad" : "");
  m.textContent = r.urgent ? "⚠️ " + r.advice : "Saved. If it gets worse, tell your doctor.";
  e.target.reset(); $("#sevVal").textContent = 3; loadHome();
};
function renderSymptoms() {
  $("#symList").innerHTML = HOME.symptoms.map((s) => `<div class="card item"><div><b>${esc(s.symptom)}</b>${s.urgent ? ` <span class="pill b">urgent - doctor alerted</span>` : ""}
    <div class="due">${esc(s.at.replace("T", " ").slice(0, 16))}${s.severity ? ` · ${s.severity}/10` : ""}${s.notes ? " · " + esc(s.notes) : ""}</div></div></div>`).join("");
}

/* ------------------------------------------------------------- messages */
function renderRequests() {
  $("#reqList").innerHTML = HOME.requests.length ? HOME.requests.map((r) => `<div class="card req ${r.urgency}">
    <div class="muted small">${r.kind === "appointment" ? "Appointment request" : r.kind === "symptom" ? "Symptom report" : "Question"} · ${esc(r.created.replace("T", " ").slice(0, 16))}</div>
    <div>${esc(r.text)}</div>
    ${r.reply ? `<div class="reply"><b>Doctor:</b> ${esc(r.reply)}</div>` : `<div class="muted small" style="margin-top:6px">Waiting for reply…</div>`}</div>`).join("")
    : `<p class="muted">No messages yet.</p>`;
}
$("#reqForm").onsubmit = async (e) => {
  e.preventDefault();
  const f = Object.fromEntries(new FormData(e.target));
  try { await api("/api/requests", { method: "POST", body: JSON.stringify(f) }); e.target.reset(); loadHome(); } catch (err) { alert(err.message); }
};

if (CODE) start();
