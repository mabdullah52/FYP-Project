"""Agentic AI assistant for the patient app.

The assistant does not just chat: it plans and calls TOOLS that act on the patient's
data (read doctor-released results, build a follow-up plan, set reminders, keep a
symptom diary with red-flag triage, send questions / appointment requests to the doctor).

Two interchangeable "brains" drive the same tools:
  * ClaudeBrain  - Claude via the Anthropic API with tool use (set ANTHROPIC_API_KEY)
  * OfflineBrain - built-in rule-based planner (no internet needed; always available)
Safety rules (both): never diagnose, never estimate cancer risk, results are only what
the doctor released, urgent symptoms -> emergency advice + urgent message to doctor.
"""
import json
import os
import re
from datetime import date, datetime, timedelta

# --------------------------------------------------------------------------- knowledge
GLOSSARY = {
    "nodule": "A small, round spot in the lung (usually smaller than 3 cm). Most lung nodules are NOT cancer - "
              "they are often old scars or healed infections. Doctors watch them over time to make sure they do not grow.",
    "ct": "A CT (computed tomography) scan is a detailed X-ray that takes many thin pictures of the chest. "
          "A low-dose CT uses much less radiation and is used for lung screening.",
    "lung-rads": "Lung-RADS is the scoring system doctors use for lung screening CTs. It goes from 1 (no nodules) to 4 "
                 "(needs closer checking). In this app the category is estimated from nodule SIZE only - your doctor "
                 "makes the final decision.",
    "mm": "mm means millimetres. 10 mm = 1 cm. A 6 mm nodule is about the size of a pencil eraser tip.",
    "false positive": "A false positive is a spot the computer marked but the doctor checked and decided is NOT a nodule "
                      "(for example a blood vessel). Those are removed and you will not see them here.",
    "benign": "Benign means not cancer.",
    "malignant": "Malignant means cancer. This app and its AI do not decide whether anything is cancer - only your doctor can.",
    "follow-up": "A follow-up scan is a repeat CT after some months to check whether a nodule has changed. "
                 "A nodule that stays the same size over time is very likely harmless.",
    "pet": "A PET-CT scan shows how active cells are. Doctors sometimes use it to look more closely at a larger nodule.",
    "biopsy": "A biopsy takes a tiny sample of tissue to look at under a microscope. It is only done when doctors need more certainty.",
    "ai": "The AI in this system helped your doctor find small spots in your CT. Your doctor reviewed every finding "
          "before releasing results to you.",
}
SYNONYMS = {"nodules": "nodule", "lump": "nodule", "spot": "nodule", "ct scan": "ct", "cat scan": "ct",
            "lungrads": "lung-rads", "lung rads": "lung-rads", "millimeter": "mm", "millimetre": "mm",
            "followup": "follow-up", "follow up": "follow-up", "pet scan": "pet", "pet-ct": "pet",
            "artificial intelligence": "ai", "cancerous": "malignant", "not cancer": "benign"}

# Lung-RADS v2022 baseline management (by category) - general guidance only
PLAN = {
    "1": ("Continue yearly screening (low-dose CT in 12 months).", 12),
    "2": ("Continue yearly screening (low-dose CT in 12 months).", 12),
    "3": ("A repeat low-dose CT in about 6 months is usually advised.", 6),
    "4A": ("A repeat low-dose CT in about 3 months is usually advised; your doctor may also consider a PET-CT.", 3),
    "4B": ("Further tests soon are usually advised (for example a diagnostic CT, PET-CT or other evaluation). "
           "Please follow your doctor's instructions promptly.", 1),
}
RED_FLAGS = {
    "coughing up blood": ["coughing blood", "cough blood", "coughed up blood", "coughing up blood", "spitting blood",
                          "blood in my cough", "blood in sputum", "haemoptysis", "hemoptysis", "khoon"],
    "chest pain": ["chest pain", "pain in chest", "chest hurts", "seenay mein dard"],
    "severe breathlessness": ["can't breathe", "cannot breathe", "cant breathe", "struggling to breathe",
                              "severe shortness of breath", "short of breath at rest", "gasping", "saans nahi"],
    "fainting": ["faint", "passed out", "unconscious"],
}

SYSTEM_PROMPT = """You are "LungCare Assistant", a friendly helper inside a patient app for people who had a chest CT.
Speak simply (reading age ~12), warmly and briefly. If the patient writes in Urdu or Roman Urdu, answer in the same language.
Rules you must always follow:
- Only use results from the get_my_results tool. These are results the doctor has REVIEWED and RELEASED. Never invent results.
- Never diagnose, never say something is or is not cancer, never give a cancer probability. The AI only finds nodules (spots);
  most nodules are benign. The doctor decides. Encourage questions to the doctor.
- Lung-RADS categories here are estimated from size only; say the doctor's plan always wins.
- If the patient mentions coughing up blood, chest pain, severe breathlessness or fainting: tell them to seek emergency
  care now, and use log_symptom so the doctor is alerted urgently.
- Use tools to ACT: set reminders, log symptoms, send questions or appointment requests to the doctor. Confirm what you did.
- Before setting a reminder or sending something to the doctor on the patient's behalf, make sure they asked for it
  (or clearly agreed).
- For general health questions (tests such as PET-CT or biopsy, breathing exercises, staying active, quitting smoking,
  air pollution, screening, what to ask the doctor, when to get urgent help) call search_guidelines and answer ONLY from
  the passages it returns, in simple words, and name the source (e.g. "Source: WHO"). If nothing relevant is found,
  say you don't know and suggest asking the doctor.
Today's date is {today}."""


# --------------------------------------------------------------------------- tools
class Tools:
    """All actions the assistant can take. `ctx` gives access to the patient's data."""

    SPEC = [
        {"name": "get_my_results", "description": "Get the patient's CT results that the doctor has reviewed and released, "
         "with nodule sizes, size-based Lung-RADS categories, the doctor's decisions and the doctor's impression.",
         "input_schema": {"type": "object", "properties": {}}},
        {"name": "get_followup_plan", "description": "General follow-up guidance for the patient's most significant released "
         "finding (based on Lung-RADS v2022) plus a suggested date for the next scan.",
         "input_schema": {"type": "object", "properties": {}}},
        {"name": "explain_term", "description": "Explain a medical term in simple words (nodule, CT, Lung-RADS, mm, "
         "false positive, benign, malignant, follow-up, PET, biopsy, AI).",
         "input_schema": {"type": "object", "properties": {"term": {"type": "string"}}, "required": ["term"]}},
        {"name": "set_reminder", "description": "Create a reminder for the patient (e.g. next CT scan, appointment).",
         "input_schema": {"type": "object", "properties": {
             "title": {"type": "string"}, "due_date": {"type": "string", "description": "YYYY-MM-DD"},
             "note": {"type": "string"}}, "required": ["title", "due_date"]}},
        {"name": "list_reminders", "description": "List the patient's reminders.",
         "input_schema": {"type": "object", "properties": {}}},
        {"name": "log_symptom", "description": "Record a symptom in the patient's diary. Urgent symptoms are automatically "
         "sent to the doctor as URGENT.", "input_schema": {"type": "object", "properties": {
             "symptom": {"type": "string"}, "severity": {"type": "integer", "description": "1 (mild) to 10 (worst)"},
             "notes": {"type": "string"}}, "required": ["symptom"]}},
        {"name": "request_appointment", "description": "Send an appointment request to the patient's doctor.",
         "input_schema": {"type": "object", "properties": {
             "reason": {"type": "string"}, "preferred_date": {"type": "string"},
             "urgency": {"type": "string", "enum": ["routine", "soon", "urgent"]}}, "required": ["reason"]}},
        {"name": "send_question_to_doctor", "description": "Send a written question to the doctor.",
         "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}},
        {"name": "get_doctor_replies", "description": "See the patient's earlier messages to the doctor and any replies.",
         "input_schema": {"type": "object", "properties": {}}},
        {"name": "search_guidelines", "description": "Search the hospital's approved patient-guideline library (summaries of "
         "WHO, ACR Lung-RADS, USPSTF and lung-health guidance, plus official PDFs added by the hospital). Returns the most "
         "relevant passages with their source. Use for general health questions; answer only from these passages.",
         "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    ]

    def __init__(self, ctx):
        self.ctx = ctx          # PatientContext from server.py
        self.actions = []       # human-readable log of what was done (shown as chips in the UI)

    def call(self, name, args):
        fn = getattr(self, name, None)
        if fn is None or name.startswith("_"):
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**(args or {}))
        except TypeError as e:
            return {"error": f"bad arguments: {e}"}

    # ---- guideline library (RAG)
    def search_guidelines(self, query, k=3):
        from guidelines import index
        hits = index().search(query, k)
        return {"passages": [{"document": h["doc"], "section": h["section"], "text": h["text"], "source": h["source"]}
                             for h in hits]} if hits else {"passages": [], "note": "nothing relevant in the library"}

    # ---- read
    def get_my_results(self):
        portal = self.ctx.portal()
        studies = []
        for s in portal["studies"]:
            studies.append({
                "date": s["date"][:10], "released": (s.get("released_at") or "")[:10], "doctor": s.get("doctor"),
                "doctor_impression": s.get("impression"),
                "nodules": [{"size_mm": f["diameter_mm"], "lung_rads_by_size": f["lung_rads_size_category"],
                             "doctor_checked": f.get("doctor_decision") or "not individually marked",
                             "doctor_note": f.get("doctor_note")} for f in s["findings"]]})
        return {"patient_first_name": portal["patient"]["name"].split()[0], "studies": studies,
                "note": "Only doctor-reviewed and released results are shown." if studies else
                        "Your doctor has not released any results yet."}

    def get_followup_plan(self):
        portal = self.ctx.portal()
        if not portal["studies"]:
            return {"plan": "No released results yet - your doctor will contact you.", "category": None}
        s = portal["studies"][0]
        cats = [f["lung_rads_size_category"].split()[0] for f in s["findings"] if f.get("doctor_decision") != "reject"]
        order = ["1", "2", "3", "4A", "4B"]
        cat = max(cats, key=order.index) if cats else "1"
        text, months = PLAN[cat]
        base = datetime.fromisoformat((s.get("released_at") or s["date"])[:19]).date()
        due, source = _add_months(base, months), "general guideline (size-based Lung-RADS)"
        # the DOCTOR's written interval always overrides the size-based guideline
        doc = _doctor_interval(s.get("impression") or "")
        if doc:
            n, unit = doc
            due = (base + timedelta(weeks=n)) if unit == "week" else _add_months(base, n * (12 if unit == "year" else 1))
            text = f"Your doctor advised a repeat scan in {n} {unit}{'s' if n > 1 else ''}."
            source = "your doctor's instructions"
        return {"category_by_size": cat, "general_guidance": text, "suggested_next_scan_date": due.isoformat(),
                "date_source": source, "doctor_impression": s.get("impression"),
                "important": "Your doctor's impression and instructions always come first." if doc else
                             "This is general guidance. Your doctor's impression and instructions always come first."}

    def explain_term(self, term):
        t = term.lower().strip()
        t = SYNONYMS.get(t, t)
        for k, v in GLOSSARY.items():
            if k == t or k in t:
                return {"term": k, "explanation": v}
        return {"term": term, "explanation": None, "hint": "Not in the glossary - explain simply or suggest asking the doctor."}

    def list_reminders(self):
        return {"reminders": self.ctx.reminders()}

    def get_doctor_replies(self):
        return {"messages": self.ctx.portal().get("requests", [])}

    # ---- act
    def set_reminder(self, title, due_date, note=None):
        d = _parse_date(due_date)
        if d is None:
            return {"error": "due_date must be a date like 2027-03-01"}
        rid = self.ctx.add_reminder(title, d.isoformat(), note)
        self.actions.append(f"Reminder set: {title} on {d.strftime('%d %b %Y')}")
        return {"ok": True, "id": rid, "title": title, "due_date": d.isoformat()}

    def log_symptom(self, symptom, severity=None, notes=None):
        text = f"{symptom} {notes or ''}".lower()
        flags = [name for name, words in RED_FLAGS.items() if any(w in text for w in words)]
        urgent = bool(flags) or (severity is not None and int(severity) >= 8)
        self.ctx.add_symptom(symptom, severity, notes, urgent)
        self.actions.append(f"Symptom logged: {symptom}" + (" (URGENT - doctor alerted)" if urgent else ""))
        out = {"ok": True, "urgent": urgent, "red_flags": flags}
        if urgent:
            self.ctx.send_request("symptom", f"URGENT symptom reported via app: {symptom}"
                                  + (f" (severity {severity}/10)" if severity else "") + (f". {notes}" if notes else ""),
                                  "urgent")
            out["advice"] = ("These symptoms can be serious. Please go to the nearest emergency department or call "
                             "your local emergency number now. Your doctor has also been alerted.")
        return out

    def request_appointment(self, reason, preferred_date=None, urgency="routine"):
        txt = reason + (f" (preferred date: {preferred_date})" if preferred_date else "")
        r = self.ctx.send_request("appointment", txt, urgency or "routine")
        self.actions.append("Appointment request sent to your doctor")
        return {"ok": True, "request_id": r.get("id"), "status": "sent - the clinic will reply in the app"}

    def send_question_to_doctor(self, question):
        r = self.ctx.send_request("question", question, "routine")
        self.actions.append("Question sent to your doctor")
        return {"ok": True, "request_id": r.get("id")}


def _doctor_interval(text):
    """Find 'in 6 months', '3-month', 'after 2 weeks', '1 year' in the doctor's impression."""
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "six": 6, "twelve": 12, "a": 1}
    m = re.search(r"(\d+|one|two|three|four|six|twelve|a)[\s-]*(month|week|year)s?", text.lower())
    if not m:
        return None
    n = int(m.group(1)) if m.group(1).isdigit() else words[m.group(1)]
    return n, m.group(2)


def _add_months(d, m):
    y, mo = divmod(d.month - 1 + m, 12)
    import calendar
    day = min(d.day, calendar.monthrange(d.year + y, mo + 1)[1])
    return date(d.year + y, mo + 1, day)


def _parse_date(s):
    if not s:
        return None
    s = s.strip().lower()
    today = date.today()
    if s in ("today",):
        return today
    if s in ("tomorrow", "kal"):
        return today + timedelta(days=1)
    m = re.search(r"in (\d+) (day|week|month|year)s?", s)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        return (today + timedelta(days=n) if unit == "day" else today + timedelta(weeks=n) if unit == "week"
                else _add_months(today, n) if unit == "month" else _add_months(today, 12 * n))
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


# --------------------------------------------------------------------------- brains
class ClaudeBrain:
    name = "claude"

    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = os.environ.get("LUNGCARE_MODEL", "claude-sonnet-5")

    def run(self, tools, history, message):
        msgs = [{"role": h["role"], "content": h["content"]} for h in history[-12:]]
        msgs.append({"role": "user", "content": message})
        system = SYSTEM_PROMPT.format(today=date.today().isoformat())
        for _ in range(6):                       # agent loop: think -> call tools -> observe
            resp = self.client.messages.create(model=self.model, max_tokens=800, system=system,
                                               tools=Tools.SPEC, messages=msgs)
            msgs.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                return "".join(b.text for b in resp.content if b.type == "text").strip()
            results = []
            for b in resp.content:
                if b.type == "tool_use":
                    out = tools.call(b.name, b.input)
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(out, default=str)})
            msgs.append({"role": "user", "content": results})
        return "Sorry, I could not finish that. Please try again or ask your doctor."


class OfflineBrain:
    """Rule-based planner: detects one or more intents, calls the matching tools, writes the reply."""
    name = "offline"

    def run(self, tools, history, message):
        m = message.lower().strip()
        parts = []

        # 1) safety first: red-flag symptoms
        if any(w in m for words in RED_FLAGS.values() for w in words):
            sev = _first_int(m)
            r = tools.log_symptom(message[:200], sev if sev and sev <= 10 else None)
            return ("⚠️ " + r["advice"] + "\n\nI have recorded this in your symptom diary and sent an URGENT message "
                    "to your doctor.")

        intents = []
        if re.search(r"\b(hi|hello|salam|assalam|hey)\b", m) or m in ("start",):
            intents.append("summary")
        if re.search(r"result|report|mean|scan show|what did|found|explain my", m):
            intents.append("summary")
        if re.search(r"cancer|malignan|serious|dangerous|die|tumou?r", m):
            intents.append("cancer")
        if re.search(r"what happens next|what('s| is) next|next step|next scan|follow.?up|when should|what now|"
                     r"what should i do|my plan|treatment plan", m):
            intents.append("plan")
        if re.search(r"remind|reminder|alarm|don't let me forget", m):
            intents.append("remind")
        if re.search(r"appointment|book|see (the )?doctor|visit|meet", m):
            intents.append("appointment")
        if re.search(r"ask (the |my )?doctor|question for|tell (the |my )?doctor|message (the |my )?doctor", m):
            intents.append("question")
        if re.search(r"repl(y|ied|ies)|answer(ed)?|did .*doctor", m):
            intents.append("replies")
        if re.search(r"cough|pain|tired|fever|breath|weight loss|symptom|feel(ing)? (bad|unwell|sick)", m):
            intents.append("symptom")
        if re.search(r"smok|cigarette|quit|vape|tobacco|shisha|hookah", m):
            intents.append("smoking")
        if re.search(r"exercis|breathing (technique|exercise)|stay active|workout|physical activity|yoga|walk(ing)? more", m):
            intents = [i for i in intents if i != "symptom"] + ["exercises"]
        if re.search(r"what (should|can|could) i ask|questions? (to|i could|i should) ask|what to ask", m):
            intents = [i for i in intents if i != "question"] + ["ask_list"]
        if re.search(r"\bpet\b|biopsy|bronchoscop|diagnostic ct|contrast|pollution|smog|screening|vaccin|healthy lungs|diet", m) \
                and not intents:
            intents.append("library")
        if re.search(r"my reminders|list reminders|show reminders", m):
            intents = [i for i in intents if i != "remind"] + ["list_reminders"]
        terms = [k for k in GLOSSARY if re.search(r"\b" + re.escape(k) + r"\b", m)]
        terms += [v for k, v in SYNONYMS.items() if k in m and v not in terms]
        if terms and re.search(r"what is|what's|what are|meaning|mean|explain|define", m):
            intents.append("terms")

        if "remind" in intents and "plan" in intents:
            intents.remove("plan")
        seen = []
        for it in intents:
            if it in seen:
                continue
            seen.append(it)
            if it == "summary":
                parts.append(_summary_text(tools.get_my_results()))
            elif it == "cancer":
                res = tools.get_my_results()
                if res["studies"] and _needs_tests(res["studies"][0]):
                    parts.append("I can't tell you whether something is cancer - only your doctor can. Your scan shows a "
                                 "spot your doctor will want to **check further soon**, so the most important thing is to "
                                 "follow their plan. Would you like me to send your worry as a question to your doctor?")
                else:
                    parts.append("I can't tell you whether something is cancer - only your doctor can. What I can say: "
                                 "the AI only finds small spots (nodules), and **most lung nodules are not cancer**. "
                                 "Your doctor's note and follow-up plan are the best guide. Would you like me to send your "
                                 "worry as a question to your doctor?")
            elif it == "plan":
                p = tools.get_followup_plan()
                if p.get("category_by_size"):
                    parts.append(f"**Next step:** {p['general_guidance']} A suggested date for your next scan is "
                                 f"**{_nice(p['suggested_next_scan_date'])}**. {p['important']}"
                                 + (f"\n\nYour doctor wrote: “{p['doctor_impression']}”" if p.get("doctor_impression") else "")
                                 + "\n\nSay **“remind me”** and I will set a reminder for it.")
                else:
                    parts.append(p["plan"])
            elif it == "remind":
                d = _parse_date(_extract_date_phrase(m) or "")
                title = "Follow-up CT scan"
                if d is None:
                    p = tools.get_followup_plan()
                    d = _parse_date(p.get("suggested_next_scan_date") or "")
                if "appointment" in m:
                    title = "Doctor appointment"
                if d is None:
                    parts.append("For which date should I set the reminder? (e.g. “remind me on 2027-01-15”)")
                else:
                    tools.set_reminder(title, d.isoformat())
                    parts.append(f"✅ Done - I set a reminder **“{title}”** for **{d.strftime('%d %B %Y')}**.")
            elif it == "list_reminders":
                rs = tools.list_reminders()["reminders"]
                parts.append("Your reminders:\n" + "\n".join(f"• {r['title']} - {_nice(r['due_date'])}" for r in rs)
                             if rs else "You have no reminders yet.")
            elif it == "appointment":
                tools.request_appointment(message[:300])
                parts.append("✅ I sent an **appointment request** to your doctor. You will see the reply in the "
                             "Messages section.")
            elif it == "question":
                q = message.split(":", 1)[1].strip() if ":" in message else message
                tools.send_question_to_doctor(q[:1000])
                parts.append("✅ Your question was sent to your doctor. I'll show the answer here when it arrives.")
            elif it == "replies":
                msgs = tools.get_doctor_replies()["messages"]
                ans = [r for r in msgs if r.get("reply")]
                parts.append("\n".join(f"• You: “{r['text'][:80]}” → Doctor: “{r['reply']}”" for r in ans[:5])
                             if ans else "No replies from your doctor yet.")
            elif it == "symptom":
                sev = _first_int(m)
                r = tools.log_symptom(message[:200], sev if sev and sev <= 10 else None)
                parts.append("📝 I recorded this in your symptom diary." + (
                    " " + r["advice"] if r.get("urgent") else
                    " If it gets worse, or you cough up blood, have chest pain or struggle to breathe, seek urgent care. "
                    "Would you like me to tell your doctor or ask for an appointment?"))
            elif it == "smoking":
                parts.append(_from_library(tools, "how to quit smoking benefits", 2)
                             + "\n\nShall I send a message to your doctor asking about help to quit?")
            elif it == "exercises":
                parts.append(_from_library(tools, "breathing exercises pursed lip belly breathing staying active walking", 3))
            elif it == "ask_list":
                parts.append(_from_library(tools, "questions to ask your doctor result next steps", 3)
                             + "\n\nWrite **“ask my doctor: …”** and I will send your question.")
            elif it == "library":
                parts.append(_from_library(tools, message, 2))
            elif it == "terms":
                for t in dict.fromkeys(terms):
                    e = tools.explain_term(t)
                    if e.get("explanation"):
                        parts.append(f"**{e['term'].capitalize()}:** {e['explanation']}")

        if not parts:
            hits = tools.search_guidelines(message, 2)["passages"]
            from guidelines import index
            top = index().search(message, 1)
            if top and top[0]["score"] >= 3.0:
                parts.append(_format_passages(hits))
        if not parts:
            parts.append("I can help you with:\n• **Explain my results**\n• **What happens next?**\n"
                         "• **Remind me** about my next scan\n• Log a **symptom** (e.g. “I have a cough, 4/10”)\n"
                         "• **Ask my doctor:** your question\n• **Book an appointment**\n"
                         "• What is a **nodule** / **Lung-RADS**?\n• **Breathing exercises**, **quitting smoking**, "
                         "what is a **PET scan**, **questions to ask** my doctor")
        return "\n\n".join(parts)


def _format_passages(passages):
    out = []
    for p in passages:
        out.append(f"**{p['section']}** - {p['text']}")
    if passages:
        srcs = list(dict.fromkeys(p["source"] for p in passages))
        out.append("_Source: " + "; ".join(srcs) + "._ Your doctor's advice always comes first.")
    return "\n\n".join(out)


def _from_library(tools, query, k):
    hits = tools.search_guidelines(query, k)["passages"]
    return _format_passages(hits) if hits else "I could not find that in the guideline library - please ask your doctor."


def _needs_tests(study):
    """Lung-RADS 4A/4B by size (>= 8 mm solid) means further tests are usually advised."""
    return any(str(f.get("lung_rads_by_size", "")).startswith("4") for f in study.get("nodules", []))


def _summary_text(r):
    if not r["studies"]:
        return f"Hello {r['patient_first_name']} 👋 Your doctor has not released any CT results yet. I'll show them here as soon as they do."
    s = r["studies"][0]
    n = len(s["nodules"])
    lines = [f"Hello {r['patient_first_name']} 👋 Here is your CT from **{_nice(s['date'])}**, reviewed by "
             f"{s['doctor'] or 'your doctor'}."]
    if n == 0:
        lines.append("No lung nodules (spots) were reported. 🎉")
    else:
        susp = _needs_tests(s)
        word = "spot" if susp else "small spot"
        lines.append(f"Your doctor reported **{n} {word}{'s' if n > 1 else ''} (nodule{'s' if n > 1 else ''})**:")
        for i, f in enumerate(s["nodules"], 1):
            lines.append(f"• Spot {i}: about **{f['size_mm']:.1f} mm** (category {f['lung_rads_by_size'].split(' ')[0]} by size)")
        if susp:
            lines.append("At least one spot is large enough that your doctor will want **further tests soon**. "
                         "This is not a diagnosis - it means checking it properly. Please follow your doctor's plan.")
        else:
            lines.append("Most lung nodules are **not cancer**. Their size helps your doctor decide when to check again.")
    if s.get("doctor_impression"):
        lines.append(f"**Your doctor's note:** “{s['doctor_impression']}”")
    lines.append("Ask me “what happens next?” or “what is a nodule?” anytime.")
    return "\n".join(lines)


def _nice(iso):
    try:
        return datetime.fromisoformat(iso[:10]).strftime("%d %B %Y")
    except Exception:
        return iso


def _first_int(s):
    m = re.search(r"\b(\d{1,2})\s*(/\s*10)?\b", s)
    return int(m.group(1)) if m else None


def _extract_date_phrase(m):
    for pat in (r"\d{4}-\d{2}-\d{2}", r"\d{1,2}[/-]\d{1,2}[/-]\d{4}", r"in \d+ (?:day|week|month|year)s?", r"tomorrow",
                r"\d{1,2} (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{4}"):
        x = re.search(pat, m)
        if x:
            return x.group(0)
    return None


def make_brain():
    """Claude if an API key is configured and the SDK is installed, else the offline planner.
    The key can be an environment variable or pasted into patient_app/claude_api_key.txt."""
    key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_api_key.txt")
    if not os.environ.get("ANTHROPIC_API_KEY") and os.path.exists(key_file):
        key = open(key_file).read().strip()
        if key and not key.startswith("#"):
            os.environ["ANTHROPIC_API_KEY"] = key
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return ClaudeBrain()
        except Exception as e:   # SDK missing etc. -> still usable offline
            print("Claude unavailable, using offline assistant:", e)
    return OfflineBrain()
