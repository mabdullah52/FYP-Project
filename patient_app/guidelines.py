"""Retrieval-augmented guideline navigator (RAG) for the patient assistant.

Knowledge base = every .md / .txt / .pdf file in patient_app/knowledge/ (the built-in summaries of WHO,
ACR Lung-RADS, USPSTF ... guidance, plus any official PDFs the hospital adds there).
Documents are split into passages (one per heading / paragraph block) and indexed with BM25; the
assistant answers ONLY from the retrieved passages and always names the source document.
No internet and no extra packages are needed (PDF reading uses pypdf if it is installed).
"""
import glob
import math
import os
import re
from collections import Counter

KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge")
STOP = set("a an and are as at be by can do does for from has have how i if in is it its me my of on or our should "
           "so that the their them then there these they this to was what when where which who why will with you your "
           "about after before also am any been but could did get got just like may more most much need no not now "
           "one only other out over some than too very would".split())


def _stem(w):
    for suf in ("ing", "ers", "er", "es", "s", "ed"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            w = w[: -len(suf)]
            break
    if len(w) > 4 and w[-1] == w[-2]:
        w = w[:-1]                      # quitting -> quitt -> quit
    return w


def _tok(text):
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 1]


def _read(path):
    if path.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            print("pypdf not installed - skipping", os.path.basename(path), "(pip install pypdf)")
            return "", {}
        text = "\n\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
        return text, {"title": os.path.splitext(os.path.basename(path))[0].replace("_", " "),
                      "source": os.path.basename(path)}
    text = open(path, encoding="utf-8", errors="replace").read()
    meta = {}
    for key in ("title", "source"):
        m = re.search(rf"^{key}:\s*(.+)$", text, re.M)
        if m:
            meta[key] = m.group(1).strip()
            text = text.replace(m.group(0), "")
    meta.setdefault("title", os.path.splitext(os.path.basename(path))[0])
    meta.setdefault("source", os.path.basename(path))
    return text, meta


def _passages(text, meta, max_words=160):
    out, heading = [], meta["title"]
    blocks = re.split(r"\n\s*\n", text)
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        if b.startswith("#"):
            first, _, rest = b.partition("\n")
            heading = first.lstrip("#").strip()
            b = rest.strip()
            if not b:
                continue
        words = b.split()
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words])
            out.append({"doc": meta["title"], "section": heading, "source": meta["source"], "text": chunk})
    return out


class GuidelineIndex:
    def __init__(self, kb_dir=KB_DIR):
        self.passages = []
        for p in sorted(glob.glob(os.path.join(kb_dir, "*"))):
            if p.lower().endswith((".md", ".txt", ".pdf")):
                text, meta = _read(p)
                self.passages += _passages(text, meta)
        # title and section heading count twice: they say what the passage is about
        self.docs = [_tok((p["doc"] + " " + p["section"] + " ") * 2 + p["text"]) for p in self.passages]
        self.avgdl = sum(map(len, self.docs)) / max(1, len(self.docs))
        df = Counter(w for d in self.docs for w in set(d))
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def search(self, query, k=3, k1=1.5, b=0.75):
        q = _tok(query)
        scores = []
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            s = 0.0
            for w in q:
                if w in tf:
                    s += self.idf[w] * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / self.avgdl))
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [dict(self.passages[i], score=round(s, 2)) for s, i in scores[:k]]


_INDEX = None


def index():
    global _INDEX
    if _INDEX is None:
        _INDEX = GuidelineIndex()
    return _INDEX


if __name__ == "__main__":
    ix = index()
    print(len(ix.passages), "passages")
    for q in ("breathing exercises", "what is a PET scan", "how to quit smoking", "category 4A next step", "air pollution"):
        r = ix.search(q, 1)
        print(q, "->", r[0]["doc"], "/", r[0]["section"] if r else None)
