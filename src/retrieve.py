from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

KB_DIRS = [Path("kb"), Path("data/processed")]
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

# Globale (enkle) caches i prosessen
_VEC: Optional[TfidfVectorizer] = None
_MTX = None  # scipy sparse
_META: List[Dict] = []  # én entry per rad i _MTX


# ---------- Utils ----------

def _read_text_file(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def _strip_markdown_noise(txt: str) -> str:
    # Fjern kodeblokker / HTML-kommentarer / overflødig whitespace
    txt = re.sub(r"```.*?```", " ", txt, flags=re.S)
    txt = re.sub(r"<!--.*?-->", " ", txt, flags=re.S)
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()

def _title_from_markdown(txt: str, fallback: str) -> str:
    m = re.search(r"^\s*#\s+(.+)$", txt, flags=re.M)
    if m:
        return m.group(1).strip()
    # Alternativ: første ikke-tomme linje
    for line in txt.splitlines():
        s = line.strip()
        if s:
            return s[:120]
    return fallback

def _infer_doc_type(name: str, text: str) -> str:
    low = (name + " " + text[:400]).lower()
    if any(w in low for w in ["vilkår", "terms", "betingelser", "angrerett", "personvern", "gdpr", "privacy"]):
        return "regel"
    if any(w in low for w in ["pris", "timepris", "avgift", "kontingent", "kostnad", "rabatt"]):
        return "pris"
    if any(w in low for w in ["booking", "banebooking", "reserver", "matchi", "baneregler"]):
        return "booking"
    if any(w in low for w in ["håndbok"]):
        return "håndbok"
    return "annet"

def _chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + size, n)
        chunk = text[i:j]
        chunks.append(chunk)
        if j == n:
            break
        i = max(j - overlap, 0)
    return chunks

def _iter_kb_files() -> Iterable[Path]:
    seen = set()
    for d in KB_DIRS:
        if not d.exists():
            continue
        # markdown
        for p in d.rglob("*.md"):
            if p.is_file():
                seen.add(p.resolve())
        # jsonl (data/processed)
        for p in d.rglob("*.jsonl"):
            if p.is_file():
                seen.add(p.resolve())
    for p in sorted(seen):
        yield Path(p)

def _load_corpus() -> List[Dict]:
    """
    Returnerer en liste med dicts:
    { "text": ..., "source": ..., "title": ..., "doc_type": ..., "version_date": ... (valgfri) }
    """
    docs: List[Dict] = []
    for p in _iter_kb_files():
        if p.suffix.lower() == ".md":
            raw = _read_text_file(p)
            clean = _strip_markdown_noise(raw)
            title = _title_from_markdown(raw, p.stem.replace("-", " "))
            doc_type = _infer_doc_type(p.name, clean)
            for ch in _chunk(clean):
                docs.append({
                    "text": ch,
                    "source": str(p).replace("\\", "/"),
                    "title": title,
                    "doc_type": doc_type,
                    "version_date": None,
                })
        elif p.suffix.lower() == ".jsonl":
            # Forvent format per linje: {"text": "...", "metadata": {...}}
            for line in _read_text_file(p).splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                txt = obj.get("text", "")
                meta = obj.get("metadata", {})
                if not txt.strip():
                    continue
                title = meta.get("title") or _title_from_markdown(txt, Path(meta.get("source", p.stem)).stem)
                doc_type = meta.get("doc_type") or _infer_doc_type(title, txt)
                docs.append({
                    "text": _strip_markdown_noise(txt),
                    "source": meta.get("source") or str(p).replace("\\", "/"),
                    "title": title,
                    "doc_type": doc_type,
                    "version_date": meta.get("version_date"),
                })
    return docs


# ---------- Index bygging ----------

def _ensure_index() -> None:
    global _VEC, _MTX, _META
    if _VEC is not None and _MTX is not None and _META:
        return

    corpus = _load_corpus()
    _META = corpus

    texts = [d["text"] for d in corpus]
    if not texts:
        # Tomt korpus; bygg en dummy-vektorizer
        _VEC = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
        _MTX = _VEC.fit_transform([""])
        return

    _VEC = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=1,
        strip_accents="unicode",
        lowercase=True,
        norm="l2",
        sublinear_tf=True,
        max_features=60000,
    )
    _MTX = _VEC.fit_transform(texts)


# ---------- Public API ----------

def search(query: str, k: int = 6) -> List[Dict]:
    """
    Returnerer topp k treff som liste av dicts:
    { "text", "source", "title", "score", "doc_type", "version_date" }
    """
    _ensure_index()
    if _VEC is None or _MTX is None or not _META:
        return []

    q = (query or "").strip()
    if not q:
        return []

    qvec = _VEC.transform([q])
    sims = linear_kernel(qvec, _MTX).ravel()
    if sims.size == 0:
        return []

    idx = sims.argsort()[::-1][: max(k, 1)]
    hits: List[Dict] = []
    for i in idx:
        m = _META[i]
        hits.append({
            "text": m["text"],
            "source": m["source"],
            "title": m["title"],
            "doc_type": m.get("doc_type"),
            "version_date": m.get("version_date"),
            "score": float(sims[i]),
        })
    return hits
