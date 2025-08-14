from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src.utils import env_flag

# --- Konfig ---
KB_DIRS = [Path("kb"), Path("data/processed")]
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
USE_OPENAI = env_flag("USE_OPENAI", False)
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# OpenAI klient (kun hvis USE_OPENAI)
_openai = None
if USE_OPENAI:
    try:
        from openai import OpenAI
        _openai = OpenAI()
    except Exception:
        _openai = None  # vi klarer fortsatt TF-IDF

# TF-IDF state
_VEC: Optional[TfidfVectorizer] = None
_MTX = None  # scipy sparse
_META: List[Dict] = []  # én entry per rad i _MTX

# OpenAI state
_EMB: Optional[np.ndarray] = None  # shape (n_chunks, dim)
_META_OAI: List[Dict] = []

# ---------- Utils ----------
import re
def _read_text_file(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def _strip_markdown_noise(txt: str) -> str:
    txt = re.sub(r"```.*?```", " ", txt, flags=re.S)
    txt = re.sub(r" ", " ", txt, flags=re.S)
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()

def _title_from_markdown(txt: str, fallback: str) -> str:
    m = re.search(r"^\s*#\s+(.+)$", txt, flags=re.M)
    if m:
        return m.group(1).strip()
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
        for p in d.rglob("*.md"):
            if p.is_file():
                seen.add(p.resolve())
        for p in d.rglob("*.jsonl"):
            if p.is_file():
                seen.add(p.resolve())
    for p in sorted(seen):
        yield Path(p)

def _load_corpus() -> List[Dict]:
    """
    Returnerer en liste med dicts:
    { "text", "source", "title", "doc_type", "version_date", "page", "chunk_idx", "id" }
    """
    docs: List[Dict] = []
    for p in _iter_kb_files():
        source_path = str(p).replace("\\", "/")
        if p.suffix.lower() == ".md":
            raw = _read_text_file(p)
            clean = _strip_markdown_noise(raw)
            title = _title_from_markdown(raw, p.stem.replace("-", " "))
            doc_type = _infer_doc_type(p.name, clean)
            chunks = _chunk(clean)
            for ci, ch in enumerate(chunks):
                docs.append({
                    "text": ch,
                    "source": source_path,
                    "title": title,
                    "doc_type": doc_type,
                    "version_date": None,
                    "page": None,
                    "chunk_idx": ci,
                    "id": f"{source_path}#{ci}",
                })
        elif p.suffix.lower() == ".jsonl":
            ci = 0
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
                txt_clean = _strip_markdown_noise(txt)
                title = meta.get("title") or _title_from_markdown(txt, Path(meta.get("source", p.stem)).stem)
                doc_type = meta.get("doc_type") or _infer_doc_type(title, txt)
                src = (meta.get("source") or source_path).replace("\\", "/")
                page = meta.get("page")
                docs.append({
                    "text": txt_clean,
                    "source": src,
                    "title": title,
                    "doc_type": doc_type,
                    "version_date": meta.get("version_date"),
                    "page": page,
                    "chunk_idx": ci,
                    "id": f"{src}#{ci}",
                })
                ci += 1
    return docs

# ---------- Index bygging ----------
def _ensure_index_tfidf() -> None:
    global _VEC, _MTX, _META
    if _VEC is not None and _MTX is not None and _META:
        return
    corpus = _load_corpus()
    _META = corpus
    texts = [d["text"] for d in corpus]
    if not texts:
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

def _ensure_index_openai() -> None:
    global _EMB, _META_OAI
    if _EMB is not None and _META_OAI:
        return
    vec_path = DATA_DIR / "vectors.npy"
    meta_path = DATA_DIR / "meta.jsonl"
    if not vec_path.exists() or not meta_path.exists():
        raise FileNotFoundError("OpenAI-indeks mangler (kjør src.ingest i USE_OPENAI=true).")
    _EMB = np.load(vec_path)
    _META_OAI = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            _META_OAI.append(json.loads(line))

# ---------- Public API ----------
def search(query: str, k: int = 6) -> List[Dict]:
    """
    Returnerer topp k treff som liste av dicts:
    { "text", "source", "title", "score", "doc_type", "version_date", "page", "chunk_idx", "id" }
    """
    if USE_OPENAI and _openai is not None:
        _ensure_index_openai()
        # Embedd spørringen
        r = _openai.embeddings.create(model=EMBED_MODEL, input=query)
        qvec = np.array(r.data[0].embedding, dtype="float32")
        qvec = qvec / (np.linalg.norm(qvec) + 1e-12)
        # Kosinus ~ dot (siden alt er normalisert)
        sims = _EMB @ qvec  # type: ignore
        order = np.argsort(-sims)[:k]
        out: List[Dict] = []
        for idx in order:
            m = dict(_META_OAI[idx])
            m["score"] = float(sims[idx])
            out.append(m)
        return out

    # TF-IDF (din eksisterende flyt)
    _ensure_index_tfidf()
    qvec = _VEC.transform([query])  # type: ignore
    sims = linear_kernel(qvec, _MTX).ravel()  # type: ignore
    order = np.argsort(-sims)[:k]
    out: List[Dict] = []
    for idx in order:
        m = dict(_META[idx])
        m["score"] = float(sims[idx])
        out.append(m)
    return out
