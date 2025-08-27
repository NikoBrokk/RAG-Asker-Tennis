// path: src/retrieve.py
"""
Utilities for loading and querying the knowledge base.

This module provides a simple TF‑IDF based retriever as well as an optional
OpenAI‑embedding powered retriever.  The code is largely adapted from the
original RAG‑Asker‑Tennis repository but has been refactored to be more
robust.  In particular the OpenAI path now allows loading pickled numpy
files safely.  Without ``allow_pickle=True`` numpy refuses to load arrays
that contain arbitrary Python objects, which can happen if a legacy
``vectors.npy`` was saved using ``pickle`` instead of the default plain
array format.  To avoid surprises in the future, ingestion code should
always save embeddings with ``np.save`` on a dense numeric array, never
object arrays.

Functions:
    search(query: str, k: int = 6) -> List[Dict]:
        Retrieve the top‐k chunks matching the given query.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Iterable, Optional

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
        from openai import OpenAI  # type: ignore
        _openai = OpenAI()
    except Exception:
        # We still support TF-IDF mode if OpenAI cannot be imported
        _openai = None

# TF‑IDF state
_VEC: Optional[TfidfVectorizer] = None
_MTX = None  # scipy sparse
_META: List[Dict] = []  # one entry per row in _MTX

# OpenAI state
_EMB: Optional[np.ndarray] = None  # shape (n_chunks, dim)
_META_OAI: List[Dict] = []

# ---------- Utils ----------
import re

def _read_text_file(p: Path) -> str:
    """Read a text file using utf‑8 with ignore fallback."""
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def _strip_markdown_noise(txt: str) -> str:
    """Remove fenced code blocks and compress whitespace in a markdown string."""
    txt = re.sub(r"```.*?```", " ", txt, flags=re.S)
    txt = re.sub(r" ", " ", txt, flags=re.S)
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()

def _title_from_markdown(txt: str, fallback: str) -> str:
    """Extract the first top‑level heading or fallback to the first non‐empty line."""
    m = re.search(r"^\s*#\s+(.+)$", txt, flags=re.M)
    if m:
        return m.group(1).strip()
    for line in txt.splitlines():
        s = line.strip()
        if s:
            return s[:120]
    return fallback

def _infer_doc_type(name: str, text: str) -> str:
    """Heuristically infer the document type based on filename and content."""
    low = (name + " " + text[:400]).lower()
    # Rules / conditions to map into a small set of document categories
    if any(w in low for w in ["vilkår", "terms", "betingelser", "angrerett", "personvern", "gdpr", "privacy"]):
        return "regel"
    if any(w in low for w in ["innmelding", "medlemsfordel", "kontingent", "medlemskontingent"]):
        return "medlemskap"
    if any(w in low for w in ["stiftet", "medlemmer", "har rundt", "medlemstall",
                              "grusbaner", "innendørsbaner", "daglig leder", "sportslig leder", "hovedtrener"]):
        return "info"
    if any(w in low for w in ["pris", "timepris", "avgift", "kostnad", "rabatt"]):
        return "pris"
    if any(w in low for w in ["booking", "banebooking", "reserver", "matchi", "baneregler", "bane ", "baner"]):
        return "booking"
    if any(w in low for w in ["håndbok"]):
        return "håndbok"
    return "annet"

def _chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks of approximately ``size`` characters."""
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
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
    """Yield all markdown and jsonl files from configured knowledge base directories."""
    seen: set[Path] = set()
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
    """Load and chunk documents from the knowledge base into a list of metadata dicts."""
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
                source_raw = meta.get("source")
                title = meta.get("title") or _title_from_markdown(
                    txt,
                    Path(source_raw or p.stem).stem,
                )
                doc_type = meta.get("doc_type") or _infer_doc_type(title, txt)
                src = (source_raw or source_path).replace("\\", "/")
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
    """Lazy construction of the TF‑IDF index.

    If there are no documents to index, a dummy vectorizer and matrix are
    created so that subsequent calls to ``search`` do not raise errors.  The
    dummy matrix has one row and one column.  This behaviour mirrors the
    original repository and avoids a ValueError from scikit‑learn about an
    empty vocabulary.
    """
    global _VEC, _MTX, _META
    # If already built, do nothing
    if _VEC is not None and _MTX is not None and _META:
        return
    corpus = _load_corpus()
    _META = corpus
    texts = [d["text"] for d in corpus]
    # When no texts exist, build a trivial vocabulary on a dummy token
    if not texts:
        _VEC = TfidfVectorizer(ngram_range=(1, 2))
        # fit on a single dummy token to avoid empty vocabulary errors
        _MTX = _VEC.fit_transform(["dummy"])
        return
    # Otherwise build a full vectorizer.  With only one document, setting
    # ``max_df`` to a value < 1 will trigger an error because it would be
    # smaller than ``min_df``.  Therefore we conditionally omit ``max_df``
    # when the corpus is tiny.
    if len(texts) < 2:
        _VEC = TfidfVectorizer(
            ngram_range=(1, 2),
            strip_accents="unicode",
            lowercase=True,
            norm="l2",
            sublinear_tf=True,
            max_features=60000,
        )
    else:
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
    """Load the OpenAI embeddings and metadata from disk if not already loaded.

    We call ``np.load`` with ``allow_pickle=True`` to support legacy .npy files
    that may have been saved with pickle.  Modern ingestion code writes
    embeddings as plain float32 arrays and will not require pickle.
    """
    global _EMB, _META_OAI
    if _EMB is not None and _META_OAI:
        return
    vec_path = DATA_DIR / "vectors.npy"
    meta_path = DATA_DIR / "meta.jsonl"
    if not vec_path.exists() or not meta_path.exists():
        raise FileNotFoundError("OpenAI-indeks mangler (kjør src.ingest i USE_OPENAI=true).")
    try:
        # allow_pickle=True is necessary if the file contains object arrays
        _EMB = np.load(vec_path, allow_pickle=True)
    except ValueError as e:
        # Provide a more helpful error message with remediation guidance
        raise ValueError(
            f"Kunne ikke laste embeddings fra {vec_path}: {e}. "
            "Forsikre deg om at filen er lagret med np.save på en 2D float32-matrise."
        )
    # Cast to float32 and normaliser dersom nødvendig
    if _EMB.dtype != np.float32:
        try:
            _EMB = _EMB.astype(np.float32)
        except Exception:
            pass
    # Ensure 2D shape (n, d) – reshape 1D arrays if necessary
    if _EMB.ndim == 1:
        _EMB = _EMB.reshape(-1, 1)
    _META_OAI = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                _META_OAI.append(json.loads(line))
            except json.JSONDecodeError:
                continue

# ---------- Public API ----------
def search(query: str, k: int = 6) -> List[Dict]:
    """Return the top ``k`` hits as a list of metadata dicts with scores.

    Each hit contains ``text``, ``source``, ``title``, ``doc_type``, ``version_date``,
    ``page``, ``chunk_idx`` and ``score``.  When OpenAI mode is active, it uses
    cosine similarity between the query embedding and precomputed embeddings.
    Otherwise, TF‑IDF cosine similarities are used.
    """
    if USE_OPENAI and _openai is not None:
        _ensure_index_openai()
        # Embed the query using the configured embedding model
        r = _openai.embeddings.create(model=EMBED_MODEL, input=query)
        qvec = np.array(r.data[0].embedding, dtype="float32")
        qvec = qvec / (np.linalg.norm(qvec) + 1e-12)
        # Cosine similarity reduces to dot product when all vectors are normalised
        sims = _EMB @ qvec  # type: ignore
        order = np.argsort(-sims)[:k]
        out: List[Dict] = []
        for idx in order:
            m = dict(_META_OAI[int(idx)])
            m["score"] = float(sims[int(idx)])
            out.append(m)
        return out
    # TF‑IDF fallback
    _ensure_index_tfidf()
    qvec = _VEC.transform([query])  # type: ignore
    sims = linear_kernel(qvec, _MTX).ravel()  # type: ignore
    order = np.argsort(-sims)[:k]
    out: List[Dict] = []
    for idx in order:
        m = dict(_META[int(idx)])
        m["score"] = float(sims[int(idx)])
        out.append(m)
    return out
