from __future__ import annotations
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from src.utils import env_flag

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # vi faller tilbake til ren numpy hvis FAISS ikke finnes

# Les konfig fra miljø
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
USE_OPENAI = env_flag("USE_OPENAI", False)
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# Filstier
VEC_PATH = DATA_DIR / "vectors.npy"
META_PATH = DATA_DIR / "meta.jsonl"
VECTORIZER_PATH = DATA_DIR / "vectorizer.pkl"
INDEX_PATH = DATA_DIR / "index.faiss"

# Valgfri OpenAI-klient (kun brukt når USE_OPENAI=True)
_openai = None
if USE_OPENAI:
    try:
        from openai import OpenAI
        _openai = OpenAI()
    except Exception:
        _openai = None  # ved feil går vi over til TF‑IDF-modus

def _load() -> Tuple[np.ndarray, List[Dict], Optional[Dict[str, int]], Optional[Dict[str, float]]]:
    """
    Leser inn vektorene, metadata og TF‑IDF-ordbok hvis den finnes.
    """
    if not VEC_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Manglende indeksfiler – kjør ingest.py først.")

    vectors = np.load(VEC_PATH)  # shape (n_chunks, dim)
    meta: List[Dict] = []
    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))

    vocab: Optional[Dict[str, int]] = None
    idf: Optional[Dict[str, float]] = None
    # Kun i TF‑IDF-modus lagres vectorizer.pkl
    if not USE_OPENAI and VECTORIZER_PATH.exists():
        with VECTORIZER_PATH.open("rb") as vf:
            obj = pickle.load(vf)
            # Objektet kan være dict {"vocab": ..., "idf": ...}
            # eller en TfidfVectorizer – støtt begge
            if isinstance(obj, dict):
                vocab, idf = obj["vocab"], obj["idf"]
            else:
                vocab = getattr(obj, "vocabulary_", None)
                idf_array = getattr(obj, "idf_", None)
                if vocab is not None and idf_array is not None:
                    # Konverter idf-array til dict med samme rekkefølge som vokabularet
                    idf = {token: float(idf_array[idx]) for token, idx in vocab.items()}
    return vectors, meta, vocab, idf

# Enkel tokenisering (brukes bare i TF‑IDF-modus)
import re
_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)
def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())

def _embed_query_tfidf(q: str, vocab: Dict[str, int], idf: Dict[str, float]) -> np.ndarray:
    toks = _tokenize(q)
    vec = np.zeros(len(vocab), dtype="float32")
    tf: Dict[str, int] = {}
    for t in toks:
        if t in vocab:
            tf[t] = tf.get(t, 0) + 1
    for t, c in tf.items():
        j = vocab[t]
        vec[j] = float(c) * idf.get(t, 0.0)
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm

def _embed_query_openai(q: str) -> np.ndarray:
    if _openai is None:
        raise RuntimeError("OpenAI ikke tilgjengelig – sjekk USE_OPENAI og API-nøkkel.")
    r = _openai.embeddings.create(model=EMBED_MODEL, input=q)
    vec = np.array(r.data[0].embedding, dtype="float32")
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm

def _topk(vectors: np.ndarray, qvec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returnerer indeksene og poengene til de k mest relevante treffene.
    Bruker FAISS hvis tilgjengelig for raskere søk; ellers numpy dot.
    """
    if faiss is not None and INDEX_PATH.exists():
        # Last inn FAISS-indeksen én gang og bruk til søk
        index = faiss.read_index(str(INDEX_PATH))
        scores, inds = index.search(qvec.reshape(1, -1), k)
        return inds[0], scores[0]
    # Fallback: dot-produkt
    sims = vectors @ qvec
    order = np.argsort(-sims)[:k]
    return order, sims[order]

def search(q: str, k: int = 6) -> List[Dict]:
    """
    Søker i vektorindeksen og returnerer en liste med de k beste treffene.
    Hvert treff inneholder metadata fra meta.jsonl samt en ekstra 'score'.
    """
    vectors, meta, vocab, idf = _load()
    if not meta:
        return []

    # Velg embedding-metode basert på USE_OPENAI
    if USE_OPENAI and _openai is not None:
        qvec = _embed_query_openai(q)
    else:
        if vocab is None or idf is None:
            # Hvis vi mangler TF‑IDF-data må vi avbryte
            raise RuntimeError("TF‑IDF-data ikke tilgjengelig – kjør ingest.py.")
        qvec = _embed_query_tfidf(q, vocab, idf)

    inds, scores = _topk(vectors, qvec, max(k, 1))
    hits: List[Dict] = []
    for idx, score in zip(inds, scores):
        if idx < 0 or idx >= len(meta):
            continue
        m = meta[idx]
        # Lag kopi av metadata og legg til score (float)
        out = dict(m)
        out["score"] = float(score)
        hits.append(out)
    return hits
