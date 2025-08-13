from __future__ import annotations
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from src.utils import env_flag
from src.ingest import build_index

# Konfig
USE_OPENAI = env_flag("USE_OPENAI", False)
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# Artefakter
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
INDEX_PATH = DATA_DIR / "index.faiss"
VEC_PATH = DATA_DIR / "vectors.npy"
META_PATH = DATA_DIR / "meta.jsonl"
VECTORIZER_PATH = DATA_DIR / "vectorizer.pkl"

# Lazy‑loaded
_vectors: np.ndarray | None = None
_meta: List[Dict] | None = None
_vectorizer = None
_faiss_index = None
_openai_client = None

# Optional deps
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

if USE_OPENAI:
    try:
        from openai import OpenAI
        _openai_client = OpenAI()
    except Exception:
        _openai_client = None

def _load_artifacts() -> Tuple[np.ndarray, List[Dict], object | None]:
    global _vectors, _meta, _vectorizer, _faiss_index
    if _vectors is not None and _meta is not None and (USE_OPENAI or _vectorizer is not None):
        return _vectors, _meta, _vectorizer

    if not VEC_PATH.exists() or not META_PATH.exists() or (not USE_OPENAI and not VECTORIZER_PATH.exists()):
        build_index(os.getenv("KB_DIR", "kb"))

    vectors = np.load(VEC_PATH).astype("float32")
    with META_PATH.open(encoding="utf-8") as f:
        meta = [json.loads(line) for line in f]

    vectorizer = None
    if not USE_OPENAI and VECTORIZER_PATH.exists():
        with VECTORIZER_PATH.open("rb") as vf:
            vectorizer = pickle.load(vf)

    # FAISS index (hvis finnes)
    if faiss is not None and INDEX_PATH.exists():
        try:
            _faiss_index = faiss.read_index(str(INDEX_PATH))
        except Exception:
            _faiss_index = None

    _vectors, _meta, _vectorizer = vectors, meta, vectorizer
    return vectors, meta, vectorizer

def _embed_query_openai(q: str) -> np.ndarray:
    if _openai_client is None:
        raise RuntimeError("OpenAI-klient ikke tilgjengelig – sjekk OPENAI_API_KEY og 'openai'‑pakke.")
    resp = _openai_client.embeddings.create(model=EMBED_MODEL, input=q)
    v = np.array(resp.data[0].embedding, dtype="float32")[None, :]
    # normaliser til L2=1
    if faiss is not None:
        faiss.normalize_L2(v)
    else:
        v = v / (np.linalg.norm(v) + 1e-12)
    return v

def _embed_query_tfidf(q: str) -> np.ndarray:
    _, _, vectorizer = _load_artifacts()
    if vectorizer is None:
        raise RuntimeError("TF‑IDF vectorizer mangler. Kjør ingest i TF‑IDF‑modus (USE_OPENAI=false).")
    v = vectorizer.transform([q]).toarray().astype("float32")
    # normaliser
    if faiss is not None:
        faiss.normalize_L2(v)
    else:
        v = v / (np.linalg.norm(v) + 1e-12)
    return v

def _search_scores(vectors: np.ndarray, qvec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returner (idx, scores) topp-k etter cosinuslikhet."""
    if faiss is not None and _faiss_index is not None:
        # FAISS søker på IP (dot product) fordi alle er normaliserte
        scores, idx = _faiss_index.search(qvec.astype("float32"), k)
        return idx[0], scores[0]
    # Brute force dot product
    scores = vectors @ qvec.T
    scores = scores.reshape(-1)
    idx = np.argsort(-scores)[:k]
    return idx, scores[idx]

def search(q: str, k: int = 6) -> List[Dict]:
    vectors, meta, _ = _load_artifacts()
    qvec = _embed_query_openai(q) if USE_OPENAI else _embed_query_tfidf(q)
    idx, scores = _search_scores(vectors, qvec, k)
    results: List[Dict] = []
    for i, s in zip(idx, scores):
        if int(i) < 0 or int(i) >= len(meta):
            continue
        m = meta[int(i)]
        results.append({
            "id": m["id"],
            "text": m["text"],
            "source": m["source"],
            "title": m.get("title"),
            "version_date": m.get("version_date"),
            "score": float(s),
        })
    return results
