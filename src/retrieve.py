from __future__ import annotations
import json, os, pickle, re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from src.utils import env_flag
from src.ingest import DATA_DIR, INDEX_PATH, VEC_PATH, META_PATH, VECTORIZER_PATH, KB_DIR, build_index

USE_OPENAI = env_flag("USE_OPENAI", False)
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# Valgfri FAISS
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

# Valgfri OpenAI
_openai = None
if USE_OPENAI:
    try:
        from openai import OpenAI
        _openai = OpenAI()
    except Exception:
        _openai = None

_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)
def _tok(s: str) -> List[str]:
    return _TOKEN_RE.findall(s.lower())

# Cache
_vectors: np.ndarray | None = None
_meta: List[Dict] | None = None
_vocab: Dict[str, int] | None = None
_idf: Dict[str, float] | None = None
_faiss = None

def _load() -> Tuple[np.ndarray, List[Dict], Dict[str, int] | None, Dict[str, float] | None]:
    global _vectors, _meta, _vocab, _idf, _faiss
    if _vectors is not None and _meta is not None and (USE_OPENAI or (_vocab is not None and _idf is not None)):
        return _vectors, _meta, _vocab, _idf
    # bygg hvis mangler
    need = any(not p.exists() for p in [VEC_PATH, META_PATH]) or (not USE_OPENAI and not VECTORIZER_PATH.exists())
    if need:
        build_index(KB_DIR)
    vectors = np.load(VEC_PATH).astype("float32")
    meta: List[Dict] = [json.loads(l) for l in META_PATH.read_text(encoding="utf-8").splitlines()]
    vocab = idf = None
    if not USE_OPENAI and VECTORIZER_PATH.exists():
        with VECTORIZER_PATH.open("rb") as vf:
            d = pickle.load(vf)
            vocab, idf = d["vocab"], d["idf"]
    # FAISS
    _faiss = None
    if faiss is not None and INDEX_PATH.exists():
        try:
            _faiss = faiss.read_index(str(INDEX_PATH))
        except Exception:
            _faiss = None
    _vectors, _meta, _vocab, _idf = vectors, meta, vocab, idf
    return vectors, meta, vocab, idf

def _embed_query_openai(q: str) -> np.ndarray:
    if _openai is None:
        raise RuntimeError("OpenAI ikke tilgjengelig – kan ikke embedde spørring.")
    r = _openai.embeddings.create(model=EMBED_MODEL, input=q)
    v = np.asarray(r.data[0].embedding, dtype="float32")[None, :]
    v = v / (np.linalg.norm(v) + 1e-12)
    return v

def _embed_query_tfidf(q: str, vocab: Dict[str, int], idf: Dict[str, float]) -> np.ndarray:
    v = np.zeros((1, len(vocab)), dtype="float32")
    tf: Dict[str, int] = {}
    for t in _tok(q):
        if t in vocab:
            tf[t] = tf.get(t, 0) + 1
    for t, c in tf.items():
        v[0, vocab[t]] = float(c) * idf.get(t, 0.0)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v

def _topk(vectors: np.ndarray, qvec: np.ndarray, k: int):
    # FAISS hvis tilgjengelig
    if faiss is not None and INDEX_PATH.exists():
        try:
            idx = faiss.read_index(str(INDEX_PATH))
            scores, inds = idx.search(qvec.astype("float32"), k)
            return inds[0], scores[0]
        except Exception:
            pass
    # Brute‑force kosinus (dot på normaliserte vektorer)
    scores = vectors @ qvec.T
    scores = scores.reshape(-1)
    order = np.argsort(-scores)[:k]
    return order, scores[order]

def search(q: str, k: int = 6) -> List[Dict]:
    vectors, meta, vocab, idf = _load()
    qvec = _embed_query_openai(q) if USE_OPENAI else _embed_query_tfidf(q, vocab, idf)
    inds, scores = _topk(vectors, qvec, k)
    out: List[Dict] = []
    for i, s in zip(inds, scores):
        ii = int(i)
        if ii < 0 or ii >= len(meta): continue
        m = meta[ii]
        out.append({
            "id": m["id"],
            "text": m["text"],
            "source": m["source"],
            "title": m.get("title"),
            "version_date": m.get("version_date"),
            "score": float(s),
        })
    return out
