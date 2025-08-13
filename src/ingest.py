from __future__ import annotations
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from src.utils import read_markdown_files, simple_chunks, env_flag

# ---- Konfig via env ----
USE_OPENAI = env_flag("USE_OPENAI", False)
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# ---- Valgfri FAISS ----
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

# ---- Valgfri OpenAI ----
_openai_client = None
if USE_OPENAI:
    try:
        from openai import OpenAI
        _openai_client = OpenAI()
    except Exception:
        _openai_client = None

# ---- Artefaktstier ----
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
INDEX_PATH = DATA_DIR / "index.faiss"
VEC_PATH = DATA_DIR / "vectors.npy"
META_PATH = DATA_DIR / "meta.jsonl"
VECTORIZER_PATH = DATA_DIR / "vectorizer.pkl"  # kun TF-IDF

def _embed_openai(texts: List[str]) -> np.ndarray:
    if _openai_client is None:
        raise RuntimeError("OpenAI-klient ikke tilgjengelig – sjekk OPENAI_API_KEY og installert 'openai'.")
    # batch for å unngå very long payloads
    out: List[List[float]] = []
    for t in texts:
        resp = _openai_client.embeddings.create(model=EMBED_MODEL, input=t)
        out.append(resp.data[0].embedding)
    arr = np.array(out, dtype="float32")
    return arr

def _embed_tfidf(chunks: List[str]) -> Tuple[np.ndarray, object]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words=None)
    vecs = vectorizer.fit_transform(chunks).toarray().astype("float32")
    return vecs, vectorizer

def _normalise_rows(mat: np.ndarray) -> np.ndarray:
    if faiss is not None:
        faiss.normalize_L2(mat)
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

def _build_index_from_vectors(vecs: np.ndarray):
    if faiss is None:
        return None
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(INDEX_PATH))
    return index

def build_index(kb_dir: str = "kb"):
    docs = read_markdown_files(kb_dir)
    if not docs:
        raise FileNotFoundError(f"Fant ingen .md/.txt i {kb_dir}/")

    chunks: List[str] = []
    metas: List[Dict] = []
    for d in docs:
        for i, c in enumerate(simple_chunks(d["text"])):
            c = c.strip()
            if not c: continue
            chunks.append(c)
            metas.append({
                "id": f"{d['source']}#{i}",
                "text": c,
                "source": d["source"],
                "title": d["title"],
                "version_date": d["version_date"],
            })

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if USE_OPENAI:
        vecs = _embed_openai(chunks)
        vecs = _normalise_rows(vecs)
        # TF-IDF‑artefakt trengs ikke i OpenAI‑modus, men vi skriver vectors/meta for fallback
        with META_PATH.open("w", encoding="utf-8") as f:
            for m in metas: f.write(json.dumps(m, ensure_ascii=False) + "\n")
        np.save(VEC_PATH, vecs)
        idx = _build_index_from_vectors(vecs)
        print(f"[ingest] OpenAI embeddings for {len(chunks)} biter. " +
              (f"FAISS skrevet til {INDEX_PATH}" if idx is not None else "FAISS ikke tilgjengelig."))
        return idx
    else:
        vecs, vectorizer = _embed_tfidf(chunks)
        vecs = _normalise_rows(vecs)
        with META_PATH.open("w", encoding="utf-8") as f:
            for m in metas: f.write(json.dumps(m, ensure_ascii=False) + "\n")
        np.save(VEC_PATH, vecs)
        with VECTORIZER_PATH.open("wb") as vf:
            pickle.dump(vectorizer, vf)
        idx = _build_index_from_vectors(vecs)
        print(f"[ingest] TF‑IDF embeddings for {len(chunks)} biter skrevet til {VEC_PATH} og {META_PATH}. " +
              (f"FAISS skrevet til {INDEX_PATH}" if idx is not None else "FAISS ikke tilgjengelig."))
        return idx

if __name__ == "__main__":
    build_index()
