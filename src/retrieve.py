import os, json, numpy as np, faiss
from dotenv import load_dotenv
from openai import OpenAI
from src.ingest import build_index
import pathlib

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

INDEX_PATH = "data/index.faiss"
META_PATH  = "data/meta.jsonl"

_index, _meta = None, None  # caches

def _load_artifacts():
    """Laster (eller bygger) indeksen og metadata."""
    global _index, _meta
    if _index is not None and _meta is not None:
        return _index, _meta

    if pathlib.Path(INDEX_PATH).exists() and pathlib.Path(META_PATH).exists():
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, encoding="utf-8") as f:
            meta = [json.loads(l) for l in f]
    else:
        # bygg fra kb/
        index = build_index()
        with open(META_PATH, encoding="utf-8") as f:
            meta = [json.loads(l) for l in f]
    _index, _meta = index, meta
    return index, meta

def _embed_query(q: str):
    client = OpenAI(api_key=OPENAI_API_KEY)
    res = client.embeddings.create(model=EMBED_MODEL, input=[q])
    v = np.array([res.data[0].embedding], dtype="float32")
    faiss.normalize_L2(v)
    return v

def search(q: str, k: int = 6):
    """
    Embedder spørringen, gjør nearest‑neighbor‑søk og returnerer
    en liste av treff med id, text, source, meta og score.
    """
    index, meta = _load_artifacts()
    v = _embed_query(q)
    D, I = index.search(v, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1 or idx >= len(meta):
            continue
        m = meta[idx]
        results.append({
            "id": m["id"],
            "text": m["text"],
            "source": m["source"],
            "title": m.get("title"),
            "version_date": m.get("version_date"),
            "score": float(score),
        })
    return results
