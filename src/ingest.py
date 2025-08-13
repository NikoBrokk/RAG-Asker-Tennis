from __future__ import annotations
import json, math, os, pickle, re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from src.utils import read_markdown_files, simple_chunks, env_flag

# Konfig
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
KB_DIR = os.getenv("KB_DIR", "kb")
USE_OPENAI = env_flag("USE_OPENAI", False)
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# Filer
INDEX_PATH = DATA_DIR / "index.faiss"
VEC_PATH = DATA_DIR / "vectors.npy"
META_PATH = DATA_DIR / "meta.jsonl"
VECTORIZER_PATH = DATA_DIR / "vectorizer.pkl"

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
        _openai = None  # kjør likevel i TF‑IDF hvis OpenAI ikke er tilgjengelig

# ---------- Manuell TF‑IDF ----------
_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)
def _tok(s: str) -> List[str]:
    return _TOKEN_RE.findall(s.lower())

def _idf(docs_tokens: List[List[str]]) -> Dict[str, float]:
    N = len(docs_tokens)
    df: Dict[str, int] = {}
    for toks in docs_tokens:
        seen = set()
        for t in toks:
            if t not in seen:
                df[t] = df.get(t, 0) + 1
                seen.add(t)
    return {t: math.log((N + 1) / (c + 1)) + 1.0 for t, c in df.items()}

def _tfidf_matrix(chunks: List[str]) -> Tuple[np.ndarray, Dict[str, int], Dict[str, float]]:
    docs_toks = [_tok(c) for c in chunks]
    idf = _idf(docs_toks)
    vocab = {t: i for i, t in enumerate(sorted(idf.keys()))}
    M = np.zeros((len(chunks), len(vocab)), dtype="float32")
    for i, toks in enumerate(docs_toks):
        tf: Dict[str, int] = {}
        for t in toks:
            if t in vocab:
                tf[t] = tf.get(t, 0) + 1
        for t, c in tf.items():
            M[i, vocab[t]] = float(c) * idf[t]
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    M = M / norms
    return M, vocab, idf

# ---------- OpenAI embeddings ----------
def _embed_openai(texts: List[str]) -> np.ndarray:
    if _openai is None:
        raise RuntimeError("OpenAI ikke tilgjengelig – sjekk OPENAI_API_KEY og 'openai'-pakke.")
    vecs: List[List[float]] = []
    for t in texts:
        r = _openai.embeddings.create(model=EMBED_MODEL, input=t)
        vecs.append(r.data[0].embedding)
    arr = np.asarray(vecs, dtype="float32")
    # L2‑normaliser
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms

def _build_faiss(vecs: np.ndarray):
    if faiss is None:
        return None
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(INDEX_PATH))
    return index

def build_index(kb_dir: str = KB_DIR):
    docs = read_markdown_files(kb_dir)
    if not docs:
        raise FileNotFoundError(f"Ingen .md/.txt i {kb_dir}/")
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
    if not chunks:
        raise ValueError("Ingen tekst å indeksere.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if USE_OPENAI and _openai is not None:
        vecs = _embed_openai(chunks)
        # Skriv artefakter
        np.save(VEC_PATH, vecs)
        with META_PATH.open("w", encoding="utf-8") as f:
            for m in metas: f.write(json.dumps(m, ensure_ascii=False) + "\n")
        VECTORIZER_PATH.unlink(missing_ok=True)  # ikke brukt i OpenAI‑modus
        _build_faiss(vecs)
        print(f"[ingest] OpenAI‑embeddings skrevet: {len(chunks)} biter.")
        return

    # Fallback: TF‑IDF
    vecs, vocab, idf = _tfidf_matrix(chunks)
    np.save(VEC_PATH, vecs)
    with META_PATH.open("w", encoding="utf-8") as f:
        for m in metas: f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with VECTORIZER_PATH.open("wb") as vf:
        pickle.dump({"vocab": vocab, "idf": idf}, vf)
    _build_faiss(vecs)
    print(f"[ingest] TF‑IDF‑indeks skrevet: {len(chunks)} biter.")

if __name__ == "__main__":
    build_index(KB_DIR)
