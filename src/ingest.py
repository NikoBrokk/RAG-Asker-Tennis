from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.utils import env_flag

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

USE_OPENAI = env_flag("USE_OPENAI", False)
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
KB_DIR_DEFAULT = os.getenv("KB_DIR", "kb")

# --------- Korpushjelpere (gjenbruker logikken fra retrieve) ----------
import re
def _read_text_file(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def _strip(txt: str) -> str:
    txt = re.sub(r"```.*?```", " ", txt, flags=re.S)
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()

def _iter_docs(kb_root: Path) -> List[Dict]:
    out: List[Dict] = []
    for p in sorted(list(kb_root.rglob("*.md")) + list(Path("data/processed").rglob("*.jsonl"))):
        if not p.is_file():
            continue
        if p.suffix.lower() == ".md":
            raw = _read_text_file(p)
            clean = _strip(raw)
            if not clean:
                continue
            chunks = [clean[i:i+700] for i in range(0, len(clean), 700-120)]
            for ci, ch in enumerate(chunks):
        out.append({
            "text": ch,
            "source": str(p).replace("\\", "/"),
            "title": p.stem.replace("-", " "),
            "doc_type": None,
            "version_date": None,
            "page": None,
            "chunk_idx": ci,
            # Unngå backslash i f‑streng‑uttrykk; as_posix() gir alltid '/'.
            "id": f"{p.as_posix()}#{ci}",
        })
        else:
            ci = 0
            for line in _read_text_file(p).splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                txt = _strip(obj.get("text", ""))
                if not txt:
                    continue
                src = obj.get("metadata", {}).get("source") or str(p)
                out.append({
                    "text": txt,
                    "source": str(src).replace("\\", "/"),
                    "title": obj.get("metadata", {}).get("title"),
                    "doc_type": obj.get("metadata", {}).get("doc_type"),
                    "version_date": obj.get("metadata", {}).get("version_date"),
                    "page": obj.get("metadata", {}).get("page"),
                    "chunk_idx": ci,
                    "id": f"{Path(src).as_posix()}#{ci}",
                })
                ci += 1
    return out

# --------- OpenAI-embeddings eller TF-IDF til disk ----------
def _save_meta(meta: List[Dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (DATA_DIR / "meta.jsonl").open("w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

def _build_openai_embeddings(chunks: List[Dict]) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI()
    vecs: List[np.ndarray] = []
    for d in chunks:
        r = client.embeddings.create(model=EMBED_MODEL, input=d["text"])
        v = np.array(r.data[0].embedding, dtype="float32")
        v = v / (np.linalg.norm(v) + 1e-12)
        vecs.append(v)
    arr = np.vstack(vecs) if vecs else np.zeros((0, 1536), dtype="float32")
    return arr

def _build_tfidf_dense(chunks: List[Dict]) -> Tuple[np.ndarray, object]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    import pickle

    # Trekk ut tekstene, fallback til tom streng hvis listen er tom
    texts = [d["text"] for d in chunks] or [""]

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.95,
        strip_accents="unicode",
        lowercase=True,
        norm="l2",
        sublinear_tf=True,
        max_features=60000,
    )

    mtx = vec.fit_transform(texts)  # csr_matrix

    # Konverter til ndarray og rad-normaliser
    dense = mtx.toarray()
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    dense = (dense / (norms + 1e-12)).astype("float32")

    # Lagre vectorizer for gjenbruk i index/retrieve
    with (DATA_DIR / "vectorizer.pkl").open("wb") as f:
        pickle.dump(vec, f)

    return dense, vec

def _maybe_write_faiss(vectors: np.ndarray) -> None:
    if faiss is None or vectors.size == 0:
        return
    idx = faiss.IndexFlatIP(vectors.shape[1])
    idx.add(vectors)
    faiss.write_index(idx, str(DATA_DIR / "index.faiss"))

def build_index(kb_dir: str | Path = KB_DIR_DEFAULT) -> None:
    kb_root = Path(kb_dir)
    chunks = _iter_docs(kb_root)

    if USE_OPENAI:
        vectors = _build_openai_embeddings(chunks)
        np.save(DATA_DIR / "vectors.npy", vectors)
        _save_meta(chunks)
        _maybe_write_faiss(vectors)
        print(f"[ingest] OpenAI-embeddings for {len(chunks)} biter skrevet til data/vectors.npy og data/meta.jsonl.")
    else:
        vectors, _ = _build_tfidf_dense(chunks)
        np.save(DATA_DIR / "vectors.npy", vectors)
        _save_meta(chunks)
        _maybe_write_faiss(vectors)
        print(f"[ingest] TF-IDF vektorer for {len(chunks)} biter skrevet til data/vectors.npy og data/meta.jsonl.")
