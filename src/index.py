// path: src/index.py
"""
FAISS index helper functions.

This module ensures the existence of vector and metadata artifacts, and
provides utilities to load them and build a FAISS index on demand.  It is
adapted from the original RAG‑Asker‑Tennis code and modified to safely load
numpy arrays even when they were saved with pickling enabled.  Always ensure
that your ingestion pipeline writes embeddings as plain float32 arrays using
``np.save``.
"""

import json
from pathlib import Path

import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = DATA_DIR / "index.faiss"
VEC_PATH = DATA_DIR / "vectors.npy"
META_PATH = DATA_DIR / "meta.jsonl"

from src.ingest import build_index  # imported after DATA_DIR exists

def _ensure_artifacts() -> None:
    """Ensure that vectors, metadata and index files exist; build if missing."""
    missing = [p for p in [VEC_PATH, META_PATH, INDEX_PATH] if not p.exists()]
    if missing:
        print(
            f"[index] Mangler artefakter: {', '.join(str(p) for p in missing)} – bygger…"
        )
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        build_index()  # writes all three files

def load_vectors() -> np.ndarray:
    """Load the embedding matrix from disk and normalise its rows.

    The returned matrix is of dtype ``float32`` and has shape (N, D).  If the
    underlying file was saved with pickle enabled, ``allow_pickle=True``
    prevents numpy from refusing to load it.  The embeddings are normalised
    using the FAISS helper to ensure cosine similarity behaves as expected.
    """
    _ensure_artifacts()
    X = np.load(VEC_PATH, allow_pickle=True)
    # Cast to float32 if necessary
    if X.dtype != np.float32:
        try:
            X = X.astype(np.float32)
        except Exception:
            pass
    # Ensure normalisation of each vector (L2 normalise)
    if faiss is not None:
        faiss.normalize_L2(X)
    return X

def build_faiss_index() -> "faiss.Index":
    """Build and persist a FAISS inner product index from the stored vectors."""
    X = load_vectors()
    if faiss is None:
        raise RuntimeError(
            "FAISS er ikke tilgjengelig – installer faiss eller bruk TF-IDF-modus."
        )
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    faiss.write_index(index, str(INDEX_PATH))
    return index

def load_meta() -> list:
    """Load the metadata JSONL file into a list of dicts."""
    _ensure_artifacts()
    rows = []
    with META_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows

if __name__ == "__main__":
    if faiss is None:
        raise SystemExit(
            "faiss er ikke installert; kan ikke bygge indeks. Installer faiss først."
        )
    build_faiss_index()
    print("Indeks skrevet til", INDEX_PATH)
