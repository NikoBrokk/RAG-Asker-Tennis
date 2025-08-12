import json
from pathlib import Path

import faiss
import numpy as np

# Artefaktstier
INDEX_PATH = Path("data/index.faiss")
VEC_PATH = Path("data/vectors.npy")
META_PATH = Path("data/meta.jsonl")

# Bygger hele indeksen (vectors.npy, meta.jsonl, index.faiss) hvis mangler
# hentes fra src.ingest.build_index
from src.ingest import build_index


def _ensure_artifacts():
    """
    Sørger for at vectors/meta/index finnes. Hvis ikke, bygges de fra kb/.
    """
    missing = [p for p in [VEC_PATH, META_PATH, INDEX_PATH] if not p.exists()]
    if missing:
        print(f"[index] Mangler artefakter: {', '.join(str(p) for p in missing)} – bygger…")
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        build_index()  # skriver alle tre filene


def load_vectors() -> np.ndarray:
    _ensure_artifacts()
    X = np.load(VEC_PATH).astype("float32")
    faiss.normalize_L2(X)
    return X


def build_faiss_index() -> faiss.Index:
    X = load_vectors()
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    faiss.write_index(index, str(INDEX_PATH))
    return index


def load_meta():
    _ensure_artifacts()
    with META_PATH.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f]


if __name__ == "__main__":
    build_faiss_index()
    print("Indeks skrevet til", INDEX_PATH)
