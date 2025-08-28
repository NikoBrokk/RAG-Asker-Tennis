
"""
FAISS index helper functions.

Modulen sørger for at nødvendige artefakter (vectors.npy, meta.jsonl og index.faiss)
finnes, og tilbyr hjelpere for å laste dem og bygge en FAISS-indeks på forespørsel.
Den bruker ``np.load(..., allow_pickle=True)`` for å kunne lese gamle .npy-filer som
kan inneholde picklet data.
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

from src.ingest import build_index  # importeres etter at DATA_DIR er opprettet

def _ensure_artifacts() -> None:
    """Sjekk om alle artefakter finnes, og bygg dem om de mangler."""
    missing = [p for p in (VEC_PATH, META_PATH, INDEX_PATH) if not p.exists()]
    if missing:
        print(f"[index] Mangler artefakter: {', '.join(str(p) for p in missing)} – bygger…")
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        build_index()  # skriver alle tre filene

def load_vectors() -> np.ndarray:
    """Last embedding-matrisen fra disk og normaliser radene."""
    _ensure_artifacts()
    X = np.load(VEC_PATH, allow_pickle=True)
    if X.dtype != np.float32:
        try:
            X = X.astype(np.float32)
        except Exception:
            pass
    if faiss is not None:
        faiss.normalize_L2(X)
    return X

def build_faiss_index() -> "faiss.Index":
    """Bygg og persister en FAISS-indeks for indre produkt."""
    X = load_vectors()
    if faiss is None:
        raise RuntimeError("FAISS er ikke tilgjengelig – installer faiss eller bruk TF‑IDF-modus.")
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    faiss.write_index(index, str(INDEX_PATH))
    return index

def load_meta() -> list:
    """Les meta.jsonl til en liste av dicts."""
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
        raise SystemExit("faiss er ikke installert; kan ikke bygge indeks. Installer faiss først.")
    build_faiss_index()
    print("Indeks skrevet til", INDEX_PATH)
