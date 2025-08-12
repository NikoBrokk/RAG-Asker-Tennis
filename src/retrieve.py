import os
import pathlib
import faiss

INDEX_PATH = os.getenv("INDEX_PATH", "data/index.faiss")
_index = None  # lazy-loaded

def _load_if_exists():
    p = pathlib.Path(INDEX_PATH)
    if not p.exists():
        return None
    return faiss.read_index(str(p))

def get_index():
    """Returner en FAISS-indeks. Bygger den hvis mangler."""
    global _index
    if _index is not None:
        return _index

    _index = _load_if_exists()
    if _index is None:
        # Bygg i farten (skyen) første gang
        from src.ingest import build_index
        _index = build_index()  # sørger for at index skrives til INDEX_PATH
    return _index

# eksisterende funksjon som brukes av answer.py
def search(query_embedding, top_k=5):
    index = get_index()
    D, I = index.search(query_embedding, top_k)
    return D, I
