import os
import pathlib
import faiss
from openai import OpenAI
from src.utils import read_markdown_files, simple_chunks

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
INDEX_PATH = os.getenv("INDEX_PATH", "data/index.faiss")

def _embed_texts(texts):
    client = OpenAI()
    out = []
    B = 128
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        res = client.embeddings.create(model=EMBED_MODEL, input=batch)
        out.extend([e.embedding for e in res.data])
    return out

def build_index(kb_dir: str = "kb"):
    docs = read_markdown_files(kb_dir)
    chunks = [c.strip() for c in simple_chunks(docs) if isinstance(c, str) and c.strip()]
    if not chunks:
        raise ValueError("Ingen tekst å indeksere. Legg .md/.txt i kb/.")

    vecs = _embed_texts(chunks)

    import numpy as np
    xb = np.array(vecs, dtype="float32")
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)

    pathlib.Path(os.path.dirname(INDEX_PATH)).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    print(f"[ingest] Wrote FAISS index to {INDEX_PATH} ({xb.shape[0]} chunks)")
    return index
