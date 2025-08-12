# path: src/ingest.py
import os, json, pathlib
from pathlib import Path
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
from src.utils import read_markdown_files, simple_chunks

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

INDEX_PATH = Path("data/index.faiss")
VEC_PATH   = Path("data/vectors.npy")
META_PATH  = Path("data/meta.jsonl")

def _embed_texts(texts):
    client = OpenAI(api_key=OPENAI_API_KEY)
    out = []
    B = 128
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        res = client.embeddings.create(model=EMBED_MODEL, input=batch)
        out.extend([e.embedding for e in res.data])
    return out

def build_index(kb_dir: str = "kb"):
    """
    Leser dokumenter fra kb_dir, lager overlappende biter, embedder dem
    og bygger/lagrer en FAISS-indeks. Skriver også vectors.npy og meta.jsonl.
    """
    docs = read_markdown_files(kb_dir)
    if not docs:
        raise FileNotFoundError(f"Fant ingen .md eller .txt i {kb_dir}/ – legg dokumenter der.")

    chunks, metas = [], []
    for record in docs:
        text = record["text"]
        for i, chunk in enumerate(simple_chunks(text)):
            chunk = chunk.strip()
            if not chunk:
                continue
            chunks.append(chunk)
            metas.append({
                "id": f"{record['source']}#{i}",
                "text": chunk,
                "source": record['source'],
                "title": record['title'],
                "version_date": record['version_date'],
            })
    if not chunks:
        raise ValueError("Ingen tekst å indeksere etter chunking.")

    # Embedd alle biter
    vecs = np.array(_embed_texts(chunks), dtype="float32")
    faiss.normalize_L2(vecs)

    # Lag FAISS‑indeks
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    # Lagre artefakter
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    np.save(VEC_PATH, vecs)
    with META_PATH.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"[ingest] Skrev {len(chunks)} vektorer til {INDEX_PATH}, {VEC_PATH} og metadata til {META_PATH}")
    return index
