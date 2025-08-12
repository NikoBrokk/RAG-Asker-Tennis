import json, time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import os
from src.utils import read_kb_files, simple_chunks

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

def embed_batch(texts):
    res = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in res.data]

def build_corpus():
    docs = []
    for path, text in read_kb_files("kb"):
        if not text.strip():
            continue
        chunks = simple_chunks(text)
        for i, ch in enumerate(chunks):
            docs.append({
                "id": f"{path}#chunk-{i}",
                "source": path,
                "text": ch,
                "created_at": int(time.time())
            })
    return docs

if __name__ == "__main__":
    corpus = build_corpus()
    if not corpus:
        raise SystemExit("Ingen tekst funnet i kb/. Legg inn .md/.txt/.pdf og prøv igjen.")
    Path("data").mkdir(exist_ok=True)
    with open("data/corpus.jsonl", "w", encoding="utf-8") as f:
        for d in corpus: f.write(json.dumps(d, ensure_ascii=False) + "\n")
    # embed
    all_texts = [d["text"] for d in corpus]
    vecs = embed_batch(all_texts)
    import numpy as np
    np.save("data/vectors.npy", np.array(vecs, dtype="float32"))
    with open("data/meta.jsonl", "w", encoding="utf-8") as f:
        for d in corpus: f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("Ingest fullført.")
