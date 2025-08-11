import faiss, numpy as np, json
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

index = faiss.read_index("data/index.faiss")
meta = [json.loads(l) for l in open("data/meta.jsonl", encoding="utf-8")]

def embed(q):
    v = client.embeddings.create(model=EMBED_MODEL, input=[q]).data[0].embedding
    v = np.array([v], dtype="float32"); faiss.normalize_L2(v)
    return v

def search(q, k=6):
    v = embed(q)
    D, I = index.search(v, k)
    return [meta[i] | {"score": float(D[0][j])} for j, i in enumerate(I[0])]
