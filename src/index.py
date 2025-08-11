import faiss, numpy as np, json
from pathlib import Path

DIM = 1536  # for text-embedding-3-*
INDEX_PATH = Path("data/index.faiss")

def load_vectors():
    X = np.load("data/vectors.npy").astype("float32")
    faiss.normalize_L2(X)
    return X

def build_index():
    X = load_vectors()
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    faiss.write_index(index, str(INDEX_PATH))

def load_meta():
    return [json.loads(l) for l in open("data/meta.jsonl", encoding="utf-8")]

if __name__ == "__main__":
    build_index()
    print("Indeks skrevet til", INDEX_PATH)
