import os, json, numpy as np

def test_artifacts_exist():
    assert os.path.exists("data/meta.jsonl")
    assert os.path.exists("data/vectors.npy")
    assert os.path.exists("data/index.faiss")
    X = np.load("data/vectors.npy")
    assert X.ndim == 2 and X.shape[0] > 0
