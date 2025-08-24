import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import retrieve  # type: ignore

def test_load_corpus_handles_none_source(monkeypatch, tmp_path):
    sample = {"text": "hello", "metadata": {"source": None}}
    f = tmp_path / "sample.jsonl"
    f.write_text(json.dumps(sample), encoding="utf-8")
    monkeypatch.setattr(retrieve, "KB_DIRS", [tmp_path])
    docs = retrieve._load_corpus()
    assert docs and docs[0]["source"].endswith("sample.jsonl")
