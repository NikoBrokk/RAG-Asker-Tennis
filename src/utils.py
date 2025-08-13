from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List
from datetime import datetime

def _read_text_file(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return p.read_text(encoding="latin-1", errors="ignore")

def read_markdown_files(kb_dir: str) -> List[Dict]:
    root = Path(kb_dir)
    if not root.exists():
        return []
    out: List[Dict] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".md", ".txt"}:
            continue
        text = _read_text_file(p)
        title = p.stem.replace("_", " ").strip() or "Uten tittel"
        version_date = datetime.fromtimestamp(p.stat().st_mtime).date().isoformat()
        out.append({"title": title, "source": str(p), "text": text, "version_date": version_date})
    return out

def simple_chunks(text: str, size: int = 800, overlap: int = 120) -> List[str]:
    if size <= 0:
        return [text]
    toks = text.split()
    if not toks:
        return []
    out: List[str] = []
    i = 0
    while i < len(toks):
        chunk = " ".join(toks[i:i+size]).strip()
        if chunk:
            out.append(chunk)
        if i + size >= len(toks):
            break
        i += max(1, size - overlap)
    return out

def env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}
