from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List
from datetime import datetime

def _read_text_file(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return p.read_text(encoding="latin-1", errors="ignore")

def read_markdown_files(kb_dir: str) -> List[Dict]:
    """Les .md og .txt fra kb_dir og returner liste med {title, source, text, version_date}."""
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
        # enkel versjonsdato: filens mtime
        version_date = datetime.fromtimestamp(p.stat().st_mtime).date().isoformat()
        out.append({
            "title": title,
            "source": str(p),
            "text": text,
            "version_date": version_date,
        })
    return out

def simple_chunks(text: str, size: int = 800, overlap: int = 120) -> List[str]:
    """Glidende vindu over ren tekst – robust for norsk innhold."""
    if size <= 0: return [text]
    tokens = text.split()
    if not tokens: return []
    chunks: List[str] = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i+size]
        chunks.append(" ".join(chunk_tokens))
        if i + size >= len(tokens):
            break
        i += max(1, size - overlap)
    return chunks

def env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name, str(default))
    return str(v).strip().lower() in {"1", "true", "yes", "on"}
