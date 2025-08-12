from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Iterable, Dict, List

def read_markdown_files(root_dir: str | Path) -> List[Dict]:
    """
    Leser .txt og .md fra rotmappen (rekursivt) og returnerer en liste records:
    {title, source, text, version_date}
    - 'Tittel: ...' på første linje overstyrer filnavnet som tittel (hvis finnes).
    """
    root = Path(root_dir)
    records: List[Dict] = []
    for fp in root.rglob("*"):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in {".txt", ".md"}:
            continue
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            # Som nød: binær/annen encoding – hopp over
            continue

        title = fp.stem
        # Se etter linje som starter med "Tittel:"
        for line in text.splitlines():
            if line.lower().startswith("tittel:"):
                title = line.split(":", 1)[1].strip() or title
                break

        version_date = datetime.fromtimestamp(fp.stat().st_mtime).strftime("%Y-%m-%d")
        records.append(
            {
                "title": title,
                "source": str(fp.resolve()),
                "text": text.strip(),
                "version_date": version_date,
            }
        )
    return records

def simple_chunks(text: str, chunk_size: int = 800, overlap: int = 100) -> Iterable[str]:
    """
    Enkelt overlappende chunking.
    """
    if not text:
        return []
    n = len(text)
    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        yield text[start:end]
        if end == n:
            break
        start = max(0, end - overlap)
