import re
from pathlib import Path
from pypdf import PdfReader

def extract_pdf_text(path: str) -> str:
    text_parts = []
    reader = PdfReader(path)
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        # enkel rens
        t = t.replace("\u00AD", "")  # myk bindestrek
        text_parts.append(t.strip())
    return "\n\n".join(p for p in text_parts if p)

def read_kb_files(root="kb"):
    """Yield (path, text) for .md/.txt/.pdf."""
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in {".md", ".txt"}:
            yield str(p), p.read_text(encoding="utf-8", errors="ignore")
        elif p.suffix.lower() == ".pdf":
            yield str(p), extract_pdf_text(str(p))

def simple_chunks(text, max_tokens=700, overlap_tokens=120):
    paras = re.split(r"\n\s*\n", text.strip())
    chunks, buf, size = [], [], 0
    def tok_count(s): return max(1, len(s.split()))
    for para in paras:
        t = tok_count(para)
        if size + t > max_tokens and buf:
            chunks.append("\n\n".join(buf))
            overlap = buf[-1:] if overlap_tokens > 0 else []
            buf, size = overlap[:], tok_count("\n\n".join(overlap))
        buf.append(para); size += t
    if buf: chunks.append("\n\n".join(buf))
    return chunks
