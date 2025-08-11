import re
from pathlib import Path

def read_markdown_files(root="kb"):
    for p in Path(root).rglob("*.md"):
        yield str(p), p.read_text(encoding="utf-8")

def simple_chunks(text, max_tokens=700, overlap_tokens=120):
    # grov chunking på avsnitt; hold overskrift med innhold
    paras = re.split(r"\n\s*\n", text.strip())
    chunks, buf, size = [], [], 0
    def tok_count(s): return max(1, len(s.split()))
    for para in paras:
        t = tok_count(para)
        if size + t > max_tokens and buf:
            chunks.append("\n\n".join(buf))
            # overlap: behold siste avsnitt
            overlap = buf[-1:] if overlap_tokens > 0 else []
            buf, size = overlap[:], tok_count("\n\n".join(overlap))
        buf.append(para); size += t
    if buf: chunks.append("\n\n".join(buf))
    return chunks
