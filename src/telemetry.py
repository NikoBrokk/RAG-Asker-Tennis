
from __future__ import annotations
import csv, json, re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

_UNKNOWN_PATTERNS = [
    r"\bjeg vet ikke\b",
    r"\bikke i mitt datagrunnlag\b",
    r"\binget treff\b",
]

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_TLF_RE = re.compile(r"\b(?:\+?\d[\d\s\-()]{6,}\d)\b")

def _mask_pii(text: str) -> str:
    if not text:
        return text
    t = _EMAIL_RE.sub("***@***", text)
    # Masker telefonnumre: behold siste 2 siffer for feilsøking
    def _m(m: re.Match) -> str:
        raw = re.sub(r"\D", "", m.group(0))
        if len(raw) < 6:
            return m.group(0)
        return "*" * max(0, len(raw) - 2) + raw[-2:]
    t = _TLF_RE.sub(_m, t)
    return t

def is_unknown(answer_text: str) -> bool:
    if not answer_text:
        return True
    low = answer_text.strip().lower()
    return any(re.search(p, low) for p in _UNKNOWN_PATTERNS)

def _ensure_dirs() -> Path:
    outdir = Path("eval/runs")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def _append_csv(path: Path, row: Dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def log_unknown(
    *,
    question: str,
    expanded_query: str | None,
    preferred_tags: List[str] | None,
    hits: List[Dict[str, Any]] | None,
    answer_text: str,
    meta: Dict[str, Any] | None = None,
) -> None:
    """
    Logger 'jeg vet ikke'-svar til CSV og JSONL med PII-maskering.
    hits forventes som liste av dict med minst keys: source, id, score (best effort).
    """
    outdir = _ensure_dirs()
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # PII-masker fritekstfelt
    safe_q = _mask_pii(question or "")
    safe_ans = _mask_pii(answer_text or "")
    safe_expanded = _mask_pii(expanded_query or "")

    srcs, ids, scores = [], [], []
    for h in (hits or [])[:10]:
        srcs.append(str(h.get("source", "")))
        ids.append(str(h.get("id", "")))
        sc = h.get("score")
        scores.append(f"{sc:.3f}" if isinstance(sc, (int, float)) else "")

    csv_row = {
        "timestamp": ts,
        "question": safe_q,
        "answer_excerpt": safe_ans[:160],
        "expanded_query": safe_expanded,
        "preferred_tags": ",".join(preferred_tags or []),
        "top_sources": ";".join(srcs),
        "top_ids": ";".join(ids),
        "top_scores": ";".join(scores),
    }
    json_obj = {
        "timestamp": ts,
        "question": safe_q,
        "answer": safe_ans,
        "expanded_query": safe_expanded,
        "preferred_tags": preferred_tags or [],
        "hits": hits or [],
        "meta": meta or {},
    }

    _append_csv(outdir / "unknown.csv", csv_row)
    _append_jsonl(outdir / "unknown.jsonl", json_obj)
