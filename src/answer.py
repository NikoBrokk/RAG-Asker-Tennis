from __future__ import annotations
import os, re
from typing import Dict, List, Tuple, Set

from src.utils import env_flag
from src.retrieve import search

USE_OPENAI = env_flag("USE_OPENAI", False)
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

_openai = None
if USE_OPENAI:
    try:
        from openai import OpenAI
        _openai = OpenAI()
    except Exception:
        _openai = None

SYSTEM_PROMPT = (
    "Du er en vennlig og hjelpsom assistent for Asker Tennis.\n"
    "Svar kort (1–3 setninger) på norsk bokmål, med egne ord. "
    "Hvis kildene ikke dekker spørsmålet, si 'Jeg vet ikke'."
)

SYN = {
    "booking": ["booking", "booke", "banebooking", "banereservasjon", "reserver", "bane", "matchi"],
    "pris": ["pris", "avgift", "timepris", "kostnad", "medlemspris", "drop-in", "billig", "rimelig", "rabatt", "off-peak", "lavsesong"],
    "tid": ["tid", "tidspunkt", "hverdag", "helg", "dagtid", "kveld"],
}
DOC_HINTS = {"booking": SYN["booking"], "pris": SYN["pris"]}

def _expand_query(q: str) -> Tuple[str, Set[str], List[str]]:
    ql = q.lower()
    extra: List[str] = []
    preferred: Set[str] = set()
    for dt, triggers in DOC_HINTS.items():
        if any(t in ql for t in triggers):
            preferred.add(dt)
    for key in ("booking", "pris", "tid"):
        if any(t in ql for t in SYN[key]):
            extra += SYN[key]
    expanded = q if not extra else q + " " + " ".join(sorted(set(extra)))
    return expanded, preferred, sorted(set(extra))

def _first_sentence(txt: str) -> str:
    txt = re.sub(r"\s+", " ", (txt or "").strip())
    m = re.search(r"(.+?[.!?])\s", txt + " ")
    return (m.group(1) if m else txt)[:280]

def _extractive(hits: List[Dict]) -> str:
    if not hits:
        return "Jeg vet ikke"
    return _first_sentence(hits[0].get("text", "")) or "Jeg vet ikke"

def _score(h: Dict, keys: List[str], preferred: Set[str]) -> float:
    base = float(h.get("score", 0.0))
    bonus = 0.15 if h.get("doc_type") in preferred else 0.0
    txt = (h.get("text") or "").lower()
    bonus += min(0.10, 0.02 * sum(1 for t in keys if t in txt))
    return base + bonus

def _rerank(hits: List[Dict], preferred: Set[str], keys: List[str], k: int, min_score: float = 0.18) -> List[Dict]:
    scored = [(h, _score(h, keys, preferred)) for h in hits]
    scored.sort(key=lambda x: x[1], reverse=True)
    good = [h for h, s in scored if s >= min_score]
    return good[:k] if good else []

def _llm(q: str, hits: List[Dict]) -> str:
    if _openai is None:
        return _extractive(hits)
    ctx = "\n\n".join(f"Utdrag {i+1}:\n{h.get('text','')}" for i, h in enumerate(hits[:5]))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Spørsmål: {q}\n\nKontekst:\n{ctx}\n\nInstruks: Svar med egne ord i 1–3 setninger."},
    ]
    try:
        r = _openai.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.2, max_tokens=120)
        return (r.choices[0].message.content or "").strip()
    except Exception:
        return _extractive(hits)

def answer(q: str, k: int = 6) -> Tuple[str, List[Dict]]:
    qx, preferred, keys = _expand_query(q)
    raw = search(qx, max(k * 2, 6))
    hits = _rerank(raw, preferred, keys, k)
    if not hits:
        return "Jeg vet ikke", raw[:k]
    out = _llm(q, hits) if USE_OPENAI and _openai is not None else _extractive(hits)
    if not out or len(out.split()) < 2:
        out = "Jeg vet ikke"
    return out, hits
