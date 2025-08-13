from __future__ import annotations
import os
import re
from typing import List, Dict, Tuple, Set
from src.utils import env_flag
from src.retrieve import search

USE_OPENAI = env_flag("USE_OPENAI", True)
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

_openai_client = None
if USE_OPENAI:
    try:
        from openai import OpenAI
        _openai_client = OpenAI()
    except Exception:
        _openai_client = None

SYSTEM_PROMPT = """
Du er en dokumentassistent for Asker Tennis.
Svar KUN på det som blir spurt om.
Bruk maks én kort setning. Returner KUN svaret, uten forklaringer.
Hvis dokumentene ikke dekker spørsmålet, svar: "Jeg vet ikke".
Svar på norsk bokmål.
"""

# --- Domene-synonymer og hinting ---
SYN = {
    "booking": ["booking", "booke", "banebooking", "banereservasjon", "reserver", "bane", "matchi"],
    "pris": ["pris", "avgift", "timepris", "kostnad", "medlemspris", "drop-in", "billig", "rimelig", "rabatt", "off-peak", "lavsesong"],
    "tid": ["tid", "tidspunkt", "hverdag", "helg", "dagtid", "kveld"],
}

DOC_TYPE_HINTS = {
    "booking": SYN["booking"],
    "pris": SYN["pris"],
}

def _expand_query(q: str) -> Tuple[str, Set[str], List[str]]:
    ql = q.lower()
    extra: List[str] = []
    preferred: Set[str] = set()
    for dt, triggers in DOC_TYPE_HINTS.items():
        if any(t in ql for t in triggers):
            preferred.add(dt)
    # legg til synonymer hvis trigget
    if any(t in ql for t in SYN["booking"]):
        extra += SYN["booking"]
    if any(t in ql for t in SYN["pris"]):
        extra += SYN["pris"]
    if any(t in ql for t in SYN["tid"]):
        extra += SYN["tid"]
    expanded = q if not extra else q + " " + " ".join(sorted(set(extra)))
    key_terms = sorted(set(extra))
    return expanded, preferred, key_terms

def _first_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    m = re.search(r"(.+?[.!?])\s", text + " ")
    s = (m.group(1) if m else text)[:280]
    return s

def _extractive_answer(hits: List[Dict]) -> str:
    if not hits:
        return "Jeg vet ikke"
    return _first_sentence(hits[0].get("text", "")) or "Jeg vet ikke"

def _score_hit(h: Dict, key_terms: List[str], preferred_types: Set[str]) -> float:
    base = float(h.get("score", 0.0))
    bonus = 0.0
    if h.get("doc_type") in preferred_types:
        bonus += 0.15
    txt = (h.get("text") or "").lower()
    if key_terms:
        overlap = sum(1 for t in key_terms if t in txt)
        bonus += min(0.10, 0.02 * overlap)
    return base + bonus

def _rerank_and_filter(q: str, hits: List[Dict], preferred_types: Set[str], key_terms: List[str], k: int, min_score: float = 0.18) -> List[Dict]:
    # Re-rank med enkel bonus, så filtrer på min score (etter bonus)
    scored = [(h, _score_hit(h, key_terms, preferred_types)) for h in hits]
    scored.sort(key=lambda x: x[1], reverse=True)
    good = [h for h, s in scored if s >= min_score]
    return (good or [])[:k]

def _llm_answer(q: str, hits: List[Dict]) -> str:
    if _openai_client is None:
        return _extractive_answer(hits)
    context = "\n\n".join(f"Utdrag {i+1}:\n{h.get('text','')}" for i, h in enumerate(hits[:3]))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"Spørsmål: {q}\n\nKontekst (bruk kun hvis relevant):\n{context}\n\nInstruks: Svar i MAKS én kort setning. Returner bare svaret."},
    ]
    resp = _openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=40,
    )
    return resp.choices[0].message.content.strip()

def answer(q: str, k: int = 6) -> Tuple[str, List[Dict]]:
    # 1) Query-utvidelse for bedre recall i TF-IDF
    q_expanded, preferred_types, key_terms = _expand_query(q)

    # 2) Hent flere kandidater enn vi viser, så re-ranker vi
    raw_hits = search(q_expanded, max(k * 2, 6))

    # 3) Rerank + min-score gate
    hits = _rerank_and_filter(q, raw_hits, preferred_types, key_terms, k)

    # 4) Hvis ingenting relevant -> tydelig "vet ikke"
    if not hits:
        return "Jeg vet ikke", raw_hits[:k]

    # 5) LLM (kort) eller extractive
    if USE_OPENAI and _openai_client is not None:
        ans = _llm_answer(q, hits)
    else:
        ans = _extractive_answer(hits)

    # Tomt eller nonsens? Fallback til "vet ikke"
    if not ans or len(ans.split()) < 2:
        ans = "Jeg vet ikke"
    return ans, hits
