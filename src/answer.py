from __future__ import annotations
import os
import re
from typing import Dict, List, Tuple, Set

from src.utils import env_flag
from src.retrieve import search

# --- Konfig ---
USE_OPENAI = env_flag("USE_OPENAI", False)
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Valgfri OpenAI-klient
_openai = None
if USE_OPENAI:
    try:
        from openai import OpenAI
        _openai = OpenAI()
    except Exception:
        _openai = None

# Prompt til ChatGPT
SYSTEM_PROMPT = (
    "Du er en vennlig og hjelpsom assistent for Asker Tennis.\n"
    "Svar på spørsmål basert på informasjonen i vedlagte utdrag. "
    "Bruk 1–3 setninger på norsk bokmål med personlig og imøtekommende tone. "
    "Hvis dokumentene ikke dekker spørsmålet, svar ærlig at du ikke vet."
)

# Synonymliste for enkel query-utvidelse
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
    """Utvider brukerens spørsmål med synonymer og markerer foretrukne dokumenttyper."""
    ql = q.lower()
    extra: List[str] = []
    preferred: Set[str] = set()
    for dt, triggers in DOC_TYPE_HINTS.items():
        if any(t in ql for t in triggers):
            preferred.add(dt)
    for key in ("booking", "pris", "tid"):
        if any(t in ql for t in SYN[key]):
            extra += SYN[key]
    expanded = q if not extra else q + " " + " ".join(sorted(set(extra)))
    return expanded, preferred, sorted(set(extra))

def _first_sentence(text: str) -> str:
    """Hent ut første setning som fallback."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    m = re.search(r"(.+?[.!?])\s", text + " ")
    s = (m.group(1) if m else text)[:280]
    return s

def _extractive_answer(hits: List[Dict]) -> str:
    """Returner første setning fra beste treff."""
    if not hits:
        return "Jeg vet ikke"
    return _first_sentence(hits[0].get("text", "")) or "Jeg vet ikke"

def _score_hit(h: Dict, key_terms: List[str], preferred_types: Set[str]) -> float:
    base = float(h.get("score", 0.0))
    bonus = 0.0
    if h.get("doc_type") in preferred_types:
        bonus += 0.15
    txt = (h.get("text") or "").lower()
    overlap = sum(1 for t in key_terms if t in txt)
    bonus += min(0.10, 0.02 * overlap)
    return base + bonus

def _rerank_and_filter(hits: List[Dict], preferred_types: Set[str], key_terms: List[str], k: int, min_score: float = 0.18) -> List[Dict]:
    scored = [(h, _score_hit(h, key_terms, preferred_types)) for h in hits]
    scored.sort(key=lambda x: x[1], reverse=True)
    filtered = [h for h, s in scored if s >= min_score]
    return filtered[:k] if filtered else []

def _llm_answer(q: str, hits: List[Dict]) -> str:
    if _openai is None:
        return _extractive_answer(hits)
    # Lag kontekst fra inntil fem utdrag
    context_chunks = hits[:5]
    context = "\n\n".join(f"Utdrag {i+1}:\n{h.get('text','')}" for i, h in enumerate(context_chunks))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Spørsmål: {q}\n\n"
                f"Kontekst (bruk kun hvis relevant):\n{context}\n\n"
                "Instruks: Svar i opptil tre korte setninger. Ikke kopier utdragene ordrett, men skriv med egne ord."
            ),
        },
    ]
    try:
        resp = _openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=120,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return _extractive_answer(hits)

def answer(q: str, k: int = 6) -> Tuple[str, List[Dict]]:
    """
    Hovedfunksjon som tar et spørsmål og returnerer (svartekst, treffliste).
    Den utvider spørsmålet, kjører søk, re-ranker treff og genererer et svar.
    """
    q_expanded, preferred_types, key_terms = _expand_query(q)
    raw_hits = search(q_expanded, max(k * 2, 6))
    hits = _rerank_and_filter(raw_hits, preferred_types, key_terms, k)
    if not hits:
        return "Jeg vet ikke", raw_hits[:k]

    if USE_OPENAI and _openai is not None:
        ans = _llm_answer(q, hits)
    else:
        ans = _extractive_answer(hits)

    if not ans or len(ans.split()) < 2:
        ans = "Jeg vet ikke"
    return ans, hits
