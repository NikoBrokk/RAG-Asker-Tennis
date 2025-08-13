from __future__ import annotations
import os
import re
from typing import List, Dict, Tuple, Set

from src.utils import env_flag
from src.retrieve import search

# -------------------------------------------------------------------
# Configuration
#
# Always attempt to use the OpenAI API if a key is configured. The default
# environment flag USE_OPENAI still controls the behaviour, but the default
# value is now True in `.env.example`. The chat model can be overridden via
# the CHAT_MODEL environment variable. When OpenAI is unavailable the app
# falls back to simple extractive answering.

USE_OPENAI = env_flag("USE_OPENAI", True)
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

_openai_client = None
if USE_OPENAI:
    try:
        from openai import OpenAI
        _openai_client = OpenAI()
    except Exception:
        _openai_client = None

# System prompt instructing the assistant to be friendly and concise.
SYSTEM_PROMPT = """
Du er en vennlig og hjelpsom assistent for Asker Tennis.
Du svarer på spørsmål basert på innhold i klubbens dokumenter. Svar i
1–3 korte setninger på norsk, med en personlig og imøtekommende tone.
Hvis dokumentene ikke dekker spørsmålet, svarer du ærlig at du ikke vet.
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
    # return the first sentence of the top hit as a fallback
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
    """Generate a natural language answer using the OpenAI Chat API.

    We pass up to the top five context passages to provide the model with
    sufficient background. The prompt encourages the model to synthesise
    information instead of copying text verbatim and to respond concisely.
    """
    if _openai_client is None:
        return _extractive_answer(hits)
    # build context from the top hits
    context_chunks = hits[:5]
    context = "\n\n".join(f"Utdrag {i+1}:\n{h.get('text','')}" for i, h in enumerate(context_chunks))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Spørsmål: {q}\n\n"
                f"Kontekst (bruk kun hvis relevant):\n{context}\n\n"
                "Instruks: Svar i opptil tre korte setninger. Ikke bare gjenta utdragene, men skriv med egne ord."
            ),
        },
    ]
    try:
        resp = _openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=120,
        )
        answer_text = resp.choices[0].message.content.strip()
    except Exception:
        return _extractive_answer(hits)
    return answer_text

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
