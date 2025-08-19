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

# Oppdaterte synonymer og triggere for utvidet søk
SYN: Dict[str, List[str]] = {
    "booking": ["booking", "booke", "banebooking", "banereservasjon", "reserver", "bane", "baner", "matchi"],
    "pris": ["pris", "avgift", "timepris", "kostnad", "kontingent", "medlemskontingent",
             "drop-in", "billig", "rimelig", "rabatt", "off-peak", "lavsesong"],
    "kontakt": ["kontakt", "telefon", "tlf", "mail", "e-post", "email", "adresse"],
    "medlemskap": ["medlemskap", "medlem", "medlemsfordel", "innmelding", "bli medlem",
                   "junior", "barn", "voksen", "familie"],
    "parkering": ["parkering", "p-norge", "easypark", "parkere", "bil", "takstgruppe"],
    "utstyr": ["utstyr", "racket", "racketer", "ball", "baller", "leie utstyr", "låne", "strengeservice"],
    "klubbstigen": ["klubbstigen", "stige", "rankingsystem", "appen", "liga"],
    "leir": ["leir", "sommerleir", "høstferieleir", "vinterferie", "camp", "campen"],
    "info": ["klubb", "anlegg", "hall", "uteanlegg", "medlemmer", "baner", "historie",
             "stiftet", "leder", "sportslig", "hovedtrener", "daglig"]
}

DOC_HINTS: Dict[str, List[str]] = {
    "booking": SYN["booking"],
    "pris": SYN["pris"],
    "kontakt": SYN["kontakt"],
    "medlemskap": SYN["medlemskap"],
    "parkering": SYN["parkering"],
    "utstyr": SYN["utstyr"],
    "klubbstigen": SYN["klubbstigen"],
    "leir": SYN["leir"],
    "info": SYN["info"]
}

def _expand_query(q: str) -> Tuple[str, Set[str], List[str]]:
    # Fjern irrelevante ord som klubbnavn
    ql = q.lower()
    ql = re.sub(r"\basker tennisklubb\b|\basker tennis\b", " ", ql)
    ql = ql.strip()

    extra: List[str] = []
    preferred: Set[str] = set()

    # Legg til doc hints basert på triggere
    for dt, triggers in DOC_HINTS.items():
        if any(t in ql for t in triggers):
            preferred.add(dt)
    # Legg til utvidede søkeord (synonymer) basert på synonymlistene
    for key, words in SYN.items():
        if any(t in ql for t in words):
            extra += words

    # Bygg utvidet spørring
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
    # Bonus for hvert nøkkelord som faktisk forekommer i dokumentteksten
    bonus += min(0.10, 0.02 * sum(1 for t in keys if t in txt))
    return base + bonus

def _rerank(hits: List[Dict], preferred: Set[str], keys: List[str], k: int, min_score: float = 0.15) -> List[Dict]:
    scored = [(h, _score(h, keys, preferred)) for h in hits]
    scored.sort(key=lambda x: x[1], reverse=True)
    good = [h for h, s in scored if s >= min_score]
    return good[:k] if good else []

def _llm(q: str, hits: List[Dict]) -> str:
    if _openai is None:
        return _extractive(hits)
    # Bygg kontekst av topp 5 utdrag
    ctx = "\n\n".join(f"Utdrag {i+1}:\n{h.get('text','')}" for i, h in enumerate(hits[:5]))
    # Inkluder tidligere meldinger for kontekst (fra session_state via streamlit)
    history_msgs = []
    try:
        import streamlit as st
        for prev_q, prev_a in st.session_state.get("history", [])[-3:]:  # siste 3 meldinger
            # Fjern prefix "Spørsmål:" fra lagret historikkinnlegg
            uq = prev_q.replace("**Spørsmål:**", "Spørsmål:").strip()
            history_msgs.append({"role": "user", "content": uq})
            history_msgs.append({"role": "assistant", "content": prev_a})
    except Exception:
        history_msgs = []
    # Bygg meldingsliste til OpenAI chat-komplettering
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history_msgs
    messages.append({
        "role": "user",
        "content": f"Spørsmål: {q}\n\nKontekst:\n{ctx}\n\nInstruks: Svar med egne ord i 1–3 setninger."
    })
    try:
        r = _openai.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.2, max_tokens=120)
        return (r.choices[0].message.content or "").strip()
    except Exception:
        return _extractive(hits)

def answer(q: str, k: int = 6) -> Tuple[str, List[Dict]]:
    qx, preferred, keys = _expand_query(q)
    raw_hits = search(qx, max(k * 2, 6))
    hits = _rerank(raw_hits, preferred, keys, k)
    if not hits:
        return "Jeg vet ikke", raw_hits[:k]
    out = _llm(q, hits) if USE_OPENAI and _openai is not None else _extractive(hits)
    if not out or len(out.split()) < 2:
        out = "Jeg vet ikke"
    return out, hits
