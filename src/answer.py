from __future__ import annotations
import os
import re
from typing import List, Dict, Tuple
from src.utils import env_flag
from src.retrieve import search

# Standard: bruk OpenAI hvis mulig
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
Bruk maks én kort setning. Returner KUN selve svaret, uten forklaringer eller innledninger.
Hvis dokumentene ikke dekker spørsmålet, svar: "Jeg vet ikke".
Svar på norsk bokmål.
"""

def _first_sentence(text: str) -> str:
    # Kompakt første setning (punktum, spørsmålstegn eller utrop)
    text = re.sub(r"\s+", " ", text.strip())
    m = re.search(r"(.+?[.!?])\s", text + " ")
    s = (m.group(1) if m else text)[:280]
    return s

def _extractive_answer(hits: List[Dict]) -> str:
    if not hits:
        return "Jeg vet ikke"
    # Prøv første hele setning i første treff
    return _first_sentence(hits[0].get("text", "")) or "Jeg vet ikke"

def _llm_answer(q: str, hits: List[Dict]) -> str:
    if _openai_client is None:
        return _extractive_answer(hits)

    # Bruk kun de 3 beste utdragene for å holde konteksten stram
    context = "\n\n".join(f"Utdrag {i+1}:\n{h.get('text','')}" for i, h in enumerate(hits[:3]))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Spørsmål: {q}\n\n"
                f"Kontekst (kun bruk hvis relevant):\n{context}\n\n"
                "Instruks: Svar i MAKS én kort setning. Returner bare selve svaret."
            ),
        },
    ]
    resp = _openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=40,
    )
    return resp.choices[0].message.content.strip()

def answer(q: str, k: int = 6) -> Tuple[str, List[Dict]]:
    hits = search(q, k)
    if USE_OPENAI and _openai_client is not None:
        ans = _llm_answer(q, hits)
    else:
        ans = _extractive_answer(hits)
    return ans, hits
