from __future__ import annotations
import os
from typing import List, Dict
from src.utils import env_flag
from src.retrieve import search

USE_OPENAI = env_flag("USE_OPENAI", False)
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

_openai_client = None
if USE_OPENAI:
    try:
        from openai import OpenAI
        _openai_client = OpenAI()
    except Exception:
        _openai_client = None

SYSTEM_PROMPT = (
    "Du er en dokumentassistent for Asker Tennis. Svar kort, presist og høflig. "
    "Bruk KUN informasjon fra utdragene. Hvis dokumentene ikke dekker spørsmålet, si 'Jeg vet ikke'. "
    "Alltid på norsk bokmål."
)

def _llm_answer(q: str, hits: List[Dict]) -> str:
    if _openai_client is None:
        return _extractive_answer(hits)
    context = "\n\n".join(f"Utdrag {i+1}:\n{h['text']}" for i, h in enumerate(hits))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Spørsmål: {q}\n\nKontekst:\n{context}\n\nSvar kort (6–8 setninger)."},
    ]
    resp = _openai_client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.1)
    return resp.choices[0].message.content.strip()

def _extractive_answer(hits: List[Dict]) -> str:
    if not hits:
        return "Jeg vet ikke."
    # Bruk første (beste) utdrag
    return hits[0]["text"].strip()

def answer(q: str, k: int = 6):
    hits = search(q, k)
    if USE_OPENAI:
        ans = _llm_answer(q, hits[:k])
    else:
        ans = _extractive_answer(hits[:k])
    return ans, hits
