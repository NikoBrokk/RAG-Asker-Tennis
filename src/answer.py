import os
from dotenv import load_dotenv
from openai import OpenAI
from src.retrieve import search

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL     = os.getenv("CHAT_MODEL", "gpt-4o-mini")
TEMPERATURE    = float(os.getenv("TEMPERATURE", "0.1"))

client = OpenAI(api_key=OPENAI_API_KEY)

PROMPT_HEADER = (
    "Du er en kunnskapsrik, høflig assistent for Asker Tennis.\n"
    "Svar alltid på bokmål, basert på utdragene fra dokumentene. "
    "Hvis informasjonen ikke finnes i utdragene, si at du ikke vet."
)

def build_prompt(q: str, hits):
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(f"[{h['id']}]\n{h['text']}")
    context = "\n\n".join(parts)
    return f"{PROMPT_HEADER}\n\nSpørsmål: {q}\n\nKontekst:\n{context}\n\nSvar:"

def answer(q: str, k: int = 6):
    hits = search(q, k)
    prompt = build_prompt(q, hits)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
    )
    answer_text = resp.choices[0].message.content.strip()
    return answer_text, hits
