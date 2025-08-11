from openai import OpenAI
from dotenv import load_dotenv
import os
from src.retrieve import search

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

SYSTEM = "Du er en forsiktig assistent. Svar KUN basert på konteksten. Hvis utilstrekkelig, si at du ikke vet."

def build_prompt(q, hits):
    ctx = "\n\n".join(f"[{h['id']}]\n{h['text']}" for h in hits)
    return f"""Bruk kun konteksten til å svare. Siter kilder som [path#chunk].

Spørsmål: {q}

Kontekst:
{ctx}
"""

def answer(q, k=6):
    hits = search(q, k=k)
    prompt = build_prompt(q, hits)
    r = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"system","content":SYSTEM},
                  {"role":"user","content":prompt}],
        temperature=0
    )
    return r.choices[0].message.content, hits

if __name__ == "__main__":
    import sys, json
    q = " ".join(sys.argv[1:]) or "Hva er gratisplanens grenser?"
    text, hits = answer(q)
    print(text, "\n\nKilder:")
    for h in hits: print(h["id"], f"(score={h['score']:.3f})")
