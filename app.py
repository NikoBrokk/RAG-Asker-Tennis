import os
from pathlib import Path
import streamlit as st

from src.answer import answer

def _env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None and name in st.secrets:
        v = str(st.secrets[name])
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}

# Konfig (secrets > env)
USE_OPENAI = _env_flag("USE_OPENAI", False)
CHAT_MODEL = os.getenv("CHAT_MODEL", st.secrets.get("CHAT_MODEL", "gpt-4o-mini" if USE_OPENAI else "tf-idf"))
DATA_DIR = Path(os.getenv("DATA_DIR", st.secrets.get("DATA_DIR", "data")))
KB_DIR = os.getenv("KB_DIR", st.secrets.get("KB_DIR", "kb"))

from src.ingest import build_index, OPENAI_API_KEY

# Bygg indeksen hvis den mangler
def ensure_index():
    vec = DATA_DIR / "vectors.npy"
    meta = DATA_DIR / "meta.jsonl"

    # Sjekk for OpenAI API-nøkkel hvis USE_OPENAI er aktivert
    if USE_OPENAI and not OPENAI_API_KEY:
        st.error("Kan ikke bygge indeks – OPENAI_API_KEY mangler. "
                 "Sett den i .env-filen eller i Streamlit Secrets.")
        st.stop()

    if not vec.exists() or not meta.exists():
        st.info("Indeks mangler – bygger nå …")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        build_index(KB_DIR)

st.set_page_config(page_title="RAG Demo – Asker Tennis", page_icon="🔎", layout="centered")
ensure_index()

st.title("🔎 RAG Demo (GitHub)")
mode_label = "**OpenAI**" if USE_OPENAI else "**TF-IDF**"
st.caption(f"Status: indeks `ok` • Modus: {mode_label} (modell: {CHAT_MODEL})")

q = st.text_input("Skriv spørsmålet ditt:", placeholder="F.eks. Hvordan resetter jeg passordet?")
k = st.slider("Antall kilder", 2, 12, 6)

if st.button("Svar") and q.strip():
    with st.spinner("Henter…"):
        text, hits = answer(q, k=k)
    st.markdown("### Svar")
    st.write(text)
    st.markdown("### Kilder")
    if not hits:
        st.write("Ingen kilder.")
    else:
        for h in hits:
            src = h.get("source", "?")
            hid = h.get("id", "?")
            sc = float(h.get("score", 0.0))
            st.markdown(f"- **{src}** — `{hid}` (score {sc:.3f})")
