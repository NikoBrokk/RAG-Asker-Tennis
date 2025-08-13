import os
from pathlib import Path
import streamlit as st

from src.answer import answer
from src.retrieve import search  # optional import to trigger index build

# --- Konfig / modus ---
def _env_flag(name: str, default: bool = False) -> bool:
    """Les boolsk miljø/secrets variabel (true/1/yes/on)."""
    v = (
        st.secrets.get(name)
        if isinstance(st.secrets, dict) and name in st.secrets
        else os.getenv(name)
    )
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}

USE_OPENAI = _env_flag("USE_OPENAI", False)
CHAT_MODEL = os.getenv("CHAT_MODEL", st.secrets.get("CHAT_MODEL", "gpt-4o-mini" if USE_OPENAI else "tf-idf"))
DATA_DIR = Path(os.getenv("DATA_DIR", st.secrets.get("DATA_DIR", "data")))
KB_DIR = os.getenv("KB_DIR", st.secrets.get("KB_DIR", "kb"))

# --- Sørg for at indeks finnes ---
def _ensure_index() -> str:
    vectors = DATA_DIR / "vectors.npy"
    meta = DATA_DIR / "meta.jsonl"
    if vectors.exists() and meta.exists():
        return "ok"
    # Bygg indeksen hvis den mangler (første oppstart på Streamlit Cloud)
    try:
        st.info("Indeks mangler – bygger nå …")
        from src.ingest import build_index  # lazy import for å unngå sirkulær import
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        build_index(KB_DIR)
        return "built"
    except Exception as e:
        st.error(f"Kunne ikke bygge indeks automatisk: {e}")
        return "error"

status = _ensure_index()

# --- UI ---
st.set_page_config(page_title="RAG Demo – Asker Tennis", page_icon="🔎", layout="centered")
st.title("🔎 RAG Demo (GitHub)")

# Statuslinje
mode_label = "**OpenAI**" if USE_OPENAI else "**TF‑IDF**"
st.caption(f"Status: indeks `{status}` • Modus: {mode_label}" + (f" (`{CHAT_MODEL}`)" if USE_OPENAI else ""))

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
            # Robust mot manglende felt
            src = h.get("source", "?")
            hid = h.get("id", "?")
            sc = float(h.get("score", 0.0))
            st.markdown(f"- **{src}** — `{hid}` (score {sc:.3f})")
