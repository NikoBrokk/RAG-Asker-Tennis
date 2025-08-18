
import os
from pathlib import Path
import streamlit as st

from src.answer import answer, _expand_query  # _expand_query for logging-kontekst
from src.ingest import build_index, OPENAI_API_KEY
from src.telemetry import log_unknown, is_unknown  # ny: logging av "vet ikke"

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
DEBUG_UI = _env_flag("DEBUG_UI", False)  # styrer visning av intern info-boks

# Bygg indeksen hvis den mangler
def ensure_index():
    vec = DATA_DIR / "vectors.npy"
    meta = DATA_DIR / "meta.jsonl"

    if USE_OPENAI and not OPENAI_API_KEY:
        st.error(
            "Kan ikke bygge indeks – OPENAI_API_KEY mangler. "
            "Sett den i .env-filen eller i Streamlit Secrets."
        )
        st.stop()

    if not vec.exists() or not meta.exists():
        st.info("Indeks mangler – bygger nå …")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        build_index(KB_DIR)

st.set_page_config(page_title="Chatbot Asker Tennis", page_icon="🎾", layout="centered")
ensure_index()

st.title("🎾 Chatbot Asker Tennis")
mode_label = "**OpenAI**" if USE_OPENAI else "**TF-IDF**"
st.caption(f"Status: indeks `ok` • Modus: {mode_label} (modell: {CHAT_MODEL})")

q = st.text_input("Skriv spørsmålet ditt:", placeholder="F.eks. Hva koster det å leie bane?")
k = 6  # beholdt som i din versjon

if st.button("Svar") and q.strip():
    with st.spinner("Henter…"):
        text, hits = answer(q, k=k)

    st.markdown("### Svar")
    st.write(text)

    # Logg "jeg vet ikke"-tilfeller for videre forbedring av KB
    try:
        if is_unknown(text):
            expanded, preferred, extra = _expand_query(q)
            log_unknown(
                question=q,
                expanded_query=expanded,
                preferred_tags=list(preferred),
                hits=hits if isinstance(hits, list) else [],
                answer_text=text,
                meta={
                    "k": k,
                    "chat_model": CHAT_MODEL,
                    "use_openai": USE_OPENAI,
                },
            )
            if DEBUG_UI:
                st.info("(Internt) Logget som 'vet ikke' i eval/runs/unknown.*")
    except Exception as e:
        if DEBUG_UI:
            st.warning(f"(Internt) Klarte ikke å logge 'vet ikke': {e}")

    st.markdown("### Kilder")
    if not hits:
        st.write("Ingen kilder.")
    else:
        for h in hits:
            src = h.get("source", "?")
            hid = h.get("id", "?")
            try:
                sc = float(h.get("score", 0.0))
                sc_txt = f"{sc:.3f}"
            except Exception:
                sc_txt = str(h.get("score", ""))
            st.markdown(f"- **{src}** — `{hid}` (score {sc_txt})")
