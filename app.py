import os
from pathlib import Path
import streamlit as st

from src.answer import answer, _expand_query
from src.ingest import build_index, OPENAI_API_KEY
from src.telemetry import log_unknown, is_unknown

def _env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None and name in st.secrets:
        v = str(st.secrets[name])
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}

# Konfigurasjon (secrets > env)
USE_OPENAI = _env_flag("USE_OPENAI", False)
CHAT_MODEL = os.getenv("CHAT_MODEL", st.secrets.get("CHAT_MODEL", "gpt-4o-mini" if USE_OPENAI else "tf-idf"))
DATA_DIR = Path(os.getenv("DATA_DIR", st.secrets.get("DATA_DIR", "data")))
KB_DIR = os.getenv("KB_DIR", st.secrets.get("KB_DIR", "kb"))
DEBUG_UI = _env_flag("DEBUG_UI", False)

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

st.set_page_config(page_title="🎾 Chatbot Asker Tennis", page_icon="🎾", layout="centered")
ensure_index()

# Initialiser samtalehistorikk
if "history" not in st.session_state:
    st.session_state.history = []

st.title("🎾 Chatbot Asker Tennis")
st.caption("Få svar på ofte stilte spørsmål om klubben, banebooking, priser m.m.")

mode_label = "**OpenAI**" if USE_OPENAI else "**TF-IDF**"
st.caption(f"Status: indeks `ok` • Modus: {mode_label} (modell: {CHAT_MODEL})")

# Input og knapper
q = st.text_input("Skriv spørsmålet ditt:", placeholder="F.eks. Hva koster det å leie bane?")
col1, col2 = st.columns([1, 1])
svar_btn = col1.button("Svar")
reset_btn = col2.button("Start ny samtale")

if reset_btn:
    st.session_state.history = []
    # Tilbakestill inputfeltet også (clear on frontend)
    q = ""
    st.experimental_rerun()

if svar_btn and q.strip():
    with st.spinner("Henter svar…"):
        text, hits = answer(q, k=6)
        # Lagre i historikk
        st.session_state.history.append(("**Spørsmål:** " + q, text))

    st.markdown("### Svar")
    # Vis hele samtalehistorikken
    for i, (user_q, answer_text) in enumerate(st.session_state.history, start=1):
        # Bruk en litt annen visuell separasjon mellom parene
        if i < len(st.session_state.history):
            st.write(f"{user_q}")
            st.write(answer_text)
            st.markdown("---")  # horisontal linje mellom meldinger
        else:
            # Siste Q&A (nyeste)
            st.write(f"{user_q}")
            st.write(answer_text)

    # Logg "vet ikke"-tilfeller for videre forbedring av KB
    try:
        if is_unknown(text):
            expanded, preferred, extra = _expand_query(q)
            log_unknown(
                question=q,
                expanded_query=expanded,
                preferred_tags=list(preferred),
                hits=hits if isinstance(hits, list) else [],
                answer_text=text,
                meta={"k": 6, "chat_model": CHAT_MODEL, "use_openai": USE_OPENAI},
            )
            if DEBUG_UI:
                st.info("(Internt) Logget spørsmål uten svar i eval/runs/unknown.*")
    except Exception as e:
        if DEBUG_UI:
            st.warning(f"(Internt) Klarte ikke å logge unknown-spm: {e}")

    # Vis kilder uten at lenker forstyrrer
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
            # Unngå klikkbare lenker (bruk kodeformat hvis src er URL)
            if src.startswith("http"):
                st.markdown(f"- `{src}` — `{hid}` (score {sc_txt})")
            else:
                st.markdown(f"- **{src}** — `{hid}` (score {sc_txt})")
