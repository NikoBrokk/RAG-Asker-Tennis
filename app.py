import streamlit as st
from src.answer import answer
from src.retrieve import search  # optional import to trigger index build

st.set_page_config(page_title="RAG Demo – Asker Tennis", page_icon="🔎", layout="centered")
st.title("🔎 RAG Demo (GitHub)")

q = st.text_input("Skriv spørsmålet ditt:", placeholder="F.eks. Hvordan resetter jeg passordet?")
k = st.slider("Antall kilder", 2, 12, 6)

if st.button("Svar") and q.strip():
    with st.spinner("Henter…"):
        text, hits = answer(q, k=k)
    st.markdown("### Svar")
    st.write(text)
    st.markdown("### Kilder")
    for h in hits:
        st.markdown(f"- **{h['source']}** — `{h['id']}` (score {h['score']:.3f})")
