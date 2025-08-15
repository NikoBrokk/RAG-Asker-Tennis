from __future__ import annotations
import json, os, re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.utils import env_flag

try:
    import streamlit as st  # type: ignore
except Exception:
    st = None

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def _get_secret(name: str) -> str | None:
    """
    Hent verdi fra miljøvariabel eller Streamlit Secrets – trygt i CI.
    - Foretrekker ENV først (CI/GHA setter ofte env direkte).
    - Aksess til st.secrets kapsles i try/except siden bare *berøring* kan kaste
      StreamlitSecretNotFoundError hvis secrets.toml ikke finnes.
    """
    # 1) ENV først
    val = os.getenv(name)
    if isinstance(val, str) and val.strip():
        return val.strip()
    # 2) Streamlit Secrets (best effort)
    try:
        import streamlit as _st  # lokal import for å unngå sideeffekter
        try:
            sval = _st.secrets[name]  # kan kaste KeyError/StreamlitSecretNotFoundError
            return sval.strip() if isinstance(sval, str) else sval
        except Exception:
            return None
    except Exception:
        return None

# ---------- Konfig ----------
# Tving OpenAI som default i Cloud; kan overstyres via USE_OPENAI=0
USE_OPENAI = env_flag("USE_OPENAI", True)
EMBED_MODEL = _get_secret("EMBED_MODEL") or "text-embedding-3-small"
DATA_DIR = Path(_get_secret("DATA_DIR") or "data")
KB_DIR_DEFAULT = _get_secret("KB_DIR") or "kb"

# ---------- OpenAI-klient på modulnivå ----------
OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")
OPENAI_PROJECT = _get_secret("OPENAI_PROJECT")  # valgfri (for sk-proj-… nøkler)
if not OPENAI_API_KEY:
    msg = ("Mangler/ugyldig `OPENAI_API_KEY`. "
           "Legg inn nøkkelen i Streamlit Secrets eller `.env`.")
    if st:
        st.error(msg)
    raise RuntimeError(msg)
client = OpenAI(api_key=OPENAI_API_KEY, project=OPENAI_PROJECT or None)

# ---------- Korpushjelpere (gjenbruker logikken fra retrieve) ----------
def _read_text_file(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def _strip(txt: str) -> str:
    # fjern codefences og komprimer whitespace
    txt = re.sub(r"```.*?```", " ", txt, flags=re.S)
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()

def _iter_docs(kb_root: Path) -> List[Dict]:
    out: List[Dict] = []

    for p in sorted(list(kb_root.rglob("*.md")) + list(Path("data/processed").rglob("*.jsonl"))):
        if not p.is_file():
            continue

        if p.suffix.lower() == ".md":
            raw = _read_text_file(p)
            clean = _strip(raw)
            if not clean:
                continue

            chunks = [clean[i:i + 700] for i in range(0, len(clean), 700 - 120)]
            for ci, ch in enumerate(chunks):
                out.append({
                    "text": ch,
                    "source": str(p).replace("\\", "/"),
                    "title": p.stem.replace("-", " "),
                    "doc_type": None,
                    "version_date": None,
                    "page": None,
                    "chunk_idx": ci,
                    "id": f"{p.as_posix()}#{ci}",
                })

        else:
            ci = 0
            for line in _read_text_file(p).splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                txt = _strip(obj.get("text", ""))
                if not txt:
                    continue

                src = obj.get("metadata", {}).get("source") or str(p)
                out.append({
                    "text": txt,
                    "source": str(src).replace("\\", "/"),
                    "title": obj.get("metadata", {}).get("title"),
                    "doc_type": obj.get("metadata", {}).get("doc_type"),
                    "version_date": obj.get("metadata", {}).get("version_date"),
                    "page": obj.get("metadata", {}).get("page"),
                    "chunk_idx": ci,
                    "id": f"{Path(src).as_posix()}#{ci}",
                })
                ci += 1

    return out

# ---------- OpenAI-embeddings eller TF-IDF til disk ----------
def _save_meta(meta: List[Dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (DATA_DIR / "meta.jsonl").open("w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

def _build_openai_embeddings(chunks: List[Dict], batch_size: int = 64) -> np.ndarray:
    """Bygg normaliserte embeddings med batching og bedre feilhåndtering."""
    if not OPENAI_API_KEY:
        msg = (
            "OPENAI_API_KEY mangler eller er ugyldig. "
            "Sett den i .env-filen eller i Streamlit Secrets."
        )
        if st:
            st.error(msg)
        raise RuntimeError(msg)

    texts = [d["text"] for d in chunks]
    vecs: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            r = client.embeddings.create(
                model=EMBED_MODEL,
                input=batch
            )
        except openai.error.AuthenticationError:
            msg = "Feil ved autentisering mot OpenAI – sjekk API-nøkkelen."
            if st:
                st.error(msg)
            raise RuntimeError(msg)
        except Exception as e:
            msg = f"Uventet feil ved henting av embeddings: {e}"
            if st:
                st.error(msg)
            raise RuntimeError(msg)

        for item in r.data:
            v = np.asarray(item.embedding, dtype="float32")
            v = v / (np.linalg.norm(v) + 1e-12)
            vecs.append(v)

    if not vecs:
        return np.zeros((0, 1536), dtype="float32")

    return np.vstack(vecs)


def _build_tfidf_dense(chunks: List[Dict]) -> Tuple[np.ndarray, object]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pickle

    texts = [d["text"] for d in chunks] or [""]
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.95,
        strip_accents="unicode",
        lowercase=True,
        norm="l2",
        sublinear_tf=True,
        max_features=60000,
    )

    mtx = vec.fit_transform(texts)
    dense = mtx.toarray()
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    dense = (dense / (norms + 1e-12)).astype("float32")

    with (DATA_DIR / "vectorizer.pkl").open("wb") as f:
        pickle.dump(vec, f)
    return dense, vec

def _maybe_write_faiss(vectors: np.ndarray) -> None:
    if faiss is None or vectors.size == 0:
        return
    idx = faiss.IndexFlatIP(vectors.shape[1])
    idx.add(vectors)
    faiss.write_index(idx, str(DATA_DIR / "index.faiss"))

def build_index(kb_dir: str | Path = KB_DIR_DEFAULT) -> None:
    kb_root = Path(kb_dir)
    chunks = _iter_docs(kb_root)

    if USE_OPENAI:
        vectors = _build_openai_embeddings(chunks)
        np.save(DATA_DIR / "vectors.npy", vectors)
        _save_meta(chunks)
        _maybe_write_faiss(vectors)
        print(f"[ingest] OpenAI-embeddings for {len(chunks)} biter skrevet til {DATA_DIR}/vectors.npy og {DATA_DIR}/meta.jsonl.")
    else:
        vectors, _ = _build_tfidf_dense(chunks)
        np.save(DATA_DIR / "vectors.npy", vectors)
        _save_meta(chunks)
        _maybe_write_faiss(vectors)
        print(f"[ingest] TF-IDF vektorer for {len(chunks)} biter skrevet til {DATA_DIR}/vectors.npy og {DATA_DIR}/meta.jsonl.")
