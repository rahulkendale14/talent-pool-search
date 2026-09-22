"""
index.py
--------
Embeds every chunk from data/chunks.json and saves a searchable index.

EMBEDDING PATH (read this first):
  We try sentence-transformers/all-MiniLM-L6-v2 first, because a real
  embedding model captures semantic similarity ("led checkout redesign" ~=
  "owned payments product area") in a way plain keyword matching cannot.

  If the model can't be downloaded (no network access to Hugging Face, a
  proxy blocks it, etc.) we FALL BACK to a TF-IDF vectorizer as the
  "semantic-ish" signal. This is explicitly logged to the console and
  written into data/index_meta.json so nothing is silently degraded --
  a PM presenting this demo needs to know which path actually ran,
  because it changes what the semantic-vs-keyword comparison actually
  proves.

  IMPORTANT: TF-IDF is NOT a semantic method -- it's still keyword-based
  (weighted keyword overlap). When this fallback is active, the "hybrid"
  search in search.py effectively becomes two different keyword-ish views
  (TF-IDF + BM25) rather than a true semantic/keyword hybrid. We say this
  plainly in the README rather than pretending otherwise.
"""

import json
import os
import pickle
import sys

CHUNKS_PATH = "data/chunks.json"
INDEX_PATH = "data/index.pkl"
META_PATH = "data/index_meta.json"


def try_load_sentence_transformer():
    """Attempt the real embedding path. Returns (model, path_name) or (None, None)."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model, "sentence-transformers/all-MiniLM-L6-v2"
    except Exception as e:
        print(f"[index.py] sentence-transformers path FAILED: {type(e).__name__}: {e}")
        return None, None


def build_index():
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)
    texts = [c["text"] for c in chunks]

    print("[index.py] Attempting embedding path: sentence-transformers (all-MiniLM-L6-v2)...")
    model, path_name = try_load_sentence_transformer()

    if model is not None:
        print(f"[index.py] SUCCESS -- using {path_name}")
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        index_obj = {
            "backend": "sentence-transformers",
            "model_name": "all-MiniLM-L6-v2",
            "chunk_ids": [c["chunk_id"] for c in chunks],
            "embeddings": embeddings,
        }
    else:
        print("[index.py] FALLBACK ENGAGED -- using TF-IDF (scikit-learn) instead of a real "
              "sentence embedding model. This is an explicit, visible fallback caused by the "
              "model download being unavailable in this environment (see console output above).")
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(texts)
        path_name = "TF-IDF (scikit-learn) [FALLBACK]"
        index_obj = {
            "backend": "tfidf",
            "model_name": "tfidf-fallback",
            "chunk_ids": [c["chunk_id"] for c in chunks],
            "vectorizer": vectorizer,
            "tfidf_matrix": tfidf_matrix,
        }

    os.makedirs("data", exist_ok=True)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index_obj, f)

    meta = {
        "backend": index_obj["backend"],
        "model_name": index_obj["model_name"],
        "display_name": path_name,
        "num_chunks": len(chunks),
        "fallback_used": index_obj["backend"] == "tfidf",
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[index.py] Indexed {len(chunks)} chunks using backend='{index_obj['backend']}'")
    print(f"[index.py] Wrote {INDEX_PATH} and {META_PATH}")
    return meta


if __name__ == "__main__":
    build_index()
