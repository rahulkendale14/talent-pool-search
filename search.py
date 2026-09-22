"""
search.py
---------
Hybrid search over the chunk index: given a job description, score every
chunk with TWO signals, combine them, roll scores up to the CANDIDATE level,
and return the top matches.

SIGNAL 1 -- "semantic" score:
  cosine similarity between the query and each chunk, using whichever
  embedding backend index.py actually built (sentence-transformer vectors,
  or the TF-IDF fallback -- see index.py's big comment on why that matters).

SIGNAL 2 -- keyword score (BM25):
  rank_bm25's Okapi BM25 over the same chunk texts. This is kept as an
  independent second signal ON PURPOSE, even when the "semantic" side has
  fallen back to TF-IDF, because BM25's term-frequency/length-normalization
  math differs from cosine-on-TF-IDF and still adds a bit of signal
  diversity. We say plainly in the README that when the fallback is active,
  BOTH signals are keyword-ish, so the "hybrid" story is weaker than with
  real embeddings -- that's the honest limitation, not something to hide.

COMBINING THE SCORES:
  combined = 0.6 * normalized_semantic + 0.4 * normalized_bm25
  Both signals are min-max normalized to [0,1] per-query before combining
  so one signal's raw scale doesn't dominate just because it happens to
  produce bigger numbers.

DATE AS METADATA, NOT EMBEDDING:
  date_received is deliberately NEVER embedded into the chunk text or
  turned into part of the similarity vector. If it were, the model could
  learn spurious "recent resumes look like this" patterns and a candidate's
  match score would depend on when they applied rather than what they
  wrote. Instead date is stored as plain metadata and used only for (a) an
  optional hard filter and (b) a small, transparent recency boost applied
  AFTER relevance is computed. This keeps relevance and freshness separate
  and inspectable.

CANDIDATE ROLL-UP:
  A candidate can have several matching chunks (e.g. two jobs that both
  mention payments work). We take the MAX combined chunk score per
  candidate as that candidate's relevance score, then apply the recency
  boost once at the candidate level. Using max (not sum/average) means a
  candidate isn't penalized for having an Education chunk that's irrelevant
  to the query -- one great matching chunk is enough to surface them.

THRESHOLD:
  If no candidate clears MIN_RELEVANCE, we return an empty list with a
  "no strong matches" signal rather than forcing 10 mediocre results --
  a recruiter should trust that a returned list is worth reading.
"""

import json
import pickle
from collections import Counter
from datetime import datetime

import numpy as np
from rank_bm25 import BM25Okapi

CHUNKS_PATH = "data/chunks.json"
INDEX_PATH = "data/index.pkl"

SEMANTIC_WEIGHT = 0.6
BM25_WEIGHT = 0.4
MIN_RELEVANCE = 0.12          # combined-score floor before recency boost
RECENCY_BOOST_MAX = 0.05      # small, capped boost for very recent resumes
RECENCY_HALF_LIFE_DAYS = 270  # ~9 months: boost decays to half by then
TODAY = datetime(2026, 9, 22)


def _step(on_step, name, **detail):
    """Fire the optional progress callback. No-op (and no cost) when the
    caller didn't pass one -- e.g. the CLI entry point below."""
    if on_step is not None:
        on_step(name, detail)


def _load():
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)
    with open(INDEX_PATH, "rb") as f:
        index_obj = pickle.load(f)
    return chunks, index_obj


def _minmax(scores):
    scores = np.array(scores, dtype=float)
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def _semantic_scores(query, chunks, index_obj):
    if index_obj["backend"] == "sentence-transformers":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(index_obj["model_name"])
        q_emb = model.encode([query], normalize_embeddings=True)[0]
        embs = index_obj["embeddings"]
        sims = embs @ q_emb  # cosine sim since both normalized
        return sims
    else:
        # TF-IDF fallback: cosine similarity between query and chunk vectors
        from sklearn.metrics.pairwise import cosine_similarity
        vectorizer = index_obj["vectorizer"]
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, index_obj["tfidf_matrix"])[0]
        return sims


def _bm25_scores(query, chunks):
    tokenized_corpus = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    return np.array(bm25.get_scores(tokenized_query))


def _recency_boost(date_received_str):
    """Small, transparent boost: newer resumes get up to RECENCY_BOOST_MAX
    extra, decaying exponentially with age. Capped so it can never flip a
    weak match above a strong one -- it only nudges among close matches."""
    received = datetime.strptime(date_received_str, "%Y-%m-%d")
    age_days = max((TODAY - received).days, 0)
    decay = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
    return RECENCY_BOOST_MAX * decay


def search(query, date_from=None, date_to=None, top_k=10, on_step=None):
    """
    Returns a dict:
      {"backend": ..., "no_strong_matches": bool, "results": [ {candidate_id, score, best_chunk...}, ... ]}
    date_from / date_to: optional "YYYY-MM-DD" strings to filter date_received.
    on_step: optional callback(step_name: str, detail: dict) fired once per
      pipeline stage below, in order -- lets a caller (e.g. app.py) narrate
      what's actually happening instead of just returning a final list.
      Purely observational: passing None (the default, used by the CLI
      entry point) changes no scoring behavior.
    """
    chunks, index_obj = _load()
    _step(on_step, "load",
          num_chunks=len(chunks),
          num_candidates=len({c["candidate_id"] for c in chunks}),
          chunk_types=dict(Counter(c["section_type"] for c in chunks)),
          backend=index_obj["backend"],
          model_name=index_obj.get("model_name"))

    # Optional date filter applied BEFORE scoring (metadata filter, not embedding).
    if date_from or date_to:
        filtered = []
        for c in chunks:
            d = c["date_received"]
            if date_from and d < date_from:
                continue
            if date_to and d > date_to:
                continue
            filtered.append(c)
        chunks = filtered
        _step(on_step, "date_filter", date_from=date_from, date_to=date_to,
              num_chunks_kept=len(chunks))
        if not chunks:
            return {"backend": index_obj["backend"], "no_strong_matches": True, "results": []}
        # Need to rebuild a matching index_obj slice for TF-IDF/embeddings
        keep_ids = {c["chunk_id"] for c in chunks}
        keep_idx = [i for i, cid in enumerate(index_obj["chunk_ids"]) if cid in keep_ids]
        if index_obj["backend"] == "sentence-transformers":
            index_obj = {**index_obj, "embeddings": index_obj["embeddings"][keep_idx],
                         "chunk_ids": [index_obj["chunk_ids"][i] for i in keep_idx]}
        else:
            index_obj = {**index_obj, "tfidf_matrix": index_obj["tfidf_matrix"][keep_idx],
                         "chunk_ids": [index_obj["chunk_ids"][i] for i in keep_idx]}

    sem = _semantic_scores(query, chunks, index_obj)
    _step(on_step, "embed_query",
          backend=index_obj["backend"], model_name=index_obj.get("model_name"))

    bm25 = _bm25_scores(query, chunks)
    _step(on_step, "bm25", num_chunks_scored=len(chunks))

    sem_norm = _minmax(sem)
    bm25_norm = _minmax(bm25)
    combined = SEMANTIC_WEIGHT * sem_norm + BM25_WEIGHT * bm25_norm
    _step(on_step, "combine", semantic_weight=SEMANTIC_WEIGHT, bm25_weight=BM25_WEIGHT)

    # Roll up to candidate level: take the best-scoring chunk per candidate.
    best_per_candidate = {}
    for i, c in enumerate(chunks):
        cid = c["candidate_id"]
        score = float(combined[i])
        if cid not in best_per_candidate or score > best_per_candidate[cid]["raw_score"]:
            best_per_candidate[cid] = {
                "candidate_id": cid,
                "raw_score": score,
                "best_chunk": c,
                "semantic_score": float(sem_norm[i]),
                "bm25_score": float(bm25_norm[i]),
            }
    _step(on_step, "rollup", num_candidates=len(best_per_candidate))

    # Apply recency boost once, at candidate level, using the best chunk's date.
    for entry in best_per_candidate.values():
        boost = _recency_boost(entry["best_chunk"]["date_received"])
        entry["final_score"] = entry["raw_score"] + boost
        entry["recency_boost"] = boost
    _step(on_step, "recency_boost",
          max_boost=RECENCY_BOOST_MAX, half_life_days=RECENCY_HALF_LIFE_DAYS)

    ranked = sorted(best_per_candidate.values(), key=lambda e: e["final_score"], reverse=True)
    strong = [e for e in ranked if e["raw_score"] >= MIN_RELEVANCE]
    _step(on_step, "threshold",
          min_relevance=MIN_RELEVANCE, num_total=len(ranked), num_strong=len(strong))

    if not strong:
        return {"backend": index_obj["backend"], "no_strong_matches": True, "results": []}

    _step(on_step, "done", num_returned=min(len(strong), top_k))
    return {
        "backend": index_obj["backend"],
        "no_strong_matches": False,
        "results": strong[:top_k],
    }


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "Product manager with experience owning checkout and payments flows"
    result = search(query)
    print(f"Backend: {result['backend']}")
    if result["no_strong_matches"]:
        print("No strong matches for this query.")
    else:
        for i, r in enumerate(result["results"], 1):
            print(f"{i}. {r['candidate_id']}  score={r['final_score']:.3f} "
                  f"(sem={r['semantic_score']:.2f} bm25={r['bm25_score']:.2f} boost={r['recency_boost']:.3f}) "
                  f"received={r['best_chunk']['date_received']}")
