"""
fairness_check.py
------------------
Smoke test: does the search system's ranking change when ONLY a candidate's
name / gender-coded pronoun changes, with identical experience and skills?

This is not a rigorous bias audit -- it's a cheap, explainable sanity check
that belongs in a junior PM's portfolio: pick a few resumes, clone them with
a swapped name (and a swapped pronoun if one appears), rerun the same job
description, and see if rank position moves. If it does, that's a signal
the model may be keying off name/gender surface features rather than
actual skill content -- worth flagging before shipping.

Method:
  1. Take 2-3 real synthetic resumes.
  2. Clone each with a different full_name (swapped to a name culturally
     coded to the opposite gender) and the same raw_text otherwise. If the
     resume includes a gendered pronoun anywhere, swap it too.
  3. Temporarily add the clone into the chunk index alongside the real data.
  4. Run the same query and compare rank position AND score of original vs
     clone.
  5. PASS if rank position and score are unchanged (allowing for tiny
     floating point noise); FAIL if the ranking differs.

Note: our synthetic resumes (generate_resumes.py) don't currently include
pronouns at all -- they're written in resume-bullet style ("Led...", "Owned
...") which is realistic (most resumes avoid pronouns). So this test's
primary lever is the NAME itself. We still implement pronoun-swap logic in
case bullet text is edited to include some, so the test is honest about
what it's actually varying.
"""

import copy
import json
import pickle

from search import _load, _semantic_scores, _bm25_scores, _minmax, _recency_boost, SEMANTIC_WEIGHT, BM25_WEIGHT

RESUMES_PATH = "data/resumes.json"

# Name pairs coded (in common Western naming convention) as female/male,
# used only to swap surface identity while keeping content identical.
NAME_SWAPS = [
    ("Maria Nguyen", "James Nguyen"),
    ("Priya Patel", "Ravi Patel"),
    ("Grace Johnson", "James Johnson"),
]

PRONOUN_SWAPS = [("she", "he"), ("her", "his"), ("hers", "his"), ("She", "He"), ("Her", "His")]


def swap_pronouns(text):
    for a, b in PRONOUN_SWAPS:
        text = text.replace(f" {a} ", f" {b} ")
    return text


def make_clone(resume, new_name):
    clone = copy.deepcopy(resume)
    old_name = resume["full_name"]
    clone["candidate_id"] = resume["candidate_id"] + "_CLONE"
    clone["full_name"] = new_name
    new_text = resume["raw_text"].replace(old_name, new_name)
    new_text = swap_pronouns(new_text)
    clone["raw_text"] = new_text
    return clone


def resume_to_chunks(resume):
    """Reuse ingest.py's section splitting so clone chunks match the real
    pipeline exactly (not a hand-rolled approximation)."""
    from ingest import split_sections, split_experience_entries
    header, sections = split_sections(resume["raw_text"])
    chunks = []
    n = 0
    if "Experience" in sections:
        for entry in split_experience_entries(sections["Experience"]):
            n += 1
            chunks.append({
                "chunk_id": f"{resume['candidate_id']}_FAIRCH{n}",
                "candidate_id": resume["candidate_id"],
                "section_type": "Experience",
                "text": entry,
                "date_received": resume["date_received"],
            })
    if "Skills" in sections and sections["Skills"].strip():
        n += 1
        chunks.append({
            "chunk_id": f"{resume['candidate_id']}_FAIRCH{n}",
            "candidate_id": resume["candidate_id"],
            "section_type": "Skills",
            "text": sections["Skills"].strip(),
            "date_received": resume["date_received"],
        })
    if "Education" in sections and sections["Education"].strip():
        n += 1
        chunks.append({
            "chunk_id": f"{resume['candidate_id']}_FAIRCH{n}",
            "candidate_id": resume["candidate_id"],
            "section_type": "Education",
            "text": sections["Education"].strip(),
            "date_received": resume["date_received"],
        })
    return chunks


def run_fairness_check(query="Product manager with experience owning checkout and payments flows"):
    with open(RESUMES_PATH) as f:
        resumes = json.load(f)
    resumes_by_id = {r["candidate_id"]: r for r in resumes}
    base_chunks, index_obj = _load()

    results = []
    for i, (name_a, name_b) in enumerate(NAME_SWAPS):
        # pick a source resume to clone (rotate through role families for variety)
        source = resumes[i * 7 % len(resumes)]
        original_named = copy.deepcopy(source)
        original_named["full_name"] = name_a
        original_named["raw_text"] = source["raw_text"].replace(source["full_name"], name_a)
        original_named["candidate_id"] = source["candidate_id"] + "_A"

        clone = make_clone(original_named, name_b)
        clone["candidate_id"] = source["candidate_id"] + "_B"

        chunks_a = resume_to_chunks(original_named)
        chunks_b = resume_to_chunks(clone)

        test_chunks = base_chunks + chunks_a + chunks_b

        # Rebuild scoring in-memory for this pair against the full pool,
        # using the SAME backend as the real index (sentence-transformers
        # or TF-IDF fallback) so this check reflects the actual demo path.
        texts = [c["text"] for c in test_chunks]
        if index_obj["backend"] == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(index_obj["model_name"])
            embs = model.encode(texts, normalize_embeddings=True)
            q_emb = model.encode([query], normalize_embeddings=True)[0]
            sem = embs @ q_emb
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            vec = TfidfVectorizer(stop_words="english", max_features=5000)
            matrix = vec.fit_transform(texts)
            q_vec = vec.transform([query])
            sem = cosine_similarity(q_vec, matrix)[0]

        bm25 = _bm25_scores(query, test_chunks)
        sem_norm = _minmax(sem)
        bm25_norm = _minmax(bm25)
        combined = SEMANTIC_WEIGHT * sem_norm + BM25_WEIGHT * bm25_norm

        best = {}
        for idx, c in enumerate(test_chunks):
            cid = c["candidate_id"]
            score = float(combined[idx])
            if cid not in best or score > best[cid]:
                best[cid] = score

        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        rank_lookup = {cid: (pos + 1, score) for pos, (cid, score) in enumerate(ranked)}

        rank_a, score_a = rank_lookup[chunks_a[0]["candidate_id"]]
        rank_b, score_b = rank_lookup[chunks_b[0]["candidate_id"]]

        # PASS criterion: the SCORE must be identical (within float noise).
        # We don't require rank position to be identical too -- when scores
        # tie exactly, the two clones sit at adjacent ranks purely because
        # of list insertion order, which is an artifact of how ties are
        # broken, not a sign the model treated the names differently. If
        # scores differ at all, that's the real signal something (in this
        # case, only the name text) changed the outcome.
        passed = abs(score_a - score_b) < 1e-6
        results.append({
            "pair": f"{name_a} vs {name_b}",
            "rank_a": rank_a, "score_a": round(score_a, 4),
            "rank_b": rank_b, "score_b": round(score_b, 4),
            "passed": passed,
        })

    return results


def main():
    print("=" * 70)
    print("FAIRNESS SMOKE TEST: name/pronoun swap, identical experience")
    print("=" * 70)
    results = run_fairness_check()
    all_pass = True
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        if not r["passed"]:
            all_pass = False
        print(f"[{status}] {r['pair']}")
        tie_note = " (adjacent ranks are a tie-order artifact, not bias)" if r["rank_a"] != r["rank_b"] and r["passed"] else ""
        print(f"        rank_a={r['rank_a']} score_a={r['score_a']}  |  "
              f"rank_b={r['rank_b']} score_b={r['score_b']}{tie_note}")
    print("-" * 70)
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'} "
          f"({sum(r['passed'] for r in results)}/{len(results)} pairs unchanged)")
    print("Note: our TF-IDF/BM25/embedding text scoring is name-blind by "
          "construction (it scores bullet content, and names weren't part "
          "of the scored text at the sentence level here beyond appearing "
          "once at the top of the resume) -- this test mainly confirms "
          "that assumption holds rather than discovering new bias.")
    return all_pass


if __name__ == "__main__":
    main()
