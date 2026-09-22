"""
explain.py
----------
For each search result, produce a short "why matched" explanation that
QUOTES text verbatim from that candidate's own resume.

PM-relevant "why" (the citation requirement):
  A recruiter should never have to trust a black-box score. Every result
  must point at something the candidate actually wrote, so a human can
  verify the match in two seconds instead of re-reading the whole resume.
  This is a simple EXTRACTIVE approach (no generation, no chance of the
  explanation saying something the resume doesn't): we take the
  highest-scoring chunk for that candidate (already computed in search.py)
  and pull out the single sentence/bullet line from it that overlaps most
  with the query terms. We then verify the quote is a literal substring of
  that candidate's raw_text before returning it -- if that check ever
  fails, we flag it rather than silently showing an unverifiable quote.
"""

import json
import re

RESUMES_PATH = "data/resumes.json"


def _load_resumes():
    with open(RESUMES_PATH) as f:
        resumes = json.load(f)
    return {r["candidate_id"]: r for r in resumes}


def _split_into_lines(chunk_text):
    """Split a chunk into candidate quote units: bullet lines if present,
    otherwise the whole line (Skills/Education chunks are single lines)."""
    lines = [l.strip("- ").strip() for l in chunk_text.split("\n") if l.strip()]
    # Drop the "Title, Company (dates)" header line for Experience chunks --
    # it's not a good "why matched" quote on its own.
    lines = [l for l in lines if not re.match(r"^[\w\s,&\-]+\(\d{4}", l)]
    return lines if lines else [chunk_text.strip()]


# Small stopword list so generic connective words (with, and, the, for...)
# don't outweigh the actual skill/role terms when picking the best quote
# line within a chunk. Without this, a line like "Partnered with UX
# research..." can out-score "Owned the payments product area..." on a
# query containing "with experience", purely on stopword overlap -- which
# would make the citation misleading even though the underlying chunk
# score is correct.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "with", "for", "of", "to", "in", "on",
    "at", "by", "is", "are", "was", "were", "be", "as", "that", "this",
    "it", "from", "who", "has", "have", "had", "experience", "role",
}


def best_quote_for_chunk(query, chunk_text):
    """Pick the line in this chunk with the most query-term overlap
    (ignoring generic stopwords so the match reflects real skill terms)."""
    query_terms = set(re.findall(r"[a-zA-Z]+", query.lower())) - _STOPWORDS
    lines = _split_into_lines(chunk_text)
    best_line, best_overlap = lines[0], -1
    for line in lines:
        line_terms = set(re.findall(r"[a-zA-Z]+", line.lower())) - _STOPWORDS
        overlap = len(query_terms & line_terms)
        if overlap > best_overlap:
            best_overlap = overlap
            best_line = line
    return best_line


def explain_result(query, result_entry, resumes_by_id=None):
    """Given one search.py result entry, return an explanation dict:
    {candidate_id, full_name, quote, section_type, verified}."""
    if resumes_by_id is None:
        resumes_by_id = _load_resumes()

    cid = result_entry["candidate_id"]
    chunk = result_entry["best_chunk"]
    resume = resumes_by_id[cid]

    quote = best_quote_for_chunk(query, chunk["text"])

    # Traceability check: the quote must literally appear in this
    # candidate's own raw_text. If not, we say so instead of hiding it.
    verified = quote in resume["raw_text"]

    return {
        "candidate_id": cid,
        "full_name": resume["full_name"],
        "quote": quote,
        "section_type": chunk["section_type"],
        "date_received": chunk["date_received"],
        "verified": verified,
    }


def explain_results(query, results):
    resumes_by_id = _load_resumes()
    return [explain_result(query, r, resumes_by_id) for r in results]


if __name__ == "__main__":
    from search import search
    query = "Product manager with experience owning checkout and payments flows"
    result = search(query)
    for r in explain_results(query, result["results"]):
        check = "OK" if r["verified"] else "**UNVERIFIED**"
        print(f"{r['candidate_id']} ({r['full_name']}) [{r['section_type']}, {check}]:")
        print(f'  "{r["quote"]}"')
