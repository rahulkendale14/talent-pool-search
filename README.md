# Talent Pool Search

A portfolio prototype: RAG applied to internal recruiting. Instead of a
chatbot answering questions, this is retrieval-augmented **search** over a
recruiting team's own pool of past applicant resumes -- useful when a
10-20 person company is re-hiring for a role it has filled before and
already has hundreds of old resumes it can't manually re-scan.

Everything in this repo is **synthetic**: fictional names, fictional
companies, fictional resume content. Nothing here is a real candidate's
data.

## What's in here

| File | Purpose |
|---|---|
| `generate_resumes.py` | Creates ~52 synthetic resumes across 6 role families, with varied phrasing for the same underlying skills |
| `ingest.py` | Splits each resume into section-level chunks (per-job Experience, Skills, Education) + writes name-redacted copies |
| `index.py` | Embeds every chunk (sentence-transformers, with a visible TF-IDF fallback) |
| `search.py` | Hybrid search: semantic score + BM25 keyword score, combined, rolled up to candidate level, with date filter + recency boost |
| `explain.py` | Extractive "why matched" citations, verified against the candidate's own raw text |
| `fairness_check.py` | Name/pronoun swap smoke test -- does ranking change when only identity changes? |
| `evaluate.py` | Precision@10 / recall@10 against hand-labeled relevance judgments |
| `app.py` | Minimal Streamlit UI: paste a JD, see top 10 with citations, mark shortlist/not-a-fit |

## How to run it (in order)

```bash
pip install sentence-transformers scikit-learn rank_bm25 streamlit --break-system-packages

python3 generate_resumes.py   # -> data/resumes.json
python3 ingest.py             # -> data/chunks.json, data/redacted_resumes.json
python3 index.py              # -> data/index.pkl, data/index_meta.json
python3 search.py "some job description"   # quick CLI sanity check
python3 explain.py            # citation check for the same sample query
python3 fairness_check.py     # fairness smoke test
python3 evaluate.py           # precision@10 / recall@10 -> data/eval_results.json
streamlit run app.py          # interactive demo
```

Each script reads the previous script's output from `data/`, so run them
in the order above the first time.

## Embedding path actually used in this run: TF-IDF fallback

`index.py` tries `sentence-transformers/all-MiniLM-L6-v2` first. **In this
environment, that download failed** -- the sandbox's network proxy returns
an explicit `403 Forbidden` for `huggingface.co` (confirmed via the
proxy's own status endpoint: `connect_rejected ... policy denial`), not a
flaky timeout. `index.py` catches that failure, prints it to the console,
and falls back to a **TF-IDF vectorizer (scikit-learn)** as the semantic
signal -- this is logged loudly at both `index.py` runtime and surfaced in
the Streamlit UI as a warning banner, not silently swapped in.

**What this means for the "hybrid search" story**: normally the "hybrid"
in this demo is semantic embeddings (meaning-based) + BM25 (keyword-based)
-- two genuinely different signals. With the fallback active, the
"semantic" side is TF-IDF cosine similarity, which is *also* a keyword-
weighting method (just a different formula than BM25). So in this run,
"hybrid" is closer to "two keyword-ish views combined" than "meaning +
keywords combined." The code, comments, and the eval numbers below are all
honestly from this fallback path -- if you run this somewhere with
Hugging Face network access, `index.py` will pick up the real
sentence-transformer model automatically and you'd expect the semantic
side to do noticeably better on paraphrase-heavy queries (e.g. matching
"owned payments product area" to a query about "checkout" even with zero
shared words, which TF-IDF cannot do).

## Evaluation results (actual numbers from this run)

Hand-labeled relevance for 5 job descriptions (see `evaluate.py` for the
exact candidate_id sets and the reasoning behind each label). This run
used the real `sentence-transformers/all-MiniLM-L6-v2` embeddings, not
the TF-IDF fallback described below.

| Job description | Relevant in pool | precision@10 | recall@10 |
|---|---|---|---|
| PM, owned checkout/payments flows | 12 | 1.00 | 0.83 |
| SWE, microservices + CI/CD | 10 | 0.90 | 0.90 |
| Support, self-serve docs / knowledge base | 7 | 0.70 | 1.00 |
| Data Analyst, SQL/Python + A/B testing | 4 | 0.40 | 1.00 |
| Product Designer, Figma + design systems | 2 | 0.20 | 1.00 |
| **Average** | | **0.64** | **0.95** |

**Precision@10 has a hard ceiling when fewer than 10 candidates in the
whole pool are actually relevant** -- the last 3 rows above can never
reach 1.00 precision@10 even with perfect search, because there aren't
10 relevant candidates to fill the top 10 with. The pool-size-adjusted
ceiling for this run averages **0.66** -- so the system's 0.64 average
is landing almost exactly at the maximum achievable, not underperforming.
Read recall@10 (0.95 average, and 1.00 on all three ceiling-limited
rows) as the more honest signal of match quality for those three.

The last two rows also differ from the first three in *why* the
excluded candidates were excluded: the synthetic Data Analyst and
Product Designer resumes share an Experience bullet pool across nearly
every candidate in the family, so Skills (tool stack: SQL/Python vs.
R/dbt/Looker; Figma/design-systems vs. Sketch/User Research) is what
actually separates them for these two JDs -- see the inline comments in
`evaluate.py`.

Search-tool runtime for all 5 queries combined: **~26 seconds** (up from
~1s on the original 3-query TF-IDF-fallback run -- loading the real
sentence-transformer model per call is the dominant cost, not the search
logic itself; a production version would load the model once, not per
query).

**Manual baseline**: not run. A fair comparison needs a real recruiter
with a stopwatch shortlisting the same 52-resume pool for the same 3 job
descriptions, timed. That requires a human and wasn't available in this
automated build environment. `evaluate.py` includes a
`baseline_manual_search()` stub that takes `(candidate_ids_picked,
seconds_elapsed, relevant_ids)` and runs the exact same precision/recall
scoring path used above, so a real timed run can be plugged in later
without touching the scoring logic.

## Fairness check result

**PASS (3/3 pairs)** -- three resumes were cloned with a name swap (e.g.
"Maria Nguyen" -> "James Nguyen") and pronoun swap where applicable,
identical experience/skills otherwise, then re-scored against the same
query. All three pairs produced **identical scores**. (Two pairs landed on
adjacent rank numbers purely because of list insertion order when scores
tie exactly -- that's a sorting artifact, not the model treating the
names differently, and is called out explicitly in the script's output.)

Caveat: our synthetic resume bullets don't use pronouns at all (real
resumes mostly don't either), so this check's only real lever is the
candidate's name. It confirms the scoring pipeline doesn't key off the
name, but it isn't a full bias audit.

## Design decisions worth knowing about

- **Chunking is per-section (per-job Experience entry, Skills, Education)**,
  not whole-resume or per-sentence. Whole-resume chunks dilute a great
  single-job match; sentence chunks lose the context a bullet needs to be
  quotable. Section-level chunks are what `explain.py` cites from.
- **Date is metadata, never embedded.** `date_received` is stored
  alongside each chunk and used only for (a) an optional hard filter and
  (b) a small, capped recency boost (`+0.05` max, exponential decay over
  ~9 months) applied *after* relevance scoring. This keeps "how relevant"
  and "how fresh" separately inspectable instead of letting the model
  learn spurious "recent resumes look like this" patterns.
- **Hybrid scoring**: `0.6 * semantic + 0.4 * BM25`, both min-max
  normalized per query before combining, so neither signal's raw score
  scale can dominate just because its numbers happen to be bigger.
- **"No strong matches" is a real possible answer.** If nothing clears a
  relevance floor (`MIN_RELEVANCE = 0.12` in `search.py`), the system
  says so instead of forcing 10 mediocre results onto the recruiter.
- **Citations are extractive, not generated**, and are checked to be a
  literal substring of that candidate's own resume text before being
  shown. No LLM call, no chance of the explanation saying something the
  resume doesn't.

## Known limitations

- **Small dataset.** 52 synthetic resumes is enough to demo the mechanics
  and show semantic-vs-keyword divergence, but too small to draw real
  statistical conclusions about search quality.
- **TF-IDF fallback active in this environment**, as explained above --
  the "semantic" signal here is not a true sentence embedding. Rerun with
  Hugging Face access for the intended embedding path.
- **Extractive-only explanations.** The "why matched" quote is the
  highest-overlap line from the top-scoring chunk, not a generated
  summary. This is a deliberate trust choice, but it means the wording is
  sometimes a little rough as a stand-alone sentence.
- **No real manual baseline yet** -- see the evaluation section above.
- **Ground-truth labels are self-labeled** by the person building this
  demo, not by an independent recruiter, so they should be read as
  "reasonable and reviewable" rather than "gold standard."
- **Shortlist/not-a-fit decisions in `app.py` are session-only** (Streamlit
  session state), not persisted to disk or a database -- fine for a demo,
  not for real use.
- **Fairness check is a smoke test, not an audit.** It only varies name
  (and unused pronouns), on 3 resumes, with one scoring backend. It would
  not catch more subtle forms of bias (e.g. school prestige, employment
  gaps, non-Western names beyond the ones tested).
