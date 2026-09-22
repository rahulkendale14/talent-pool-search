"""
app.py
------
Minimal Streamlit demo UI for Talent Pool Search.

Flow: paste a job description -> see top 10 candidates with why-matched
quotes and date_received -> mark each shortlist / not-a-fit.

This is intentionally simple -- it's a demo to show the RAG-for-recruiting
mechanics end to end, not a production ATS. Shortlist decisions are kept
in Streamlit session state only (nothing persisted to disk), which is
fine for a portfolio demo but is called out in the README as a limitation.

Run with:  streamlit run app.py
"""

import json

import streamlit as st

from search import search
from explain import explain_results

st.set_page_config(page_title="Talent Pool Search", layout="wide")

st.title("Talent Pool Search")
st.info(
    "This demo uses fictional resumes for testing. Try any job description "
    "and see how the matching works."
)
st.caption(
    "Internal resume search for re-hiring: paste a job description, get the "
    "top candidates from your existing resume pool with quoted evidence for "
    "why each one matched."
)

# Surface which embedding backend is actually running -- this is the
# single most important "under the hood" fact for a PM demoing this, so
# it's shown up front rather than buried in logs.
try:
    with open("data/index_meta.json") as f:
        meta = json.load(f)
    if meta["fallback_used"]:
        st.warning(
            f"Embedding backend: **TF-IDF fallback** (sentence-transformers model "
            f"could not be downloaded in this environment). The 'semantic' score "
            f"below is TF-IDF cosine similarity, not a true sentence embedding -- "
            f"see README for what this means for result quality.",
            icon="⚠️",
        )
    else:
        st.success(f"Embedding backend: **{meta['display_name']}**", icon="✅")
except FileNotFoundError:
    st.error("No index found. Run generate_resumes.py, ingest.py, then index.py first.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    date_from = st.text_input("Date received from (YYYY-MM-DD)", value="")
    date_to = st.text_input("Date received to (YYYY-MM-DD)", value="")
    st.markdown("---")
    st.markdown(
        "**Scoring**: 60% semantic similarity + 40% BM25 keyword score, "
        "plus a small recency boost (max +0.05, decaying over ~9 months). "
        "Date is metadata only -- never embedded into the match score "
        "beyond that capped boost."
    )

if "shortlist" not in st.session_state:
    st.session_state.shortlist = {}  # candidate_id -> "shortlist" | "not_a_fit"

query = st.text_area(
    "Job description",
    height=120,
    placeholder="e.g. Looking for a Product Manager who has owned checkout or payments "
                "flows and driven measurable conversion improvements.",
)

run = st.button("Search", type="primary")


def _step_text(name, d):
    """Turn one on_step callback into the human-readable line shown in the
    live processing trace -- same idea as narrating a tool call: name the
    concrete mechanism and the real numbers from this run, not a label."""
    if name == "load":
        backend = (
            f"sentence-transformers/{d['model_name']}"
            if d["backend"] == "sentence-transformers" else "TF-IDF fallback"
        )
        types = ", ".join(f"{n} {t}" for t, n in d["chunk_types"].items())
        return (
            f"**Loaded index** — {d['num_chunks']} chunks across "
            f"{d['num_candidates']} candidates ({types}). Embedding backend: **{backend}**."
        )
    if name == "date_filter":
        return (
            f"**Applied date filter** ({d['date_from'] or '…'} to {d['date_to'] or '…'}) "
            f"— {d['num_chunks_kept']} chunks remain."
        )
    if name == "embed_query":
        backend = (
            f"sentence-transformers/{d['model_name']}"
            if d["backend"] == "sentence-transformers" else "TF-IDF vectorizer"
        )
        return f"**Embedded the query** with {backend}, cosine similarity vs. every chunk."
    if name == "bm25":
        return f"**Scored {d['num_chunks_scored']} chunks with BM25** (keyword/term-frequency signal)."
    if name == "combine":
        return (
            f"**Combined hybrid score** — {int(d['semantic_weight']*100)}% semantic "
            f"+ {int(d['bm25_weight']*100)}% BM25 (both min-max normalized first)."
        )
    if name == "rollup":
        return f"**Rolled up to candidate level** — best-scoring chunk per candidate, {d['num_candidates']} candidates scored."
    if name == "recency_boost":
        return (
            f"**Applied recency boost** — up to +{d['max_boost']}, decaying with a "
            f"~{d['half_life_days']}-day half-life. Relevance and freshness stay separate."
        )
    if name == "threshold":
        return (
            f"**Relevance floor check** ({d['min_relevance']}) — {d['num_strong']} of "
            f"{d['num_total']} candidates cleared it."
        )
    if name == "done":
        return f"**Done** — returning top {d['num_returned']}."
    return f"{name}: {d}"


if run and query.strip():
    with st.status("Processing search…", expanded=True) as status:
        def on_step(name, detail):
            st.write(_step_text(name, detail))

        result = search(
            query,
            date_from=date_from.strip() or None,
            date_to=date_to.strip() or None,
            on_step=on_step,
        )
        status.update(
            label=f"Search complete — backend: {result['backend']}",
            state="complete",
        )
    st.session_state["last_result"] = result
    st.session_state["last_query"] = query

if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    query = st.session_state["last_query"]

    if result["no_strong_matches"]:
        st.info(
            "No strong matches for this job description. Try broadening it, or "
            "widening the date filter -- returning 10 weak matches would waste "
            "recruiter time, so the system declines instead."
        )
    else:
        explanations = {e["candidate_id"]: e for e in explain_results(query, result["results"])}
        st.subheader(f"Top {len(result['results'])} candidates")

        for rank, r in enumerate(result["results"], 1):
            cid = r["candidate_id"]
            exp = explanations[cid]
            with st.container(border=True):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**#{rank}. {exp['full_name']}**  ({cid})")
                    st.markdown(
                        f"> “{exp['quote']}” "
                        f"*(from {exp['section_type']} section)*"
                    )
                    st.caption(
                        f"Date received: {exp['date_received']}  |  "
                        f"score={r['final_score']:.3f} "
                        f"(semantic={r['semantic_score']:.2f}, "
                        f"bm25={r['bm25_score']:.2f}, "
                        f"recency boost=+{r['recency_boost']:.3f})"
                    )
                    if not exp["verified"]:
                        st.error("Quote could not be verified against the raw resume text.")
                with col2:
                    choice = st.radio(
                        "Decision",
                        ["Undecided", "Shortlist", "Not a fit"],
                        key=f"decision_{cid}",
                        label_visibility="collapsed",
                    )
                    st.session_state.shortlist[cid] = choice

        decided = {cid: v for cid, v in st.session_state.shortlist.items() if v != "Undecided"}
        if decided:
            st.markdown("---")
            st.subheader("Your decisions so far")
            shortlisted = [cid for cid, v in decided.items() if v == "Shortlist"]
            rejected = [cid for cid, v in decided.items() if v == "Not a fit"]
            st.write(f"Shortlisted: {shortlisted if shortlisted else 'none yet'}")
            st.write(f"Not a fit: {rejected if rejected else 'none yet'}")
