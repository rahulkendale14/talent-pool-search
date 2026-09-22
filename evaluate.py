"""
evaluate.py
-----------
Evaluation harness: hand-labeled relevance judgments for 3 sample job
descriptions, scored against the search system with precision@10 and
recall@10.

HOW THE LABELS WERE MADE:
  I (the person building this demo) read every synthetic resume in the
  relevant role family for each JD (see data/resumes.json) and manually
  decided which candidates actually contain a bullet matching the JD's
  core ask -- e.g. for the payments/checkout JD, a candidate only counts
  as "relevant" if one of their Experience bullets is literally about
  checkout, payments, or the purchase flow (not just "is a PM"). This
  mirrors how a recruiter would judge relevance: role title alone isn't
  enough, the actual experience has to match. The label sets are the
  GROUND_TRUTH dict below -- they're plain data, so anyone reviewing this
  code can check them against data/resumes.json themselves.

METRICS:
  precision@10 = (# of the top 10 results that are truly relevant) / 10
  recall@10    = (# of the top 10 results that are truly relevant) / (total relevant in the pool)

  Precision tells a recruiter "how much of what I'm shown is worth my
  time." Recall tells them "how much of what's actually in my pool did
  this search surface." Both matter for a re-hiring tool: too low
  precision wastes recruiter time, too low recall means good candidates
  never get seen.

MANUAL BASELINE (NOT RUN HERE):
  The real comparison a PM would want is: how long does it take a human
  recruiter to manually shortlist ~10 candidates from the same 52-resume
  pool for the same JD, and how does their precision/recall compare? That
  requires an actual person with a stopwatch, which isn't available in
  this automated environment, so it was NOT run. `baseline_manual_search`
  below is a stub: pass a real timed run's results (candidate_ids they
  picked + seconds elapsed) and it slots into this harness's exact
  scoring path, so plugging in a real baseline later is a few lines, not
  a rewrite.
"""

import json
import time

from search import search

GROUND_TRUTH = {
    "PM checkout/payments": {
        "query": (
            "We are hiring a Product Manager who has owned checkout or "
            "payments flows and driven measurable conversion improvements."
        ),
        # Every Product Manager candidate whose resume contains a bullet
        # literally about checkout, payments, or the purchase flow.
        "relevant": {
            "C001", "C002", "C003", "C004", "C006", "C007",
            "C009", "C010", "C011", "C012", "C013", "C014",
        },
        # Excluded: C008 (growth/KPI/pricing only) and C005 (UX research,
        # reporting, discovery interviews, pricing experiments only) --
        # neither has a checkout/payments/purchase-flow bullet.
    },
    "SWE microservices/CI-CD": {
        "query": (
            "Looking for a Software Engineer with hands-on experience "
            "decomposing monolithic systems into microservices and "
            "building CI/CD automation."
        ),
        "relevant": {
            "C015", "C016", "C017", "C018", "C019",
            "C020", "C021", "C022", "C023", "C024",
        },
        # All 10 synthetic SE candidates happen to have a microservices-
        # or CI/CD-related bullet -- noted plainly, not hidden, even
        # though it makes this JD a weaker precision test than the other
        # two (see README limitations).
    },
    "Support self-serve docs": {
        "query": (
            "Seeking a Customer Support specialist skilled at building "
            "self-serve documentation and knowledge base content to "
            "reduce repeat tickets."
        ),
        "relevant": {
            "C025", "C026", "C028", "C029", "C030", "C031", "C032",
        },
        # Excluded: C027 (ticket resolution, billing escalation, new-hire
        # training -- no self-serve doc / FAQ / knowledge base bullet).
    },
    "Data Analyst SQL/Python": {
        "query": (
            "We're hiring a Data Analyst fluent in SQL and Python, with "
            "hands-on A/B testing and statistics experience, to support "
            "experimentation and churn analysis."
        ),
        # NOTE on how this label differs from the other three: the 6
        # synthetic Data Analyst resumes share an Experience bullet pool
        # (churn models, automated reporting, funnel instrumentation all
        # appear across nearly every candidate), so Experience content
        # alone doesn't separate them the way it does for PM/Support.
        # Skills DOES split them into two real tool stacks -- SQL/Python/
        # Tableau/A-B Testing/Statistics vs. R/dbt/Looker/Experimentation/
        # Data Modeling -- and the JD's explicit "SQL and Python" ask is a
        # genuine recruiter filter on that stack. Relevant = the SQL/Python
        # cluster.
        "relevant": {"C047", "C050", "C051", "C052"},
        # Excluded: C048, C049 (same job function, same-shaped Experience
        # bullets, but their listed Skills are R/dbt/Looker -- not SQL/
        # Python -- so they don't match this JD's explicit tool-stack ask).
    },
    "Product Designer Figma/design-systems": {
        "query": (
            "We're hiring a Product Designer skilled in Figma with a "
            "strong design-systems background and WCAG accessibility "
            "expertise, who has led usability testing and prototyping."
        ),
        # Same situation as the Data Analyst set above: all 6 synthetic
        # Product Designer resumes share a checkout/purchase-flow redesign
        # bullet pool in Experience, so Skills is again the real
        # differentiator -- Figma/Design Systems/Usability Testing/
        # Prototyping/Accessibility vs. Sketch/User Research/Interaction
        # Design/Wireframing. This is the tightest precision test in the
        # eval set (2 of 6 relevant), intentionally, to balance out the
        # SWE set where all 10 match.
        "relevant": {"C043", "C044"},
        # Excluded: C041, C042, C045, C046 (Sketch/User Research/
        # Interaction Design/Wireframing stack -- no Figma, design
        # systems, or WCAG accessibility listed).
    },
}


def precision_recall_at_k(retrieved_ids, relevant_ids, k=10):
    top_k = retrieved_ids[:k]
    hits = len(set(top_k) & relevant_ids)
    precision = hits / k if k > 0 else 0.0
    recall = hits / len(relevant_ids) if relevant_ids else 0.0
    return precision, recall, hits


def run_evaluation():
    print("=" * 70)
    print("EVALUATION: precision@10 / recall@10 vs hand-labeled relevance")
    print("=" * 70)

    all_metrics = []
    for label, spec in GROUND_TRUTH.items():
        result = search(spec["query"], top_k=10)
        retrieved_ids = [r["candidate_id"] for r in result["results"]]
        precision, recall, hits = precision_recall_at_k(retrieved_ids, spec["relevant"])
        # When fewer than 10 candidates in the whole pool are actually
        # relevant, precision@10 can never reach 1.0 even with a perfect
        # search -- the other (10 - n_relevant) slots are necessarily
        # filled by non-matches. Surfacing that ceiling alongside the raw
        # number keeps a low precision@10 from being misread as a quality
        # regression when it's really a metric artifact of a small
        # ground-truth set (see the Data Analyst / Product Designer JDs).
        max_precision = min(len(spec["relevant"]), 10) / 10
        all_metrics.append({"label": label, "precision": precision, "recall": recall,
                             "hits": hits, "n_relevant": len(spec["relevant"]),
                             "n_retrieved": len(retrieved_ids),
                             "max_achievable_precision_at_10": max_precision})
        print(f"\n[{label}]")
        print(f"  Query: {spec['query']}")
        print(f"  Relevant in pool: {len(spec['relevant'])}  |  Retrieved (top 10): {len(retrieved_ids)}")
        print(f"  Hits: {hits}")
        print(f"  precision@10 = {precision:.2f}   recall@10 = {recall:.2f}"
              + (f"   (ceiling: {max_precision:.2f}, only {len(spec['relevant'])} relevant exist)"
                 if max_precision < 1.0 else ""))

    avg_p = sum(m["precision"] for m in all_metrics) / len(all_metrics)
    avg_r = sum(m["recall"] for m in all_metrics) / len(all_metrics)
    avg_max_p = sum(m["max_achievable_precision_at_10"] for m in all_metrics) / len(all_metrics)
    print("\n" + "-" * 70)
    print(f"AVERAGE across {len(all_metrics)} queries: "
          f"precision@10 = {avg_p:.2f}   recall@10 = {avg_r:.2f}")
    print(f"AVERAGE max-achievable precision@10 (pool-size ceiling): {avg_max_p:.2f}")
    print("-" * 70)
    return all_metrics, avg_p, avg_r


def baseline_manual_search(candidate_ids_picked, seconds_elapsed, relevant_ids):
    """
    STUB for a real human timed baseline. Not run in this environment
    (needs an actual recruiter with a stopwatch), but wired to the same
    scoring function so a real run just calls this with:
      - candidate_ids_picked: the list of candidate_ids the human shortlisted
      - seconds_elapsed: how long it took them
      - relevant_ids: the same ground-truth set used above
    and gets directly comparable precision/recall + a time number to put
    next to the search tool's near-instant runtime.
    """
    precision, recall, hits = precision_recall_at_k(candidate_ids_picked, relevant_ids,
                                                      k=len(candidate_ids_picked))
    return {
        "precision": precision,
        "recall": recall,
        "hits": hits,
        "seconds_elapsed": seconds_elapsed,
    }


if __name__ == "__main__":
    t0 = time.time()
    metrics, avg_p, avg_r = run_evaluation()
    elapsed = time.time() - t0
    avg_max_p = sum(m["max_achievable_precision_at_10"] for m in metrics) / len(metrics)
    print(f"\n(Search-tool runtime for all {len(metrics)} queries: {elapsed:.2f}s -- "
          f"this is the number a real manual baseline, once run, would be compared against.)")

    with open("data/eval_results.json", "w") as f:
        json.dump({"metrics": metrics, "avg_precision_at_10": avg_p,
                    "avg_recall_at_10": avg_r,
                    "avg_max_achievable_precision_at_10": avg_max_p,
                    "search_runtime_seconds": elapsed}, f, indent=2)
    print("Wrote data/eval_results.json")
