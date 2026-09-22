"""
ingest.py
---------
Parses raw resume text into SECTION-LEVEL chunks and writes them to
data/chunks.json. Also writes a name-redacted copy of each resume
(data/redacted_resumes.json) used later by fairness_check.py.

PM-relevant "why" (chunking strategy):
  - Whole-resume chunks are too coarse: a single "why matched" quote for a
    5-job resume is either huge or meaningless, and a candidate's best
    match (one bullet about payments) gets diluted by everything else on
    the page when we embed the whole document as one vector.
  - Sentence-level chunks are too fine: they lose context (a bullet like
    "reduced drop-off by 18%" is meaningless without knowing it was the
    checkout project) and multiply the index size for little benefit.
  - Section-level (one chunk per Experience entry, one for Skills, one for
    Education) is the sweet spot for a resume: each chunk is a coherent,
    quotable unit that maps to something a recruiter actually reads as one
    idea ("this job", "these skills").
"""

import json
import os
import re

IN_PATH = "data/resumes.json"
CHUNKS_PATH = "data/chunks.json"
REDACTED_PATH = "data/redacted_resumes.json"


def split_sections(raw_text):
    """Split a resume into its top-level sections by the section headers
    we control in generate_resumes.py (Experience / Skills / Education)."""
    pattern = r"\n(Experience|Skills|Education)\n"
    parts = re.split(pattern, raw_text)
    # parts[0] is the header (name + role line); then alternating (section_name, body)
    sections = {}
    header = parts[0].strip()
    for i in range(1, len(parts), 2):
        name = parts[i]
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections[name] = body
    return header, sections


def split_experience_entries(experience_body):
    """Each job in Experience starts with a "Title, Company (dates)" line
    followed by "- bullet" lines. Split into one chunk per job entry so a
    citation can point at a single role, not the whole career history."""
    entries = []
    current = []
    for line in experience_body.split("\n"):
        if line.strip() == "":
            continue
        if not line.startswith("-"):
            # new job entry begins
            if current:
                entries.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        entries.append("\n".join(current))
    return entries


def redact_name(raw_text, full_name):
    """Replace the candidate's name (and first/last name individually,
    since resumes sometimes repeat just the first name) with [CANDIDATE]
    for the fairness smoke test, which reruns search on name-only edits."""
    redacted = raw_text.replace(full_name, "[CANDIDATE]")
    first, last = full_name.split(" ", 1)
    redacted = redacted.replace(first, "[CANDIDATE]").replace(last, "[CANDIDATE]")
    return redacted


def main():
    with open(IN_PATH) as f:
        resumes = json.load(f)

    all_chunks = []
    redacted_resumes = []
    chunk_id = 0

    for r in resumes:
        header, sections = split_sections(r["raw_text"])

        # Experience -> one chunk per job entry
        if "Experience" in sections:
            for entry in split_experience_entries(sections["Experience"]):
                chunk_id += 1
                all_chunks.append({
                    "chunk_id": f"CH{chunk_id:04d}",
                    "candidate_id": r["candidate_id"],
                    "section_type": "Experience",
                    "text": entry,
                    "date_received": r["date_received"],
                })

        # Skills -> one chunk (it's already a compact, dense line)
        if "Skills" in sections and sections["Skills"].strip():
            chunk_id += 1
            all_chunks.append({
                "chunk_id": f"CH{chunk_id:04d}",
                "candidate_id": r["candidate_id"],
                "section_type": "Skills",
                "text": sections["Skills"].strip(),
                "date_received": r["date_received"],
            })

        # Education -> one chunk
        if "Education" in sections and sections["Education"].strip():
            chunk_id += 1
            all_chunks.append({
                "chunk_id": f"CH{chunk_id:04d}",
                "candidate_id": r["candidate_id"],
                "section_type": "Education",
                "text": sections["Education"].strip(),
                "date_received": r["date_received"],
            })

        redacted_resumes.append({
            "candidate_id": r["candidate_id"],
            "redacted_text": redact_name(r["raw_text"], r["full_name"]),
        })

    os.makedirs("data", exist_ok=True)
    with open(CHUNKS_PATH, "w") as f:
        json.dump(all_chunks, f, indent=2)
    with open(REDACTED_PATH, "w") as f:
        json.dump(redacted_resumes, f, indent=2)

    print(f"Ingested {len(resumes)} resumes -> {len(all_chunks)} chunks -> {CHUNKS_PATH}")
    from collections import Counter
    print("Chunk types:", Counter(c["section_type"] for c in all_chunks))
    print(f"Wrote redacted resumes -> {REDACTED_PATH}")


if __name__ == "__main__":
    main()
