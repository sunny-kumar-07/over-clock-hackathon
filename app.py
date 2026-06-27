import json
import time
from datetime import datetime

import pandas as pd
import streamlit as st

import config as cfg
import scoring as sc
from rank import score_all, build_submission_rows

st.set_page_config(page_title="Candidate Ranker — Sandbox", layout="wide")

st.title("Candidate Ranker — Sandbox")
st.markdown(
    "A hybrid rule + semantic-similarity ranking engine for the "
    "**Senior AI Engineer — Founding Team** role. "
    "Runs fully locally — no network calls, no GPU. "
    "[See `config.py`](.) for the JD interpretation driving every score below."
)

with st.expander("How this works (click to expand)", expanded=False):
    st.markdown(
        """
1. **Honeypot check** — flags internally-inconsistent profiles (e.g. "expert" skill
   claimed with 0 months experience, career history that doesn't add up to stated
   years of experience).
2. **Hard disqualifiers** — JD-explicit dealbreakers (consulting-only career,
   pure-research-only background, no recent hands-on code, title-chasing, etc.),
   each applying its own penalty multiplier.
3. **Semantic fit** — TF-IDF + SVD similarity between each candidate's profile text
   and an "ideal candidate" reference description distilled from the JD. Catches
   candidates whose actual work matches even without buzzword-matching skills lists.
4. **Skill / experience / role / location fit** — structured scoring against the
   JD's must-have/nice-to-have skill families, experience band, and location preferences.
5. **Behavioral availability multiplier** — combines recruiter response rate, login
   recency, open-to-work status, and interview completion rate. A perfect-on-paper
   but unreachable candidate gets down-weighted, per the JD's explicit instruction.
        """
    )

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------

st.subheader("1. Load candidates")
st.info(
    "**Demo mode:** Use the bundled 50-candidate sample to see the pipeline "
    "run live in this sandbox. For the full 100K-candidate run, use the command: "
    "`python rank.py --candidates candidates.jsonl --out output/submission.csv` "
    "on your local machine — it completes in ~90 seconds."
)
source = st.radio(
    "Choose a data source",
    ["Use bundled sample_candidates.json (50 candidates)", "Upload your own .json / .jsonl"],
    index=0,
    horizontal=False,
)

candidates = None

if source.startswith("Use bundled"):
    try:
        with open("sample_candidates.json") as f:
            candidates = json.load(f)
        st.success(f"Loaded {len(candidates)} bundled sample candidates.")
    except FileNotFoundError:
        st.error(
            "sample_candidates.json not found alongside app.py. "
            "Place the hackathon's sample_candidates.json in this directory, "
            "or switch to 'Upload your own' below."
        )
else:
    st.warning(
        "⚠️ Streamlit upload limit is 200MB. "
        "For large files (candidates.jsonl is 465MB), use the local command instead."
    )
    uploaded = st.file_uploader("Upload a candidates file", type=["json", "jsonl"])
    if uploaded is not None:
        raw = uploaded.read().decode("utf-8")
        try:
            candidates = json.loads(raw)
            if isinstance(candidates, dict):
                candidates = [candidates]
        except json.JSONDecodeError:
            candidates = [json.loads(line) for line in raw.splitlines() if line.strip()]
        st.success(f"Loaded {len(candidates)} candidates from upload.")

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

if candidates:
    st.subheader("2. Run the ranking pipeline")
    top_k = st.slider("Shortlist size (top K)", min_value=5, max_value=min(100, len(candidates)),
                       value=min(20, len(candidates)))
    as_of_date = st.date_input("Reference date (for recency scoring)", value=datetime(2026, 6, 20))

    if st.button("Run ranking", type="primary"):
        with st.spinner(f"Scoring {len(candidates)} candidates..."):
            t0 = time.time()
            as_of = datetime.combine(as_of_date, datetime.min.time())
            results = score_all(candidates, as_of=as_of, verbose=False)
            rows = build_submission_rows(results, top_k=top_k)
            elapsed = time.time() - t0

        st.success(f"Ranked {len(candidates)} candidates in {elapsed:.2f}s — fully local, no network/GPU.")

        df = pd.DataFrame(rows)
        st.subheader(f"3. Top {len(rows)} ranked shortlist")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download as CSV",
            data=df.to_csv(index=False),
            file_name="ranked_shortlist.csv",
            mime="text/csv",
        )

       
        honeypot_hits = sum(1 for r in rows if "inconsistent" in r["reasoning"].lower())
        st.caption(f"Honeypot/inconsistent profiles in this shortlist: {honeypot_hits} / {len(rows)}")
else:
    st.info("Load a candidate dataset above to run the pipeline.")

st.divider()
st.caption(
    "This sandbox uses the exact same scoring.py / config.py / rank.py as the full "
    "100K-candidate submission pipeline — nothing here is mocked or simplified."
)
