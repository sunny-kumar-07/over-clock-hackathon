# Candidate Ranker

An AI-recruiter ranking engine for the **Senior AI Engineer — Founding
Team** role. Built for the India Runs Data & AI Challenge.

Given 100,000 candidate profiles, it produces a ranked top-100 shortlist —
fully **offline, CPU-only, no GPU**, in under two minutes.

---

## Quick start

```bash
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./output/team_xxx.csv
```

That's it — one command, no network access required, no GPU. On a 6-core /
16GB laptop this completes in **~90-115 seconds** for the full 100K-candidate
dataset (well under the 5-minute budget).

To validate the output before submitting:

```bash
python validate_submission.py output/team_xxx.csv
```

To explore the ranker interactively (small-sample sandbox demo):

```bash
streamlit run app.py
```

---

## The problem

Traditional keyword filters reject good candidates whose resumes don't use
the "right" buzzwords, and accept bad candidates who stuff their skills list
with trendy keywords regardless of what they actually did. The JD for this
role explicitly calls this out — it asks for **semantic understanding of
actual experience**, not keyword matching, while also requiring the system
to **catch keyword-stuffers, inconsistent/fake profiles, and unavailable
candidates.**

The dataset is built to test exactly this: it contains
[honeypot profiles](#3-honeypotconsistency-check) with impossible internal
inconsistencies, and a realistic mix of strong-but-buzzword-light candidates
next to weak-but-keyword-stuffed ones.

---

## Why this architecture

The single hardest constraint in this challenge is **no network access, no
GPU, 5-minute budget, on 100,000 candidates.** That rules out calling a
hosted LLM per-candidate at ranking time — the obvious "just ask an AI" path
isn't viable here, by design. So the system had to be built as something a
real recruiting platform could actually run in production: fast, explainable,
and cheap per query.

We landed on a **hybrid rule-engine + local-semantic-similarity** approach:

1. Things that are **objective and rule-derivable** from the JD's explicit
   instructions (e.g. "we will not move forward with pure-research-only
   backgrounds") are encoded as deterministic rules — fast, auditable, and
   directly traceable to a line in the job description.
2. Things that require **understanding meaning, not just keywords** (e.g.
   "this person's resume never says RAG but their work history is clearly a
   recommendation system") are handled by **TF-IDF + truncated SVD (LSA)
   semantic similarity** — a classical, fully local, deterministic technique
   that captures conceptual similarity without needing a downloaded neural
   model or any network call.
3. **Behavioral/availability signals** are applied as a separate
   multiplicative layer on top, per the JD's explicit instruction that a
   perfect-on-paper but unreachable candidate should rank lower.

This means every score the system produces can be explained in one sentence
— important both for the `reasoning` column requirement and for being able
to defend the system in a live interview.

---

## Pipeline architecture

```
candidates.jsonl (100K)
        │
        ▼
┌───────────────────────────┐
│  Stage A: Semantic fit     │  TF-IDF + SVD similarity between each
│  (batched, vectorized)     │  candidate's profile text and a JD-derived
│                            │  "ideal candidate" reference paragraph
└───────────────┬───────────┘
                │
                ▼
┌───────────────────────────┐
│  Stage B: Per-candidate    │  • Honeypot/consistency check
│  rule-based scoring        │  • Hard disqualifier rules (JD dealbreakers)
│                            │  • Must-have / nice-to-have skill coverage
│                            │  • Experience / role / location / notice fit
│                            │  • Behavioral availability multiplier
└───────────────┬───────────┘
                │
                ▼
┌───────────────────────────┐
│  Composite score =         │
│  base_score                │   base_score = weighted sum of semantic +
│    × disqualifier_mult     │   skills + experience + role + location
│    × behavioral_mult       │
│  (honeypots → near-zero)   │
└───────────────┬───────────┘
                │
                ▼
   Sort, take top 100, generate
   per-candidate reasoning text,
   write submission CSV
```

### File-by-file

| File | Purpose |
|---|---|
| `config.py` | The JD's requirements encoded as structured data — must-have skill families, nice-to-haves, hard disqualifier rules and their penalty weights, honeypot detection thresholds, location/experience preferences, composite scoring weights, and the "ideal candidate" reference text used for semantic matching. **No logic lives here, only judgment calls** — this is the file to read to understand *what we believe the JD is asking for.* |
| `scoring.py` | All scoring mechanics: honeypot detection, disqualifier rule checks, skill coverage scoring, semantic similarity (TF-IDF/SVD), experience/role/location/notice-period fit, behavioral multiplier, composite score combination, and reasoning text generation. Pure functions, no I/O. |
| `rank.py` | Orchestration — single entry point. Loads candidates, runs the full scoring pipeline, ranks, writes the submission CSV in the required format. |
| `app.py` | Streamlit sandbox demo — runs the exact same pipeline on a small sample interactively, for the hosted-sandbox-link requirement. |
| `validate_submission.py` | Provided by the organizers — validates output format before submission. |

---

## Methodology in detail

### 1. JD interpretation (`config.py`)

We read the job description closely and translated its explicit statements
into structured rules rather than vague scoring. For example, the JD states
several **hard dealbreakers** almost verbatim:

- Entire career at consulting/services firms (TCS, Infosys, Wipro, etc.) →
  penalty
- Pure research background with no production/deployment evidence → heaviest
  penalty (JD: "will not move forward")
- "AI experience" limited to recent LangChain/OpenAI-wrapper work with no
  pre-LLM ML background → penalty
- Senior title with no hands-on coding evidence in 18+ months → penalty
- CV/speech/robotics-only specialization without NLP/IR exposure → penalty
- Job-hopping pattern (short average tenure across multiple roles) → penalty
- 5+ years experience but zero external validation (no OSS/papers/talks)
  found anywhere in the profile → penalty

Each rule applies its own **multiplicative penalty**, not a hard zero/reject
— except honeypots, since the JD's own compute-rules spec frames most of
these as "probably won't move forward" rather than an absolute rule, while
the spec is explicit only about the pure-research case.

We also captured the JD's **must-have signal families** (embeddings/
retrieval, vector DB/hybrid search, strong Python, ranking-evaluation
experience) at the *concept* level — matching synonyms and related terms,
not a single exact tool name, per the JD's own statement: *"we don't care
which model — we care about the operational experience."*

### 2. Semantic fit (`scoring.compute_semantic_scores`)

We concatenate each candidate's headline, summary, job titles, and job
descriptions into one text blob, and compare it against a short "ideal
candidate" paragraph (in `config.IDEAL_CANDIDATE_TEXT`) distilled from the
JD, using **TF-IDF vectorization (unigrams + bigrams) followed by truncated
SVD (Latent Semantic Analysis)** and cosine similarity. This is computed
**once, in batch, across all 100K candidates simultaneously** — a single
vectorized operation rather than per-candidate, which is what keeps the
whole pipeline under 2 minutes.

This step is what lets a candidate whose resume never says "RAG" or
"Pinecone" still score well if their actual job descriptions clearly
describe building a retrieval/ranking/recommendation system — and what
correctly demotes a candidate whose skills list is packed with AI buzzwords
but whose actual job title and work history are unrelated (e.g. a Marketing
Manager with a padded skills section — a deliberate trap pattern we found
while inspecting the real dataset).

### 3. Honeypot/consistency check

While inspecting `candidates.jsonl` directly, we found concrete examples of
internally-impossible profiles, for example:

- A candidate claiming **"expert" proficiency in MLflow, Photoshop, and
  Content Writing — each with 0 months of stated experience.**
- A candidate whose career history sums to **251 months** of work, while
  their profile states only **119 months (~10 years)** of total experience.

We built three detection rules directly from these observed patterns:
1. Expert/advanced proficiency claimed with near-zero `duration_months`
2. Career-history total duration far exceeding stated `years_of_experience`
3. Claimed proficiency level far above the candidate's own tested
   `skill_assessment_scores` for that skill

Any candidate matching these is pushed to a near-zero score. In testing,
this kept the final top-100 shortlist at **0 honeypots**, while flagging a
broader set of profiles (~3,000 of 100K) as inconsistent across the full
pool — we treated this as an acceptable precision/recall tradeoff, since
missing a true honeypot (false negative) risks automatic disqualification
at Stage 3, while an over-cautious flag on a borderline profile just costs
that one candidate a lower rank, not the whole submission.

### 4. Behavioral availability multiplier

Combines `recruiter_response_rate`, recency of `last_active_date`,
`open_to_work_flag`, and `interview_completion_rate` into a single
multiplier (capped between 0.30x and 1.10x) applied on top of the base fit
score — directly implementing the JD's instruction that availability and
reachability matter, not just on-paper fit.

### 5. Reasoning generation

Each of the top-100 candidates gets a reasoning sentence built from their
*actual* deciding factors (semantic score, which must-have skill families
matched, role relevance, behavioral availability) rather than a fixed
template with only the name swapped — disqualified or honeypot-flagged
candidates would never appear in the top 100 in the first place, since
their scores are forced near-zero before ranking.

---

## Results (full 100K-candidate run)

- **Runtime:** ~93-115 seconds end-to-end (budget: 5 minutes)
- **Honeypots in final top-100:** 0 / 100
- **Output:** passes `validate_submission.py` with no errors
- Top-ranked candidates are consistently real ML/AI/Search/Recommendation
  engineers at credible product companies, with a smooth, non-increasing
  score gradient from rank 1 to rank 100

---

## Design tradeoffs & honest limitations

- **TF-IDF/SVD vs. neural embeddings:** we chose classical TF-IDF+SVD over a
  downloaded sentence-transformer model specifically to guarantee zero
  network dependency at ranking time, even for a one-time model download.
  This trades some semantic nuance for total reproducibility and
  zero-network-risk in the judges' re-run environment. A documented upgrade
  path (precomputing and caching neural embeddings ahead of time, loaded —
  not downloaded — at ranking time) is possible but not used in the core
  submission.
- **Disqualifier penalties are multiplicative, not hard rejects** (except
  honeypots) — a judgment call based on the JD's own softer "probably won't
  move forward" language for most rules. This is the single biggest knob to
  revisit if shortlist quality needs adjusting.
- **Honeypot detection trades precision for recall** — we'd rather
  over-flag borderline profiles than let a true honeypot slip into the top
  100.

---

## Team

Built by a 3-person team. See `submission_metadata.yaml` for contact and
compute details, and an honest declaration of how AI tools were used during
development.
