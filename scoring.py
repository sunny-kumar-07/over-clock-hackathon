

from __future__ import annotations
import re
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

import config as cfg


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def candidate_full_text(candidate: dict) -> str:
    """Concatenate all free-text fields used for semantic matching and
    keyword/marker scanning: headline, summary, career history descriptions
    and titles, skill names."""
    p = candidate["profile"]
    parts = [p.get("headline", ""), p.get("summary", ""), p.get("current_title", "")]
    for job in candidate.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    for s in candidate.get("skills", []):
        parts.append(s.get("name", ""))
    return " ".join(parts).lower()


def _any_marker_present(text: str, markers: list[str]) -> bool:
    return any(m.lower() in text for m in markers)


def _count_markers_present(text: str, markers: list[str]) -> int:
    return sum(1 for m in markers if m.lower() in text)


# ---------------------------------------------------------------------------
# 1. HONEYPOT / CONSISTENCY CHECK
# ---------------------------------------------------------------------------

def check_honeypot(candidate: dict) -> tuple[bool, list[str]]:
    """Returns (is_honeypot, reasons). Mirrors the patterns we found by
    inspecting real candidates.jsonl entries (e.g. CAND_0003582, CAND_0007353)."""
    reasons = []

   
    min_months = cfg.HONEYPOT_RULES["expert_skill_near_zero_duration_months"]
    max_endorsements_for_flag = cfg.HONEYPOT_RULES["expert_skill_max_endorsements_for_flag"]
    for s in candidate.get("skills", []):
        if s.get("proficiency") in ("expert", "advanced") and s.get("duration_months", 999) < min_months:
            endorsements = s.get("endorsements", 0)
            if endorsements <= max_endorsements_for_flag:
                reasons.append(f"claims {s['proficiency']} {s['name']} with {s['duration_months']}mo experience and {endorsements} endorsements")
            else:
                reasons.append(f"claims {s['proficiency']} {s['name']} with {s['duration_months']}mo experience")

    
    total_months = sum(j.get("duration_months", 0) for j in candidate.get("career_history", []))
    yoe_months = candidate["profile"].get("years_of_experience", 0) * 12
    overshoot = cfg.HONEYPOT_RULES["career_months_vs_yoe_overshoot_months"]
    if total_months > yoe_months + overshoot:
        reasons.append(f"career history totals {total_months}mo vs stated {yoe_months:.0f}mo experience")

    
    gap_threshold = cfg.HONEYPOT_RULES["skill_assessment_vs_claim_gap"]
    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}
    proficiency_floor = {"expert": 85, "advanced": 70, "intermediate": 50, "beginner": 25}
    for s in candidate.get("skills", []):
        name = s.get("name")
        if name in assessments:
            expected_floor = proficiency_floor.get(s.get("proficiency"), 0)
            tested = assessments[name]
            if expected_floor - tested > gap_threshold:
                reasons.append(f"claims {s['proficiency']} {name} but tested assessment score is only {tested}")

    return (len(reasons) > 0, reasons)


# ---------------------------------------------------------------------------
# 2. HARD DISQUALIFIER RULES
# ---------------------------------------------------------------------------

def check_disqualifiers(candidate: dict) -> tuple[float, list[str]]:
    """Returns (combined_penalty_multiplier, reasons_fired). Multiple rules
    can fire; penalties multiply together (compounding penalty)."""
    text = candidate_full_text(candidate)
    career = candidate.get("career_history", [])
    profile = candidate["profile"]
    reasons = []
    multiplier = 1.0

    # --- Pure research, no production evidence ---
    if career:
        research_titles = sum(1 for j in career if _any_marker_present(j.get("title", "").lower(), cfg.RESEARCH_ONLY_TITLE_MARKERS))
        has_production_evidence = _any_marker_present(text, cfg.PRODUCTION_EVIDENCE_MARKERS)
        if research_titles == len(career) and not has_production_evidence:
            multiplier *= cfg.DISQUALIFIER_PENALTIES["pure_research_no_production"]
            reasons.append("entire career in pure-research roles, no production deployment evidence")

    # --- LangChain-wrapper-only AI experience, <12mo, no pre-LLM ML background ---
    has_langchain_only = _any_marker_present(text, cfg.LANGCHAIN_WRAPPER_MARKERS)
    has_pre_llm_ml = _any_marker_present(text, cfg.PRE_LLM_ML_MARKERS)
    if has_langchain_only and not has_pre_llm_ml:
        # check recency: any AI-relevant job under 12 months that's the only AI exposure
        recent_ai_only = False
        for j in career:
            if j.get("duration_months", 999) < 12 and _any_marker_present(j.get("description", "").lower(), cfg.LANGCHAIN_WRAPPER_MARKERS):
                recent_ai_only = True
        if recent_ai_only:
            multiplier *= cfg.DISQUALIFIER_PENALTIES["langchain_only_no_pre_llm"]
            reasons.append("AI experience limited to recent LangChain/OpenAI wrapper work, no pre-LLM ML background")

    # --- No hands-on code in 18+ months (architecture/lead title, no IC evidence) ---
    if career:
        current_job = next((j for j in career if j.get("is_current")), career[0])
        title_lower = current_job.get("title", "").lower()
        if _any_marker_present(title_lower, cfg.ARCHITECTURE_ONLY_TITLE_MARKERS):
            tenure = current_job.get("duration_months", 0)
            has_ic_evidence = _any_marker_present(current_job.get("description", "").lower(), cfg.IC_CODE_EVIDENCE_MARKERS)
            if tenure >= cfg.NO_RECENT_CODE_MONTHS_THRESHOLD and not has_ic_evidence:
                multiplier *= cfg.DISQUALIFIER_PENALTIES["no_recent_code_18mo"]
                reasons.append(f"in '{current_job.get('title')}' role for {tenure}mo with no hands-on coding evidence")

    # --- Consulting-only career ---
    if career:
        consulting_jobs = sum(1 for j in career if _any_marker_present(j.get("company", "").lower(), cfg.CONSULTING_FIRMS))
        if consulting_jobs == len(career):
            multiplier *= cfg.DISQUALIFIER_PENALTIES["consulting_only_career"]
            reasons.append("entire career at consulting/services firms, no product-company experience")

    # --- CV/speech/robotics-only without NLP/IR ---
    has_cv_speech = _any_marker_present(text, cfg.CV_SPEECH_ROBOTICS_ONLY_MARKERS)
    has_nlp_ir = _any_marker_present(text, cfg.NLP_IR_MARKERS)
    if has_cv_speech and not has_nlp_ir:
        multiplier *= cfg.DISQUALIFIER_PENALTIES["cv_speech_robotics_only"]
        reasons.append("background is CV/speech/robotics without NLP/IR exposure")

    # --- Title-chaser: short avg tenure across multiple jobs ---
    if len(career) >= cfg.TITLE_CHASER_MIN_JOBS_TO_EVALUATE:
        avg_tenure = np.mean([j.get("duration_months", 0) for j in career])
        if avg_tenure < cfg.TITLE_CHASER_MAX_AVG_TENURE_MONTHS:
            multiplier *= cfg.DISQUALIFIER_PENALTIES["title_chaser"]
            reasons.append(f"average tenure {avg_tenure:.0f}mo across {len(career)} roles suggests title-chasing")

    # --- Closed-source only, 5+ years, no external validation ---
    # Use github_activity_score (structured signal) as the primary evidence,
    # since text markers like "github"/"open source" almost never appear
    # literally in profile descriptions even for genuinely active OSS
    # contributors (verified against real data). Text markers are kept as a
    # secondary OR-condition for the rare cases where someone does write it out.
    if profile.get("years_of_experience", 0) >= cfg.CLOSED_SOURCE_ONLY_YEARS_THRESHOLD:
        github_score = candidate.get("redrob_signals", {}).get("github_activity_score", -1) or -1
        has_real_github_activity = github_score >= cfg.GITHUB_ACTIVITY_VALIDATION_THRESHOLD
        has_external_validation = has_real_github_activity or _any_marker_present(text, cfg.EXTERNAL_VALIDATION_MARKERS)
        if not has_external_validation:
            multiplier *= cfg.DISQUALIFIER_PENALTIES["closed_source_only_5yr"]
            reasons.append(f"{profile.get('years_of_experience')}yrs experience with no external validation found (GitHub activity score {github_score})")

    return (multiplier, reasons)


# ---------------------------------------------------------------------------
# 3. MUST-HAVE / NICE-TO-HAVE SKILL COVERAGE
# ---------------------------------------------------------------------------

def compute_must_have_coverage(candidate: dict) -> tuple[float, list[str]]:
    """Weighted coverage across the 4 must-have skill families. Concept-level
    matching against full candidate text (skills + descriptions), not just
    the skills list, per JD: 'we don't care which model — operational
    experience does'.

    When the candidate has a tested skill_assessment_scores entry for one of
    the specific skill names in a matched family, blend in that tested score
    -- rewards verified competence over a profile that merely mentions the
    right words (catches keyword-stuffers within a matched family, not just
    across families)."""
    text = candidate_full_text(candidate)
    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}
    score = 0.0
    matched_families = []
    for family, markers in cfg.MUST_HAVE_SKILL_FAMILIES.items():
        weight = cfg.MUST_HAVE_FAMILY_WEIGHTS[family]
        if _any_marker_present(text, markers):
            matched_families.append(family)
            # look for a tested assessment score on any skill name belonging
            # to this family (case-insensitive match against assessments keys)
            tested_scores = [v for k, v in assessments.items() if k.lower() in [m.lower() for m in markers] or any(m.lower() in k.lower() for m in markers)]
            if tested_scores:
                competence = max(tested_scores) / 100.0  # best tested evidence for the family
                blend = cfg.SKILL_ASSESSMENT_BLEND_WEIGHT
                family_credit = weight * ((1 - blend) * 1.0 + blend * competence)
            else:
                family_credit = weight  # no tested data available, full text-match credit
            score += family_credit
    return (score, matched_families)


def compute_nice_to_have_bonus(candidate: dict) -> float:
    text = candidate_full_text(candidate)
    hits = _count_markers_present(text, cfg.NICE_TO_HAVE_SKILLS)
    bonus = min(hits / max(len(cfg.NICE_TO_HAVE_SKILLS), 1), 1.0) * cfg.NICE_TO_HAVE_BONUS_CAP
    return bonus


# ---------------------------------------------------------------------------
# 4. EXPERIENCE FIT
# ---------------------------------------------------------------------------

def compute_experience_fit(candidate: dict) -> float:
    yoe = candidate["profile"].get("years_of_experience", 0)
    lo, hi = cfg.EXPERIENCE_BAND
    sweet_lo, sweet_hi = cfg.IDEAL_EXPERIENCE_SWEET_SPOT
    if sweet_lo <= yoe <= sweet_hi:
        return 1.0
    if lo <= yoe <= hi:
        return 0.90
    years_outside = min(abs(yoe - lo), abs(yoe - hi))
    return max(0.0, cfg.EXPERIENCE_SOFT_PENALTY_OUTSIDE_BAND ** years_outside)


# ---------------------------------------------------------------------------
# 5. ROLE RELEVANCE (title actually reflects the work, not just skills list)
# ---------------------------------------------------------------------------

ROLE_RELEVANT_TITLE_MARKERS = [
    "ml engineer", "machine learning", "ai engineer", "applied scientist",
    "data scientist", "research engineer", "recommendation", "search engineer",
    "ranking engineer", "nlp engineer", "ai research",
]
ROLE_IRRELEVANT_TITLE_MARKERS = [
    "marketing", "sales", "accountant", "hr ", "human resources", "customer support",
    "operations manager", "business analyst", "civil engineer", "content writer",
]


def compute_role_relevance(candidate: dict) -> float:
    title = candidate["profile"].get("current_title", "").lower()
    if _any_marker_present(title, ROLE_RELEVANT_TITLE_MARKERS):
        return 1.0
    if _any_marker_present(title, ROLE_IRRELEVANT_TITLE_MARKERS):
        return 0.15  # JD: keyword-stuffed Marketing Manager is NOT a fit, regardless of skills
    # ambiguous engineering title (e.g. "Backend Engineer", "Software Engineer")
    return 0.55


# ---------------------------------------------------------------------------
# 6. LOCATION / NOTICE PERIOD
# ---------------------------------------------------------------------------

def compute_location_score(candidate: dict) -> float:
    loc = candidate["profile"].get("location", "").lower()
    country = candidate["profile"].get("country", "").lower()
    willing_to_relocate = candidate.get("redrob_signals", {}).get("willing_to_relocate", False)

    if country and country not in cfg.INDIA_COUNTRY_NAMES:
        return cfg.LOCATION_SCORE["outside_india"]
    if _any_marker_present(loc, cfg.PREFERRED_LOCATIONS):
        return cfg.LOCATION_SCORE["preferred"]
    if _any_marker_present(loc, cfg.WELCOME_LOCATIONS):
        return cfg.LOCATION_SCORE["welcome_india"]
    # elsewhere in India: if willing to relocate, treat almost as well as welcome cities
    if willing_to_relocate:
        return cfg.RELOCATION_WILLING_SCORE
    return cfg.LOCATION_SCORE["other_india"]


def compute_notice_period_score(candidate: dict) -> float:
    days = candidate.get("redrob_signals", {}).get("notice_period_days")
    if days is None:
        return 0.80  # unknown, neutral-ish
    if days < cfg.NOTICE_PERIOD_IDEAL_DAYS:
        return cfg.NOTICE_PERIOD_SCORE_SUB_30
    if days == cfg.NOTICE_PERIOD_IDEAL_DAYS:
        return cfg.NOTICE_PERIOD_SCORE_30
    extra_periods = (days - cfg.NOTICE_PERIOD_IDEAL_DAYS) / 30.0
    return max(0.3, cfg.NOTICE_PERIOD_SCORE_OVER_30_PER_EXTRA_30D ** extra_periods)


# ---------------------------------------------------------------------------
# 7. BEHAVIORAL AVAILABILITY MULTIPLIER
# ---------------------------------------------------------------------------

def _recency_score(last_active_date_str: str, as_of: datetime) -> float:
    try:
        last_active = datetime.strptime(last_active_date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return 0.5
    days_inactive = (as_of - last_active).days
    if days_inactive <= cfg.RECENCY_FULL_SCORE_DAYS:
        return 1.0
    if days_inactive >= cfg.RECENCY_ZERO_SCORE_DAYS:
        return 0.0
    span = cfg.RECENCY_ZERO_SCORE_DAYS - cfg.RECENCY_FULL_SCORE_DAYS
    return max(0.0, 1.0 - (days_inactive - cfg.RECENCY_FULL_SCORE_DAYS) / span)


def compute_behavioral_multiplier(candidate: dict, as_of: datetime) -> float:
    sig = candidate.get("redrob_signals", {})
    w = cfg.BEHAVIORAL_WEIGHTS

    response_rate = sig.get("recruiter_response_rate", 0.0) or 0.0
    recency = _recency_score(sig.get("last_active_date"), as_of)
    open_flag = 1.0 if sig.get("open_to_work_flag") else 0.0
    interview_completion = sig.get("interview_completion_rate", 0.0) or 0.0

    # offer_acceptance_rate: -1 means no prior offers (no history = neutral, not bad)
    raw_oar = sig.get("offer_acceptance_rate", -1)
    if raw_oar is None or raw_oar == -1:
        offer_acceptance = cfg.OFFER_ACCEPTANCE_NEUTRAL_SCORE
    else:
        offer_acceptance = float(raw_oar)

    # social proof: normalize raw counts against caps so one-person outliers
    # don't dominate; clamp to [0,1]
    search_appear = min(sig.get("search_appearance_30d", 0) or 0, cfg.SEARCH_APPEARANCE_CAP_30D) / cfg.SEARCH_APPEARANCE_CAP_30D
    saved_by = min(sig.get("saved_by_recruiters_30d", 0) or 0, cfg.SAVED_BY_RECRUITERS_CAP_30D) / cfg.SAVED_BY_RECRUITERS_CAP_30D
    social_proof = (search_appear + saved_by) / 2.0

    raw = (
        w["recruiter_response_rate"] * response_rate
        + w["recency_score"] * recency
        + w["open_to_work_flag"] * open_flag
        + w["interview_completion_rate"] * interview_completion
        + w["offer_acceptance_rate"] * offer_acceptance
        + w["social_proof"] * social_proof
    )
    lo, hi = cfg.BEHAVIORAL_MULTIPLIER_RANGE
    return lo + raw * (hi - lo)


# ---------------------------------------------------------------------------
# 8. BATCH SEMANTIC SIMILARITY (TF-IDF + SVD, vectorized across all candidates)
# ---------------------------------------------------------------------------

def compute_semantic_scores(candidate_texts: list[str]) -> np.ndarray:
    """Returns an array of cosine similarity scores (0-1) between each
    candidate's text and the ideal-candidate reference text, using TF-IDF +
    truncated SVD (LSA). Fully local, deterministic, no network/GPU."""
    corpus = candidate_texts + [cfg.IDEAL_CANDIDATE_TEXT.lower()]

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    n_components = min(200, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(tfidf_matrix)

    ideal_vec = reduced[-1].reshape(1, -1)
    candidate_vecs = reduced[:-1]

    sims = cosine_similarity(candidate_vecs, ideal_vec).flatten()
    # normalize to 0-1 (cosine sim on SVD space can go slightly negative)
    sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-9)
    return sims


# ---------------------------------------------------------------------------
# 9. REASONING TEXT GENERATOR (per-candidate, built from real values, not a
#    fixed template — varies based on which factors actually drove the score)
# ---------------------------------------------------------------------------

def generate_reasoning(candidate: dict, semantic_score: float, must_have_families: list[str],
                        role_relevance: float, disqualifier_reasons: list[str],
                        honeypot_reasons: list[str], behavioral_multiplier: float) -> str:
    p = candidate["profile"]
    sig = candidate.get("redrob_signals", {})
    yoe = p.get("years_of_experience")
    title = p.get("current_title")
    company = p.get("current_company")
    loc = p.get("location", "")
    notice = sig.get("notice_period_days")
    github = sig.get("github_activity_score", -1)
    willing_relocate = sig.get("willing_to_relocate", False)

    if honeypot_reasons:
        return f"Flagged as inconsistent/low-trust profile: {honeypot_reasons[0]}."

    if disqualifier_reasons:
        return f"{title} ({yoe}yrs, {company}) — does not fit: {disqualifier_reasons[0]}."

    family_label = {
        "embeddings_retrieval": "embeddings/retrieval",
        "vector_db_hybrid_search": "vector DB/hybrid search",
        "python_strong": "Python",
        "eval_frameworks": "ranking evaluation",
    }
    matched = [family_label[f] for f in must_have_families]
    skills_str = ", ".join(matched) if matched else "limited must-have signal"

    # availability phrasing from real signal values
    rr = sig.get("recruiter_response_rate", 0) or 0
    if behavioral_multiplier > 0.85:
        avail = f"highly responsive (response rate {rr:.0%})"
    elif behavioral_multiplier > 0.55:
        avail = f"moderately reachable (response rate {rr:.0%})"
    else:
        avail = f"low engagement (response rate {rr:.0%})"

    # notice period phrasing
    if notice is not None:
        notice_str = f"{notice}-day notice"
    else:
        notice_str = "notice period unknown"

    # github phrasing
    if github and github >= 25:
        github_str = f"GitHub activity {github:.0f}/100"
    elif github == -1:
        github_str = "no GitHub linked"
    else:
        github_str = f"low GitHub activity ({github:.0f}/100)"

    # location
    loc_note = f"{loc}"
    if willing_relocate and loc:
        loc_note += " (open to relocate)"

    return (
        f"{title}, {yoe}yrs, {company} ({loc_note}): covers {skills_str}; "
        f"semantic fit {semantic_score:.2f}; {avail}; {notice_str}; {github_str}."
    )


# ---------------------------------------------------------------------------
# 10. COMPOSITE SCORE
# ---------------------------------------------------------------------------

def compute_composite_base(semantic_score: float, must_have_score: float,
                            experience_fit: float, role_relevance: float,
                            location_score: float, notice_score: float,
                            nice_to_have_bonus: float,
                            must_have_families: list[str]) -> float:
    w = cfg.COMPOSITE_WEIGHTS
    logistics = 0.6 * location_score + 0.4 * notice_score
    base = (
        w["semantic_fit"] * semantic_score
        + w["must_have_coverage"] * must_have_score
        + w["experience_fit"] * experience_fit
        + w["role_relevance"] * role_relevance
        + w["location_logistics"] * logistics
    )
   
    if len(must_have_families) == len(cfg.MUST_HAVE_SKILL_FAMILIES):
        base += cfg.FULL_MUST_HAVE_COVERAGE_BONUS
    return min(1.0, base + nice_to_have_bonus)