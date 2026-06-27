# ---------------------------------------------------------------------------
# 1. ROLE BASICS
# ---------------------------------------------------------------------------

EXPERIENCE_BAND = (5, 9)          
EXPERIENCE_SOFT_PENALTY_OUTSIDE_BAND = 0.85  
IDEAL_EXPERIENCE_SWEET_SPOT = (6, 8)  

# ---------------------------------------------------------------------------
# 2. MUST-HAVE SIGNALS  
# ---------------------------------------------------------------------------

MUST_HAVE_SKILL_FAMILIES = {
    "embeddings_retrieval": [
        "sentence-transformers", "sentence transformers", "sentence-transformer",
        "sentence-transformers", "all-minilm", "re-ranker", "reranker",
        "cross-encoder", "bi-encoder", "dense vector", "approximate nearest neighbor",
        "ann search", "vector search", "embedding model", "text embedding",
        "openai embeddings", "bge", "e5", "embeddings", "dense retrieval",
        "semantic search", "retrieval", "RAG",
    ],
    "vector_db_hybrid_search": [
        "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
        "elasticsearch", "faiss", "vector database", "hybrid search", "bm25",
    ],
    "python_strong": ["python"],
    "eval_frameworks": [
        "ndcg", "mrr", "map", "a/b testing", "ab testing", "offline evaluation",
        "evaluation framework", "learning to rank", "ranking evaluation",
        "evaluation", "online evaluation", "experiment", "experiment tracking",
        "metric", "metrics", "click-through", "ctr", "precision@", "recall@",
        "hit rate", "mlflow", "wandb", "ablation",
    ],
}

MUST_HAVE_FAMILY_WEIGHTS = {
    "embeddings_retrieval": 0.30,
    "vector_db_hybrid_search": 0.30,
    "python_strong": 0.15,
    "eval_frameworks": 0.25,
}

SKILL_ASSESSMENT_BLEND_WEIGHT = 0.35   

# ---------------------------------------------------------------------------
# 3. NICE-TO-HAVE SIGNALS  
# ---------------------------------------------------------------------------

NICE_TO_HAVE_SKILLS = [
    "lora", "qlora", "peft", "fine-tuning llms", "fine tuning",
    "xgboost", "learning-to-rank", "neural ranking",
    "distributed systems", "large-scale inference", "inference optimization",
    "open source", "open-source",
]
NICE_TO_HAVE_BONUS_CAP = 0.10   

# ---------------------------------------------------------------------------
# 4. HARD DISQUALIFIER RULES 
# ---------------------------------------------------------------------------

CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mindtree",
    "ltimindtree", "l&t infotech",
]


RESEARCH_ONLY_TITLE_MARKERS = [
    "research scientist", "research fellow", "phd researcher",
    "postdoctoral", "academic researcher", "research intern",
]
PRODUCTION_EVIDENCE_MARKERS = [
    "deployed", "production", "shipped", "real users", "scale",
    "live system", "in prod", "rolled out",
]


LANGCHAIN_WRAPPER_MARKERS = ["langchain", "openai api", "gpt wrapper", "chatgpt wrapper"]
PRE_LLM_ML_MARKERS = [
    "scikit-learn", "xgboost", "pytorch", "tensorflow", "recommendation",
    "ranking", "classification model", "regression model", "feature engineering",
]

CV_SPEECH_ROBOTICS_ONLY_MARKERS = [
    "computer vision", "image classification", "object detection",
    "speech recognition", "robotics", "tts", "asr",
]
NLP_IR_MARKERS = [
    "nlp", "natural language processing", "information retrieval", "ir ",
    "text classification", "named entity", "search", "retrieval", "ranking",
]


TITLE_CHASER_MAX_AVG_TENURE_MONTHS = 18
TITLE_CHASER_MIN_JOBS_TO_EVALUATE = 3   
ARCHITECTURE_ONLY_TITLE_MARKERS = ["architect", "tech lead", "engineering manager", "director"]
IC_CODE_EVIDENCE_MARKERS = ["built", "implemented", "wrote", "coded", "developed", "shipped"]
NO_RECENT_CODE_MONTHS_THRESHOLD = 18


EXTERNAL_VALIDATION_MARKERS = ["open source", "open-source", "published", "paper", "talk", "conference", "blog"]
CLOSED_SOURCE_ONLY_YEARS_THRESHOLD = 5
GITHUB_ACTIVITY_VALIDATION_THRESHOLD = 25   


DISQUALIFIER_PENALTIES = {
    "pure_research_no_production": 0.05,      
    "langchain_only_no_pre_llm":    0.15,
    "no_recent_code_18mo":          0.20,
    "consulting_only_career":       0.15,
    "cv_speech_robotics_only":      0.20,
    "title_chaser":                 0.35,
    "framework_enthusiast_only":    0.50,      
    "closed_source_only_5yr":       0.55,
}

# ---------------------------------------------------------------------------
# 5. HONEYPOT / DATA-CONSISTENCY CHECKS 
# ---------------------------------------------------------------------------

HONEYPOT_PENALTY_MULTIPLIER = 0.01   

HONEYPOT_RULES = {
    "expert_skill_near_zero_duration_months": 3,   
    "career_months_vs_yoe_overshoot_months": 24,    
    "skill_assessment_vs_claim_gap": 40,            
    "expert_skill_max_endorsements_for_flag": 1,    
}

# ---------------------------------------------------------------------------
# 6. LOCATION / LOGISTICS 
# ---------------------------------------------------------------------------

PREFERRED_LOCATIONS = ["pune", "noida"]
WELCOME_LOCATIONS = ["hyderabad", "mumbai", "delhi", "ncr", "gurgaon", "gurugram"]
INDIA_COUNTRY_NAMES = ["india"]

LOCATION_SCORE = {
    "preferred": 1.00,    
    "welcome_india": 0.90,    
    "other_india": 0.80,    
    "outside_india": 0.55,    
}

RELOCATION_WILLING_SCORE = 0.92

NOTICE_PERIOD_IDEAL_DAYS = 30
NOTICE_PERIOD_SCORE_SUB_30 = 1.00
NOTICE_PERIOD_SCORE_30 = 0.90
NOTICE_PERIOD_SCORE_OVER_30_PER_EXTRA_30D = 0.85   

# ---------------------------------------------------------------------------
# 7. BEHAVIORAL AVAILABILITY MULTIPLIER 
# ---------------------------------------------------------------------------

BEHAVIORAL_WEIGHTS = {
    "recruiter_response_rate": 0.20,
    "recency_score": 0.20,
    "open_to_work_flag": 0.10,
    "interview_completion_rate": 0.15,
    "offer_acceptance_rate": 0.10,
    "social_proof": 0.10,
    "profile_completeness": 0.08,
    "avg_response_time": 0.07,
}
BEHAVIORAL_MULTIPLIER_RANGE = (0.30, 1.10)

RECENCY_FULL_SCORE_DAYS = 14
RECENCY_ZERO_SCORE_DAYS = 180

OFFER_ACCEPTANCE_NEUTRAL_SCORE = 0.7

SEARCH_APPEARANCE_CAP_30D = 20
SAVED_BY_RECRUITERS_CAP_30D = 10

PROFILE_COMPLETENESS_FULL_SCORE = 80   # score at or above this = full credit

AVG_RESPONSE_FAST_HOURS = 24    # <= 24hrs = full score
AVG_RESPONSE_SLOW_HOURS = 168   # >= 168hrs (1 week) = zero score

VERIFIED_CONTACT_BONUS = 0.03   # small additive bonus, both email AND phone verified

APPLICATIONS_SWEET_SPOT_MIN = 5    # actively searching
APPLICATIONS_SWEET_SPOT_MAX = 15   # not desperate/unfocused
APPLICATIONS_BONUS = 0.02          # small additive bonus for being in sweet spot

# ---------------------------------------------------------------------------
# 8. FINAL COMPOSITE WEIGHTS
# ---------------------------------------------------------------------------

COMPOSITE_WEIGHTS = {
    "semantic_fit":        0.35,
    "must_have_coverage":  0.30,
    "experience_fit":      0.12,
    "role_relevance":      0.15,
    "location_logistics":  0.08,
}

FULL_MUST_HAVE_COVERAGE_BONUS = 0.08

# ---------------------------------------------------------------------------
# 9. "IDEAL CANDIDATE" REFERENCE TEXT — 
# ---------------------------------------------------------------------------

IDEAL_CANDIDATE_TEXT = """
Senior AI engineer with production experience building embeddings-based
retrieval, ranking, and recommendation systems deployed to real users at
meaningful scale. Hands-on with vector databases and hybrid search
infrastructure such as Pinecone, Weaviate, Qdrant, Milvus, OpenSearch,
Elasticsearch, or FAISS combined with BM25 keyword retrieval. Strong Python
engineer who cares about code quality. Experienced designing evaluation
frameworks for ranking systems using NDCG, MRR, MAP, offline-to-online
correlation, and A/B test interpretation. Has shipped at least one
end-to-end search, ranking, or recommendation system at a product company,
not only a services/consulting environment. Comfortable with both deep
technical work on modern ML systems (embeddings, retrieval, ranking, LLMs,
fine-tuning) and scrappy product engineering, shipping working systems
quickly. Has opinions about hybrid vs dense retrieval, offline vs online
evaluation, and when to fine-tune vs prompt an LLM, grounded in real systems
they built.
""".strip()
