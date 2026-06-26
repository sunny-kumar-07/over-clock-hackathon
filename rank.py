from __future__ import annotations
import argparse
import csv
import gzip
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import config as cfg
import scoring as sc


def load_candidates(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    candidates = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
    return candidates


def score_all(candidates: list[dict], as_of: datetime, verbose: bool = True) -> list[dict]:
    """Runs the full scoring pipeline over all candidates. Returns a list of
    result dicts (one per candidate) with final score + components needed
    for reasoning generation."""

    n = len(candidates)

    # --- Stage A: batch semantic similarity (vectorized, fast) ---
    t0 = time.time()
    texts = [sc.candidate_full_text(c) for c in candidates]
    semantic_scores = sc.compute_semantic_scores(texts)
    if verbose:
        print(f"[semantic similarity] {n} candidates in {time.time()-t0:.1f}s")

    # --- Stage B: per-candidate rule-based scoring ---
    t0 = time.time()
    results = []
    honeypot_count = 0
    disqualified_count = 0

    for i, c in enumerate(candidates):
        is_honeypot, hp_reasons = sc.check_honeypot(c)
        dq_mult, dq_reasons = sc.check_disqualifiers(c)
        mh_score, mh_families = sc.compute_must_have_coverage(c)
        nth_bonus = sc.compute_nice_to_have_bonus(c)
        exp_fit = sc.compute_experience_fit(c)
        role_rel = sc.compute_role_relevance(c)
        loc_score = sc.compute_location_score(c)
        notice_score = sc.compute_notice_period_score(c)
        behavioral_mult = sc.compute_behavioral_multiplier(c, as_of)

        base = sc.compute_composite_base(
            semantic_score=semantic_scores[i],
            must_have_score=mh_score,
            experience_fit=exp_fit,
            role_relevance=role_rel,
            location_score=loc_score,
            notice_score=notice_score,
            nice_to_have_bonus=nth_bonus,
            must_have_families=mh_families,
        )

        final_score = base * dq_mult * behavioral_mult
        if is_honeypot:
            final_score *= cfg.HONEYPOT_PENALTY_MULTIPLIER
            honeypot_count += 1
        if dq_reasons:
            disqualified_count += 1

        results.append({
            "candidate_id": c["candidate_id"],
            "score": final_score,
            "semantic_score": semantic_scores[i],
            "must_have_families": mh_families,
            "role_relevance": role_rel,
            "disqualifier_reasons": dq_reasons,
            "honeypot_reasons": hp_reasons,
            "behavioral_multiplier": behavioral_mult,
            "_candidate": c,
        })

    if verbose:
        print(f"[rule-based scoring] {n} candidates in {time.time()-t0:.1f}s")
        print(f"  honeypots flagged: {honeypot_count}")
        print(f"  disqualifier(s) fired: {disqualified_count}")

    return results


def build_submission_rows(results: list[dict], top_k: int = 100) -> list[dict]:
    """Sorts by score desc, breaks ties by candidate_id ascending (per
    validate_submission.py requirement), takes top_k, generates reasoning
    text only for these (saves compute on the full 100K)."""

    
    for r in results:
        r["_rounded_score"] = round(min(r["score"], 1.0), 4)

    results_sorted = sorted(results, key=lambda r: (-r["_rounded_score"], r["candidate_id"]))
    top = results_sorted[:top_k]

    rows = []
    for rank, r in enumerate(top, start=1):
        reasoning = sc.generate_reasoning(
            candidate=r["_candidate"],
            semantic_score=r["semantic_score"],
            must_have_families=r["must_have_families"],
            role_relevance=r["role_relevance"],
            disqualifier_reasons=r["disqualifier_reasons"],
            honeypot_reasons=r["honeypot_reasons"],
            behavioral_multiplier=r["behavioral_multiplier"],
        )
        rows.append({
            "candidate_id": r["candidate_id"],
            "rank": rank,
            "score": r["_rounded_score"],
            "reasoning": reasoning,
        })

    for i in range(1, len(rows)):
        if rows[i]["score"] > rows[i - 1]["score"]:
            rows[i]["score"] = rows[i - 1]["score"]

    return rows


def write_csv(rows: list[dict], out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for row in rows:
            writer.writerow([row["candidate_id"], row["rank"], f"{row['score']:.4f}", row["reasoning"]])


def main():
    parser = argparse.ArgumentParser(description="Redrob candidate ranking pipeline")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl (or .jsonl.gz)")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--as-of", default="2026-06-20", help="Reference date for recency scoring (YYYY-MM-DD)")
    args = parser.parse_args()

    start = time.time()
    as_of = datetime.strptime(args.as_of, "%Y-%m-%d")

    print(f"Loading candidates from {args.candidates} ...")
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates)} candidates.")

    results = score_all(candidates, as_of=as_of, verbose=True)
    rows = build_submission_rows(results, top_k=args.top_k)
    write_csv(rows, args.out)

    elapsed = time.time() - start
    print(f"\nWrote top {len(rows)} candidates to {args.out}")
    print(f"Total runtime: {elapsed:.1f}s")
    if elapsed > 300:
        print("WARNING: exceeded 5-minute budget.", file=sys.stderr)


if __name__ == "__main__":
    main()