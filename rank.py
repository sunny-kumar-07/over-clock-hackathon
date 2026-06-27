from __future__ import annotations
import argparse
import gzip
import json
import os
import sys
import time
import orjson
import math
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import numpy as np

import config as cfg
import scoring as sc


def _score_single(c: dict, text: str, semantic_score: float, as_of: datetime) -> dict:
    is_honeypot, hp_reasons = sc.check_honeypot(c)
    dq_mult, dq_reasons = sc.check_disqualifiers(c, text)
    mh_score, mh_families = sc.compute_must_have_coverage(c, text)
    nth_bonus = sc.compute_nice_to_have_bonus(c, text)
    profile_quality_bonus = sc.compute_profile_quality_bonus(c, text)
    exp_fit = sc.compute_experience_fit(c)
    role_rel = sc.compute_role_relevance(c)
    loc_score = sc.compute_location_score(c)
    notice_score = sc.compute_notice_period_score(c)
    behavioral_mult = sc.compute_behavioral_multiplier(c, as_of)

    base = sc.compute_composite_base(
        semantic_score=semantic_score,
        must_have_score=mh_score,
        experience_fit=exp_fit,
        role_relevance=role_rel,
        location_score=loc_score,
        notice_score=notice_score,
        nice_to_have_bonus=nth_bonus,
        profile_quality_bonus=profile_quality_bonus,
        must_have_families=mh_families,
    )

    final_score = base * dq_mult * behavioral_mult
    if is_honeypot:
        final_score *= cfg.HONEYPOT_PENALTY_MULTIPLIER

    return {
        "candidate_id": c["candidate_id"],
        "score": final_score,
        "semantic_score": semantic_score,
        "must_have_families": mh_families,
        "role_relevance": role_rel,
        "disqualifier_reasons": dq_reasons,
        "honeypot_reasons": hp_reasons,
        "behavioral_multiplier": behavioral_mult,
        "_candidate": c,
    }


def worker_chunk(args):
    lines_chunk, as_of_str = args
    as_of = datetime.strptime(as_of_str, "%Y-%m-%d")
    
    candidates = []
    for line in lines_chunk:
        if line.strip():
            candidates.append(orjson.loads(line))
            
    if not candidates:
        return [], 0, 0
        
    texts = [sc.candidate_full_text(c) for c in candidates]
    semantic_scores = sc.compute_semantic_scores(texts)
    
    results = []
    honeypot_count = 0
    disqualified_count = 0
    
    for i, c in enumerate(candidates):
        res = _score_single(c, texts[i], semantic_scores[i], as_of)
        results.append(res)
        if res["honeypot_reasons"]:
            honeypot_count += 1
        if res["disqualifier_reasons"]:
            disqualified_count += 1
            
    return results, honeypot_count, disqualified_count



def worker_chunk_by_offset(args):
    import gc
    gc.disable()
    filename, start, end, as_of_str = args
    as_of = datetime.strptime(as_of_str, "%Y-%m-%d")
    
    candidates = []
    with open(filename, 'rb') as f:
        f.seek(start)
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            if line.strip():
                candidates.append(orjson.loads(line))
                
    if not candidates:
        return [], 0, 0
        
    texts = [sc.candidate_full_text(c) for c in candidates]
    semantic_scores = sc.compute_semantic_scores(texts)
    
    results = []
    honeypot_count = 0
    disqualified_count = 0
    
    for i, c in enumerate(candidates):
        res = _score_single(c, texts[i], semantic_scores[i], as_of)
        results.append(res)
        if res["honeypot_reasons"]:
            honeypot_count += 1
        if res["disqualifier_reasons"]:
            disqualified_count += 1
            
    return results, honeypot_count, disqualified_count


def get_file_chunks(filename: str, num_chunks: int) -> list[tuple[int, int]]:
    file_size = os.path.getsize(filename)
    chunk_size = file_size // num_chunks
    chunks = []
    start = 0
    with open(filename, 'rb') as f:
        for i in range(num_chunks):
            f.seek(start + chunk_size)
            f.readline()
            end = f.tell()
            if i == num_chunks - 1 or end >= file_size:
                chunks.append((start, file_size))
                break
            chunks.append((start, end))
            start = end
    return chunks


def score_file_multiprocess(filename: str, as_of: datetime, verbose: bool = True) -> list[dict]:
    t0 = time.time()
    n_cores = mp.cpu_count()
    chunks = get_file_chunks(filename, n_cores)
    
    args_list = [(filename, start, end, as_of.strftime("%Y-%m-%d")) for start, end in chunks]
    
    results = []
    honeypots = 0
    dqs = 0
    
    with mp.Pool(n_cores) as pool:
        for res_chunk, hp, dq in pool.imap_unordered(worker_chunk_by_offset, args_list):
            results.extend(res_chunk)
            honeypots += hp
            dqs += dq
            
    if verbose:
        print(f"[multiprocessing by offset] finished in {time.time()-t0:.1f}s")
        print(f"  honeypots flagged: {honeypots}")
        print(f"  disqualifier(s) fired: {dqs}")
        
    return results


def score_all_multiprocess(lines: list[str], as_of: datetime, verbose: bool = True) -> list[dict]:
    t0 = time.time()
    n_cores = mp.cpu_count()
    # Ensure at least 1 line per chunk
    chunk_size = max(1, math.ceil(len(lines) / n_cores))
    
    chunks = []
    for i in range(0, len(lines), chunk_size):
        chunks.append((lines[i:i + chunk_size], as_of.strftime("%Y-%m-%d")))
        
    results = []
    honeypots = 0
    dqs = 0
    
    with mp.Pool(n_cores) as pool:
        for res_chunk, hp, dq in pool.imap_unordered(worker_chunk, chunks):
            results.extend(res_chunk)
            honeypots += hp
            dqs += dq
            
    if verbose:
        print(f"[multiprocessing] {len(lines)} candidates in {time.time()-t0:.1f}s")
        print(f"  honeypots flagged: {honeypots}")
        print(f"  disqualifier(s) fired: {dqs}")
        
    return results


def score_all(candidates: list[dict], as_of: datetime, verbose: bool = True) -> list[dict]:
    """Sequential scoring for small samples / Streamlit app."""
    t0 = time.time()
    texts = [sc.candidate_full_text(c) for c in candidates]
    semantic_scores = sc.compute_semantic_scores(texts)
    
    results = []
    for i, c in enumerate(candidates):
        res = _score_single(c, texts[i], semantic_scores[i], as_of)
        results.append(res)

    if verbose:
        print(f"[sequential scoring] {len(candidates)} candidates in {time.time()-t0:.1f}s")

    return results


def build_submission_rows(results: list[dict], top_k: int = 100) -> list[dict]:
    """Sorts by score desc, breaks ties by candidate_id ascending (per
    validate_submission.py requirement), takes top_k, generates reasoning
    text only for these (saves compute on the full 100K)."""

    
    # Sort by raw uncapped score first to get true ranking order
    results_sorted = sorted(results, key=lambda r: (-r["score"], r["candidate_id"]))
    top = results_sorted[:top_k]

    # Normalize scores to (0, 1] range while preserving order
    # This keeps internal differentiation but satisfies validator's <= 1.0 requirement
    max_score = top[0]["score"] if top else 1.0
    for r in top:
        r["_normalized_score"] = round(r["score"] / max_score, 4)

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
            "score": r["_normalized_score"],
            "reasoning": reasoning,
        })

    for i in range(1, len(rows)):
        if rows[i]["score"] > rows[i - 1]["score"]:
            rows[i]["score"] = rows[i - 1]["score"]
        elif rows[i]["score"] == rows[i - 1]["score"]:
            # Same rounded score but different candidate_id order
            # would fail the validator's tie-break check.
            # Nudge this row down by the smallest representable
            # 4-decimal step to make it strictly less.
            rows[i]["score"] = round(rows[i - 1]["score"] - 0.0001, 4)

    return rows


def write_xlsx(rows: list[dict], out_path: str):
    from openpyxl import Workbook
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "Ranked Candidates"
    ws.append(["candidate_id", "rank", "score", "reasoning"])
    for row in rows:
        ws.append([row["candidate_id"], row["rank"], round(row["score"], 4), row["reasoning"]])
    wb.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Redrob candidate ranking pipeline")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl (or .jsonl.gz)")
    parser.add_argument("--out", required=True, help="Output XLSX path")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--as-of", default="2026-06-20", help="Reference date for recency scoring (YYYY-MM-DD)")
    args = parser.parse_args()

    start = time.time()
    as_of = datetime.strptime(args.as_of, "%Y-%m-%d")

    print(f"Processing {args.candidates} ...")
    if args.candidates.endswith(".gz"):
        opener = gzip.open
        with opener(args.candidates, "rt", encoding="utf-8") as f:
            lines = f.read().splitlines()
        print(f"Loaded {len(lines)} lines.")
        results = score_all_multiprocess(lines, as_of=as_of, verbose=True)
    else:
        results = score_file_multiprocess(args.candidates, as_of=as_of, verbose=True)
    rows = build_submission_rows(results, top_k=args.top_k)
    write_xlsx(rows, args.out)

    elapsed = time.time() - start
    print(f"\nWrote top {len(rows)} candidates to {args.out}")
    print(f"Total runtime: {elapsed:.1f}s")
    if elapsed > 300:
        print("WARNING: exceeded 5-minute budget.", file=sys.stderr)


if __name__ == "__main__":
    main()
