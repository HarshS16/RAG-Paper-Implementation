"""Run every query once and cache the result.

Retrieval and generation do not depend on the abstention threshold: the same
query retrieves the same passages and, when it is answered at all, produces
the same answer at every tau. So the expensive work is done exactly once per
query and the whole threshold sweep is derived from the cached confidence
scores afterwards (see select_threshold.py).

Besides being roughly k-times cheaper in API calls, this makes the sweep
exact. Re-generating and re-judging at every threshold, as the earlier version
did, let judge nondeterminism move the faithfulness column independently of
the threshold being studied.

The run is resumable: queries already present in results/per_query.json are
skipped, so hitting a rate limit does not cost the whole run.
"""

import json
import os

import config
from evaluate import (
    compute_hit_at_k,
    compute_precision_at_k,
    compute_reciprocal_rank,
    evaluate_answer,
    parse_scores,
)
from generator import generate_answer, generate_baseline_answer
from retriever import build_pipeline, retrieve


def load_queries():
    with open(config.QUERIES_FILE, encoding="utf-8") as f:
        return json.load(f)


def load_cache():
    if os.path.exists(config.PER_QUERY_FILE):
        with open(config.PER_QUERY_FILE, encoding="utf-8") as f:
            return {r["id"]: r for r in json.load(f)}
    return {}


def save_cache(cache, queries):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    ordered = [cache[q["id"]] for q in queries if q["id"] in cache]
    with open(config.PER_QUERY_FILE, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2, ensure_ascii=False)


def run():
    queries = load_queries()
    cache = load_cache()
    model, index, chunks = build_pipeline()

    print(f"{len(queries)} queries | {len(chunks)} chunks | k={config.TOP_K}")
    print(f"generator={config.GENERATOR_MODEL}  judge={config.JUDGE_MODEL}\n")

    for n, q in enumerate(queries, start=1):
        if q["id"] in cache:
            print(f"[{n:2}/{len(queries)}] {q['id']} cached, skipping")
            continue

        print(f"[{n:2}/{len(queries)}] {q['id']} {q['question'][:58]}")

        retrieved = retrieve(q["question"], model, index, chunks, k=config.TOP_K)
        texts = [r["text"] for r in retrieved]
        confidence = sum(r["score"] for r in retrieved) / len(retrieved) if retrieved else 0.0
        context = "\n".join(texts)
        print(f"        confidence={confidence:.3f}")

        rag_answer = generate_answer(texts, q["question"])
        rag_scores = parse_scores(evaluate_answer(q["question"], context, rag_answer))

        # The baseline sees no context, but is judged against the same context
        # the RAG system was given. That is what makes the faithfulness columns
        # comparable: it asks whether the answer is supported by the passage,
        # not whether it happens to be true in general.
        baseline_answer = generate_baseline_answer(q["question"])
        baseline_scores = parse_scores(
            evaluate_answer(q["question"], context, baseline_answer)
        )

        keywords = q.get("keywords") or []
        cache[q["id"]] = {
            "id": q["id"],
            "question": q["question"],
            "category": q["category"],
            "answerable": q["answerable"],
            "keywords": keywords,
            "confidence": confidence,
            "retrieved": [
                {"score": r["score"], "source": r["source"],
                 "section": r["section"], "text": r["text"]}
                for r in retrieved
            ],
            "hit@k": compute_hit_at_k(texts, keywords),
            "precision@k": compute_precision_at_k(texts, keywords, config.TOP_K),
            "reciprocal_rank": compute_reciprocal_rank(texts, keywords),
            "rag_answer": rag_answer,
            "relevance": rag_scores["relevance"],
            "faithfulness": rag_scores["faithfulness"],
            "baseline_answer": baseline_answer,
            "baseline_relevance": baseline_scores["relevance"],
            "baseline_faithfulness": baseline_scores["faithfulness"],
            "abstain_relevance": config.ABSTAIN_RELEVANCE,
            "abstain_faithfulness": config.ABSTAIN_FAITHFULNESS,
        }
        print(f"        rag rel={rag_scores['relevance']} faith={rag_scores['faithfulness']}"
              f" | baseline rel={baseline_scores['relevance']} faith={baseline_scores['faithfulness']}")

        save_cache(cache, queries)

    save_cache(cache, queries)

    missing = [
        r["id"] for r in cache.values()
        if r["relevance"] is None or r["faithfulness"] is None
        or r["baseline_relevance"] is None or r["baseline_faithfulness"] is None
    ]
    if missing:
        print(f"\nWARNING: judge returned unparseable scores for: {missing}")
        print("Delete those entries from per_query.json and re-run to retry them.")

    print(f"\nSaved {len(cache)} per-query records to {config.PER_QUERY_FILE}")


if __name__ == "__main__":
    run()
