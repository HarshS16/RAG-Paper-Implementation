"""Produce every number the paper reports, from the cached per-query results.

Writes results/paper_numbers.json, which is the single source of truth for
Tables II-V. Makes no API calls.
"""

import json
import os

import config
from metrics import abstention_metrics, is_generator_refusal, response_metrics


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def load():
    with open(config.PER_QUERY_FILE, encoding="utf-8") as f:
        records = json.load(f)
    with open(config.SELECTED_THRESHOLD_FILE, encoding="utf-8") as f:
        selected = json.load(f)["selected_threshold"]
    return records, selected


def retrieval_metrics(records):
    """Hit@k, Precision@k and MRR@k over the in-domain queries that carry
    ground-truth keywords. Independent of the threshold: the gate decides
    whether context is used, not what gets retrieved."""
    scored = [r for r in records if r.get("keywords")]
    return {
        "n_queries": len(scored),
        "k": config.TOP_K,
        "hit@k": _mean([r["hit@k"] for r in scored]),
        "precision@k": _mean([r["precision@k"] for r in scored]),
        "mrr@k": _mean([r["reciprocal_rank"] for r in scored]),
    }


def baseline_comparison(records, tau):
    """Table II: RAG at tau* against the no-context baseline."""
    rag = response_metrics(records, tau)

    baseline_relevance = _mean([r["baseline_relevance"] for r in records])
    baseline_faithfulness = _mean([r["baseline_faithfulness"] for r in records])
    # Same rule as the RAG side: a refusal asserts nothing, so it is excluded
    # from both the hallucination count and its denominator. Applying the rule
    # to only one of the two systems would tilt the comparison.
    baseline_substantive = [
        r for r in records
        if not is_generator_refusal(r["baseline_answer"])
        and r["baseline_faithfulness"] is not None
    ]
    baseline_hallucinations = sum(
        1 for r in baseline_substantive
        if r["baseline_faithfulness"] < config.HALLUCINATION_FAITHFULNESS_CUTOFF
    )

    return {
        "threshold": tau,
        "n": len(records),
        "rag_relevance_overall": rag["relevance"],
        "rag_relevance_answered": rag["eff_relevance"],
        "rag_faithfulness": rag["faithfulness"],
        # Faithfulness over answered queries only. Reported alongside the
        # overall figure so the reader can see how much of the headline
        # number is carried by the abstention scoring rule, which awards a
        # 5 by definition, rather than by grounded generation.
        "rag_faithfulness_answered": rag["eff_faithfulness"],
        "rag_n_answered": rag["n_answered"],
        "rag_n_substantive": rag["n_substantive"],
        "rag_generator_refusals": rag["generator_refusals"],
        "rag_hallucinations": rag["hallucinations"],
        "rag_hallucination_rate": rag["hallucination_rate"],
        "baseline_relevance": baseline_relevance,
        "baseline_faithfulness": baseline_faithfulness,
        "baseline_hallucinations": baseline_hallucinations,
        "baseline_n_substantive": len(baseline_substantive),
        "baseline_hallucination_rate": (
            baseline_hallucinations / len(baseline_substantive)
            if baseline_substantive else 0.0),
        "abstain_rate": rag["abstain_rate"],
    }


def by_category(records, tau):
    """Table V: abstain rate and relevance per query category."""
    order = ["factual", "reasoning", "comparative", "numeric",
             "in_domain_hard", "near_miss", "out_of_domain"]
    rows = []
    for category in order:
        group = [r for r in records if r["category"] == category]
        if not group:
            continue
        stats = response_metrics(group, tau)
        rows.append({
            "category": category,
            "n": len(group),
            "abstain_rate": stats["abstain_rate"],
            "generator_refusals": stats["generator_refusals"],
            "relevance": stats["relevance"],
            "faithfulness": stats["faithfulness"],
            "mean_confidence": _mean([r["confidence"] for r in group]),
        })
    return rows


def write_output_json(records, tau):
    """results/output.json -- per-query RAG vs baseline at tau*."""
    rows = []
    for r in records:
        abstained = r["confidence"] < tau
        rows.append({
            "id": r["id"],
            "query": r["question"],
            "category": r["category"],
            "answerable": r["answerable"],
            "confidence": r["confidence"],
            "abstain": abstained,
            "hit@k": r["hit@k"],
            "rag_answer": config.ABSTAIN_TEXT if abstained else r["rag_answer"],
            "rag_scores": {
                "relevance": r["abstain_relevance"] if abstained else r["relevance"],
                "faithfulness": r["abstain_faithfulness"] if abstained else r["faithfulness"],
            },
            "baseline_answer": r["baseline_answer"],
            "baseline_scores": {
                "relevance": r["baseline_relevance"],
                "faithfulness": r["baseline_faithfulness"],
            },
        })
    with open(config.OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    return rows


def main():
    records, tau = load()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    retrieval = retrieval_metrics(records)
    comparison = baseline_comparison(records, tau)
    categories = by_category(records, tau)
    abstention = abstention_metrics(records, tau)
    write_output_json(records, tau)

    print("=== Selected threshold: tau* = {}\n".format(tau))

    print("--- Table II: RAG vs baseline -------------------------------")
    print("  Relevance (overall)        RAG {:.2f}   Baseline {:.2f}".format(
        comparison["rag_relevance_overall"], comparison["baseline_relevance"]))
    print("  Relevance (answered n={:2})  RAG {:.2f}   Baseline --".format(
        comparison["rag_n_answered"], comparison["rag_relevance_answered"]))
    print("  Faithfulness (overall)     RAG {:.2f}   Baseline {:.2f}".format(
        comparison["rag_faithfulness"], comparison["baseline_faithfulness"]))
    print("  Faithfulness (answered)    RAG {:.2f}   Baseline --".format(
        comparison["rag_faithfulness_answered"]))
    print("  Hallucination rate (<3)    RAG {}/{}   Baseline {}/{}".format(
        comparison["rag_hallucinations"], comparison["rag_n_substantive"],
        comparison["baseline_hallucinations"], comparison["baseline_n_substantive"]))
    print("     (denominators exclude refusals, which assert nothing)")
    print("  Abstain rate               {:.2%}".format(comparison["abstain_rate"]))
    print("  Generator-side refusals    {}/{} (gate passed them, generator "
          "declined anyway)".format(comparison["rag_generator_refusals"],
                                    comparison["rag_n_answered"]))

    print("\n--- Table III: retrieval quality ----------------------------")
    print("  n = {} in-domain queries, k = {}".format(
        retrieval["n_queries"], retrieval["k"]))
    print("  Hit@k        {:.2f}".format(retrieval["hit@k"]))
    print("  Precision@k  {:.2f}".format(retrieval["precision@k"]))
    print("  MRR@k        {:.2f}".format(retrieval["mrr@k"]))

    print("\n--- Table V: results by category ----------------------------")
    print("  {:16}{:>3}{:>9}{:>8}{:>7}{:>7}{:>7}".format(
        "category", "n", "abstain", "genRef", "rel", "faith", "conf"))
    for row in categories:
        print("  {:16}{:>3}{:>9.2f}{:>8}{:>7.2f}{:>7.2f}{:>7.3f}".format(
            row["category"], row["n"], row["abstain_rate"],
            row["generator_refusals"], row["relevance"],
            row["faithfulness"], row["mean_confidence"]))

    print("\n--- Abstention decision at tau* -----------------------------")
    print("  TP={} FP={} TN={} FN={}  (n={} labelled)".format(
        abstention["tp"], abstention["fp"], abstention["tn"],
        abstention["fn"], abstention["n_labelled"]))
    print("  P={:.2f} R={:.2f} F1={:.2f} Spec={:.2f} Acc={:.2f} MCC={:.2f}".format(
        abstention["precision"], abstention["recall"], abstention["f1"],
        abstention["specificity"], abstention["accuracy"], abstention["mcc"]))

    payload = {
        "selected_threshold": tau,
        "generator_model": config.GENERATOR_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "embedding_model": config.EMBED_MODEL,
        "top_k": config.TOP_K,
        "baseline_comparison": comparison,
        "retrieval": retrieval,
        "by_category": categories,
        "abstention_at_selected": abstention,
    }
    with open(config.PAPER_NUMBERS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(config.RETRIEVAL_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(retrieval, f, indent=2)
    with open(config.CATEGORY_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(categories, f, indent=2)

    print("\nSaved {}".format(config.PAPER_NUMBERS_FILE))
    print("Saved {}".format(config.OUTPUT_FILE))


if __name__ == "__main__":
    main()
