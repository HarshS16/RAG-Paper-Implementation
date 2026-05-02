import json
from config import OUTPUT_FILE

with open(OUTPUT_FILE) as f:
    results = json.load(f)


def compute_metrics(results):
    rag_rel, rag_faith = [], []
    base_rel, base_faith = [], []

    for r in results:
        # RAG
        rag_rel.append(r["rag_scores"]["relevance"])
        rag_faith.append(r["rag_scores"]["faithfulness"])

        # Baseline
        base_rel.append(r["baseline_scores"]["relevance"])
        base_faith.append(r["baseline_scores"]["faithfulness"])

    return {
        "rag_avg_relevance": sum(rag_rel) / len(rag_rel),
        "rag_avg_faithfulness": sum(rag_faith) / len(rag_faith),
        "baseline_avg_relevance": sum(base_rel) / len(base_rel),
        "baseline_avg_faithfulness": sum(base_faith) / len(base_faith),
    }


def compute_hit_rate(results):
    hits = [r["hit@k"] for r in results]
    return sum(hits) / len(hits)

id="abstain_rate"
abstains = [r["is_abstain"] for r in results]
abstain_rate = sum(abstains) / len(abstains)

print("Abstain Rate:", abstain_rate)
metrics = compute_metrics(results)
hit_rate = compute_hit_rate(results)

id="effective_metrics"
valid = [r for r in results if not r["is_abstain"]]

rag_rel = [r["rag_scores"]["relevance"] for r in valid]
rag_faith = [r["rag_scores"]["faithfulness"] for r in valid]

print("RAG Relevance (no abstain):", sum(rag_rel)/len(rag_rel))
print("RAG Faithfulness (no abstain):", sum(rag_faith)/len(rag_faith))

print("\n=== FINAL METRICS ===")
print("Hit@k:", hit_rate)
print("RAG Avg Relevance:", metrics["rag_avg_relevance"])
print("RAG Avg Faithfulness:", metrics["rag_avg_faithfulness"])
print("Baseline Avg Relevance:", metrics["baseline_avg_relevance"])
print("Baseline Avg Faithfulness:", metrics["baseline_avg_faithfulness"])