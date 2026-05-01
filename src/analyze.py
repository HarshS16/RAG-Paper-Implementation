import json

with open("results/output.json") as f:
    results = json.load(f)

def compute_metrics(results):
    total_rel = 0
    total_faith = 0
    count = 0

    for r in results:
        if r["scores"]["relevance"] and r["scores"]["faithfulness"]:
            total_rel += r["scores"]["relevance"]
            total_faith += r["scores"]["faithfulness"]
            count += 1

    return {
        "avg_relevance": total_rel / count,
        "avg_faithfulness": total_faith / count
    }

metrics = compute_metrics(results)
print(metrics)
