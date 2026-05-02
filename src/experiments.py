from retriever import *
from generator import generate_answer
from evaluate import evaluate_answer, parse_scores, compute_hit_at_k
import json

# Load data
chunks = load_chunks()
embeddings, chunks = embed_chunks(chunks)
index = build_index(embeddings)

queries = [
    {"question": "Who has won the most IPL titles?", "answer": ["mumbai indians"]},
    {"question": "What is IPL auction?", "answer": ["auction", "players", "bidding"]},
    {"question": "Why has Mumbai Indians been successful in IPL?", "answer": ["leadership", "titles"]}
]

thresholds = [0.45, 0.5, 0.55, 0.6, 0.62, 0.65, 0.68, 0.7]
all_results = []

K = 3  # retrieval depth

for THRESHOLD in thresholds:
    print(f"\n===== Running for THRESHOLD = {THRESHOLD} =====")

    results = []

    for item in queries:
        query = item["question"]
        gt = item["answer"]

        # 🔹 Retrieve
        retrieved = retrieve(query, model, index, chunks, k=K)

        # 🔹 Deduplicate texts
        retrieved_texts = list({r["text"] for r in retrieved})

        # 🔹 Hit@k
        hit = compute_hit_at_k(retrieved_texts, gt)

        # 🔹 Confidence score (average)
        if len(retrieved) > 0:
            avg_score = sum(r["score"] for r in retrieved) / len(retrieved)
        else:
            avg_score = 0

        # 🔹 Abstain logic
        if avg_score < THRESHOLD:
            answer = "I don't know based on the provided context."
            is_abstain = True
        else:
            answer = generate_answer(retrieved_texts, query)
            is_abstain = False

        # 🔹 Evaluate
        evaluation_text = evaluate_answer(query, "\n".join(retrieved_texts), answer)
        rag_scores = parse_scores(evaluation_text)

        results.append({
            "rag_scores": rag_scores,
            "abstain": is_abstain,
            "hit@k": hit
        })

    # =========================
    # ✅ Aggregate INSIDE LOOP
    # =========================

    rag_rel = [r["rag_scores"]["relevance"] for r in results]
    rag_faith = [r["rag_scores"]["faithfulness"] for r in results]

    abstain_rate = sum(r["abstain"] for r in results) / len(results)
    hit_rate = sum(r["hit@k"] for r in results) / len(results)

    # Effective relevance (no abstain)
    valid = [r for r in results if not r["abstain"]]
    if len(valid) > 0:
        rag_rel_eff = [r["rag_scores"]["relevance"] for r in valid]
        eff_relevance = sum(rag_rel_eff) / len(rag_rel_eff)
    else:
        eff_relevance = 0

    all_results.append({
        "threshold": THRESHOLD,
        "relevance": sum(rag_rel) / len(rag_rel),
        "eff_relevance": eff_relevance,
        "faithfulness": sum(rag_faith) / len(rag_faith),
        "abstain_rate": abstain_rate,
        "hit_rate": hit_rate
    })

    # 🔹 Debug print
    print(f"Relevance: {all_results[-1]['relevance']:.2f}")
    print(f"Effective Relevance: {eff_relevance:.2f}")
    print(f"Faithfulness: {all_results[-1]['faithfulness']:.2f}")
    print(f"Abstain Rate: {abstain_rate:.2f}")
    print(f"Hit@k: {hit_rate:.2f}")

# =========================
# Save results
# =========================

with open("results/threshold_experiment.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\nSaved threshold experiment results")