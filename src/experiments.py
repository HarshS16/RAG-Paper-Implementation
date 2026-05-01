from retriever import *
from generator import generate_answer
from evaluate import evaluate_answer, parse_scores

import json

# Load data
chunks = load_chunks()
embeddings, texts = embed_chunks(chunks)
index = build_index(embeddings)

queries = [
    "Explain how IPL auction works and why it is important",
    "Compare IPL format with international cricket formats",
    "Why has Mumbai Indians been successful in IPL?",
    "How does IPL revenue model work?",
    "What are the key differences between IPL playoffs and league stage?",
    "Analyze the role of captaincy in IPL success",
]

results = []

for query in queries:
    
    # 🔹 RAG
    retrieved = retrieve(query, model, index, texts, k=3)
    answer = generate_answer(retrieved, query)

    evaluation_text = evaluate_answer(query, "\n".join(retrieved), answer)
    scores = parse_scores(evaluation_text)

    # 🔹 BASELINE (no context)
    baseline_answer = generate_answer([], query)
    baseline_eval = evaluate_answer(query, "", baseline_answer)
    baseline_scores = parse_scores(baseline_eval)

    # Store results
    results.append({
        "query": query,
        
        "rag_answer": answer,
        "rag_scores": scores,

        "baseline_answer": baseline_answer,
        "baseline_scores": baseline_scores,

        "retrieved": retrieved,
        "evaluation_raw": evaluation_text
    })

    # Debug print
    print("\nQUESTION:", query)
    print("RAG ANSWER:", answer)
    print("RAG EVAL:", scores)
    print("BASELINE EVAL:", baseline_scores)


# Save results
with open("results/output.json", "w") as f:
    json.dump(results, f, indent=2)

print("RETRIEVED:", retrieved)