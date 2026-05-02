import json
import matplotlib.pyplot as plt
import numpy as np

with open("results/output.json") as f:
    results = json.load(f)

# --- Extract ---
rag_rel = [r["rag_scores"]["relevance"] for r in results]
rag_faith = [r["rag_scores"]["faithfulness"] for r in results]

base_rel = [r["baseline_scores"]["relevance"] for r in results]
base_faith = [r["baseline_scores"]["faithfulness"] for r in results]

# --- Overall averages ---
rag_avg = [np.mean(rag_rel), np.mean(rag_faith)]
base_avg = [np.mean(base_rel), np.mean(base_faith)]

# --- Abstain ---
abstains = [r["abstain"] for r in results]
abstain_rate = sum(abstains) / len(abstains)

# --- Effective (no abstain) ---
valid = [r for r in results if not r["abstain"]]

if len(valid) > 0:
    rag_rel_eff = [r["rag_scores"]["relevance"] for r in valid]
    rag_faith_eff = [r["rag_scores"]["faithfulness"] for r in valid]
    rag_eff = [np.mean(rag_rel_eff), np.mean(rag_faith_eff)]
else:
    rag_eff = [0, 0]

# --- Plot ---
labels = ["Relevance", "Faithfulness"]
x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(8, 5))

plt.bar(x - width, rag_avg, width, label="RAG (Overall)")
plt.bar(x, base_avg, width, label="Baseline")
plt.bar(x + width, rag_eff, width, label="RAG (No Abstain)")

# --- Labels ---
plt.xticks(x, labels)
plt.ylabel("Score")
plt.title(f"RAG vs Baseline\nAbstain Rate: {abstain_rate:.2%}")

# --- Value labels ---
for i, v in enumerate(rag_avg):
    plt.text(i - width, v + 0.05, f"{v:.2f}", ha='center')

for i, v in enumerate(base_avg):
    plt.text(i, v + 0.05, f"{v:.2f}", ha='center')

for i, v in enumerate(rag_eff):
    plt.text(i + width, v + 0.05, f"{v:.2f}", ha='center')

plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.savefig("results/final_plot.png", dpi=300, bbox_inches='tight')
print("Saved: results/final_plot.png")

plt.show()