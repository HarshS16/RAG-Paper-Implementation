import json
import matplotlib.pyplot as plt

# Load data
with open("results/threshold_experiment.json") as f:
    data = json.load(f)

thresholds = [d["threshold"] for d in data]
relevance = [d["relevance"] for d in data]
eff_rel = [d["eff_relevance"] for d in data]
abstain = [d["abstain_rate"] for d in data]
faithfulness = [d["faithfulness"] for d in data]

# Create figure (wider for clarity)
fig, ax1 = plt.subplots(figsize=(10, 5))

# 🔹 Left axis
l1, = ax1.plot(thresholds, relevance, marker='o', linewidth=2, label="Relevance (Overall)")
l2, = ax1.plot(thresholds, eff_rel, marker='^', linewidth=2, label="Relevance (No Abstain)")
l3, = ax1.plot(thresholds, faithfulness, marker='x', linestyle=':', linewidth=2, label="Faithfulness")

ax1.set_xlabel("Threshold")
ax1.set_ylabel("Score")
ax1.set_xticks(thresholds)
ax1.grid(True, linestyle='--', alpha=0.5)

# 🔹 Optimal threshold (cleaner)
optimal_t = 0.55
ax1.axvline(optimal_t, linestyle='--', alpha=0.5)
ax1.text(optimal_t, 2.6, "Optimal ≈ 0.55", ha='center', fontsize=9)

# 🔹 Right axis
ax2 = ax1.twinx()
l4, = ax2.plot(thresholds, abstain, marker='s', linestyle='--', linewidth=2, label="Abstain Rate")
ax2.set_ylabel("Abstain Rate")

# 🔹 Legend OUTSIDE (best readability)
fig.legend(
    handles=[l1, l2, l3, l4],
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False
)

# Title
plt.title("Effect of Threshold on RAG Performance", pad=15)

# Adjust layout so legend fits
plt.tight_layout()
plt.subplots_adjust(right=0.8)

# Save
plt.savefig("results/threshold_plot.png", dpi=300, bbox_inches='tight')
print("Saved: results/threshold_plot.png")

plt.show()