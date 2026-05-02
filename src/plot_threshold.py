import json
import matplotlib.pyplot as plt

# 🔹 Better default styling (important for papers)
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11
})

# Load data
with open("results/threshold_experiment.json") as f:
    data = json.load(f)

thresholds = [d["threshold"] for d in data]
relevance = [d["relevance"] for d in data]
eff_rel = [d["eff_relevance"] for d in data]
abstain = [d["abstain_rate"] for d in data]
faithfulness = [d["faithfulness"] for d in data]

# Create figure (wider for clarity)
fig, ax1 = plt.subplots(figsize=(14, 5))

# 🔹 Left axis (main metrics)
l1, = ax1.plot(thresholds, relevance, marker='o', linewidth=2.5, markersize=6,
               label="Relevance (Overall)")
l2, = ax1.plot(thresholds, eff_rel, marker='^', linewidth=2.5, markersize=6,
               label="Relevance (No Abstain)")
l3, = ax1.plot(thresholds, faithfulness, marker='x', linestyle=':',
               linewidth=2.5, markersize=6, label="Faithfulness")

ax1.set_xlabel("Threshold")
ax1.set_ylabel("Score")
ax1.set_xticks(thresholds)
ax1.set_ylim(2.2, 5.2)  # focus the plot
ax1.grid(True, linestyle='--', alpha=0.4)

# 🔹 Optimal threshold marker (clean + readable)
optimal_t = 0.55
ax1.axvline(optimal_t, linestyle='--', alpha=0.6)
ax1.text(optimal_t, 3.0, "Optimal ≈ 0.55",
         ha='center', fontsize=10)

# 🔹 Right axis (abstain rate)
ax2 = ax1.twinx()
l4, = ax2.plot(thresholds, abstain, marker='s', linestyle='--',
               linewidth=2.5, markersize=6, label="Abstain Rate")
ax2.set_ylabel("Abstain Rate")

# 🔹 Legend (right side, inside image)
handles = [l1, l2, l3, l4]
labels = [h.get_label() for h in handles]

ax1.legend(
    handles,
    labels,
    loc="center left",
    bbox_to_anchor=(1.07, 0.5),
    frameon=True
)

# 🔹 Adjust layout so legend fits cleanly
plt.subplots_adjust(right=0.80)

# Title
plt.title("Effect of Threshold on RAG Performance", pad=15)

# Save high-quality image
plt.savefig("results/threshold_plot.png", dpi=300, bbox_inches='tight')
print("Saved: results/threshold_plot.png")