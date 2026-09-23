"""Figure 1: RAG with abstention against the no-context baseline."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRID = "#d8d7d2"
INK, INK_SOFT = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.edgecolor": INK_SOFT,
    "text.color": INK,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def main():
    with open(config.PAPER_NUMBERS_FILE, encoding="utf-8") as f:
        numbers = json.load(f)
    c = numbers["baseline_comparison"]

    groups = ["Relevance", "Faithfulness"]
    series = [
        ("RAG (all {} queries)".format(c["n"]),
         [c["rag_relevance_overall"], c["rag_faithfulness"]], BLUE),
        ("Baseline (no context)",
         [c["baseline_relevance"], c["baseline_faithfulness"]], ORANGE),
        # Answered-only on both axes: the overall faithfulness figure is lifted
        # by the scoring rule that hands every abstention a 5, so the paper
        # reports the un-lifted number beside it rather than only the headline.
        ("RAG (answered only, n={})".format(c["rag_n_answered"]),
         [c["rag_relevance_answered"], c["rag_faithfulness_answered"]], AQUA),
    ]

    x = np.arange(len(groups))
    width = 0.26

    fig, ax = plt.subplots(figsize=(6.2, 4.0))

    for i, (label, values, colour) in enumerate(series):
        offset = (i - 1) * width
        heights = [0 if v is None else v for v in values]
        bars = ax.bar(x + offset, heights, width * 0.92, label=label,
                      color=colour, edgecolor="white", linewidth=1.2, zorder=3)
        # Direct labels on every bar: the aqua slot sits below 3:1 against a
        # white surface, so identity must not rest on hue alone.
        for bar, value in zip(bars, values):
            if value is None:
                continue
            ax.annotate("{:.2f}".format(value),
                        xy=(bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Judge score (1-5)")
    ax.set_ylim(0, 5.6)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.grid(True, axis="y", linestyle="-", linewidth=0.6, color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.set_title(
        r"RAG with abstention vs. no-context baseline ($\tau^*$ = {}, "
        "abstain rate {:.0%})".format(c["threshold"], c["abstain_rate"]),
        pad=8)
    # Below the axes rather than inside them: at upper right the legend box
    # covered the value label on the tallest faithfulness bar.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3,
              framealpha=0.95, edgecolor=GRID, columnspacing=1.4)

    fig.savefig(config.FINAL_PLOT_FILE, dpi=300, bbox_inches="tight")
    print("Saved: {}".format(config.FINAL_PLOT_FILE))


if __name__ == "__main__":
    main()
