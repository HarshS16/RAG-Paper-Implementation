"""Figure 2: effect of the abstention threshold.

Two stacked panels sharing one x axis, rather than the single panel with a
twinned y axis the earlier version used. Scores live on a 1-5 scale and rates
on a 0-1 scale; putting both on one frame with two different y axes invites
the reader to compare the crossing points of curves that share no units, and
lets the apparent steepness of any curve be set by the axis limits.

The selected threshold is read from results/selected_threshold.json, so the
figure can never disagree with the selection rule the way a hard-coded
"Optimal = 0.55" annotation eventually did.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

# Categorical slots 1, 2, 3, 7 of the validated palette. Identity is carried by
# marker and dash pattern as well as hue, so the figure survives greyscale
# printing and colour-vision deficiency.
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
GRID = "#d8d7d2"
INK, INK_SOFT = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 6,
    "axes.titlesize": 6.5,
    "axes.labelsize": 6,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5,
    "legend.fontsize": 5,
    "axes.edgecolor": INK_SOFT,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def style(ax):
    ax.grid(True, linestyle="-", linewidth=0.6, color=GRID, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    with open(config.THRESHOLD_EXPERIMENT_FILE, encoding="utf-8") as f:
        rows = json.load(f)
    with open(config.SELECTED_THRESHOLD_FILE, encoding="utf-8") as f:
        selected = json.load(f)["selected_threshold"]

    taus = [r["threshold"] for r in rows]
    n = rows[0]["n"]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(3.45, 3.45), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12},
    )

    # -- panel A: judge scores, 1-5 -------------------------------------
    series_top = [
        ("Relevance (all {} queries)".format(n),
         [r["relevance"] for r in rows], BLUE, "o", "-"),
        ("Relevance (answered only)",
         [r["eff_relevance"] for r in rows], ORANGE, "^", "-"),
        ("Faithfulness (all {})".format(n),
         [r["faithfulness"] for r in rows], AQUA, "s", "--"),
        ("Faithfulness (answered only)",
         [r["eff_faithfulness"] for r in rows], VIOLET, "v", ":"),
    ]
    for label, values, colour, marker, dash in series_top:
        ax_top.plot(taus, values, marker=marker, linestyle=dash, color=colour,
                    linewidth=1.1, markersize=2.6, label=label, zorder=3)
    ax_top.set_ylabel("Judge score (1-5)")
    ax_top.set_ylim(0.8, 5.2)
    ax_top.legend(loc="lower left", framealpha=0.95, edgecolor=GRID,
                  ncol=1, handlelength=1.6, borderpad=0.35,
                  labelspacing=0.25)
    style(ax_top)

    # -- panel B: rates and decision quality, 0-1 ------------------------
    series_bottom = [
        ("Abstain rate", [r["abstain_rate"] for r in rows], BLUE, "o", "-"),
        ("Abstention precision", [r["precision"] for r in rows], ORANGE, "^", "-"),
        ("Abstention recall", [r["recall"] for r in rows], AQUA, "s", "--"),
        ("MCC", [r["mcc"] for r in rows], VIOLET, "D", "-."),
    ]
    for label, values, colour, marker, dash in series_bottom:
        ax_bottom.plot(taus, values, marker=marker, linestyle=dash, color=colour,
                       linewidth=1.1, markersize=2.6, label=label, zorder=3)
    ax_bottom.set_ylabel("Rate / coefficient (0-1)")
    ax_bottom.set_xlabel(r"Abstention threshold $\tau$")
    ax_bottom.set_ylim(-0.05, 1.08)
    ax_bottom.legend(loc="center left", framealpha=0.95, edgecolor=GRID,
                     handlelength=1.6, borderpad=0.35, labelspacing=0.25)
    style(ax_bottom)

    for ax in (ax_top, ax_bottom):
        ax.axvline(selected, color=INK_SOFT, linestyle=":", linewidth=1.4, zorder=1)
    # Rotated along the rule rather than set beside it at the top of the panel,
    # where it collided with whichever series was sitting at the ceiling.
    ax_top.annotate(
        r"selected $\tau^*$ = {}".format(selected),
        # Low in the panel: at tau* the score curves all sit above 3, so the
        # rotated label only stays clear of them near the axis floor.
        xy=(selected, 1.75), xytext=(-3, 0), textcoords="offset points",
        rotation=90, ha="right", va="center", fontsize=5.5, color=INK_SOFT,
    )

    # 0.60 and 0.62 are only 0.02 apart and their labels overlap at any
    # readable font size, so 0.62 is plotted but not labelled.
    tick_values = [t for t in taus if t != 0.62]
    ax_bottom.set_xticks(tick_values)
    ax_bottom.set_xticklabels(["{:.2f}".format(t) for t in tick_values],
                              rotation=45, ha="right")

    ax_top.set_title("Effect of the abstention threshold", pad=5)

    fig.savefig(config.THRESHOLD_PLOT_FILE, dpi=300, bbox_inches="tight")
    print("Saved: {}".format(config.THRESHOLD_PLOT_FILE))


if __name__ == "__main__":
    main()
