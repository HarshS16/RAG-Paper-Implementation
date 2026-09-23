"""Figure 3: retrieval confidence by ground-truth answerability class.

Every query is drawn as its own point. The question the figure has to answer
is whether a single scalar threshold can separate the classes at all, and that
is a question about where the extreme points of each class fall, not about
their central tendency, so no box plot or mean is substituted for the sample.

The four classes are drawn separately on purpose. Pooling the two unanswerable
groups would hide the finding: clearly out-of-domain queries sit far below
every answerable one, while near-miss queries -- which concern the IPL but ask
for something the corpus does not hold -- land inside the answerable range.
The figure computes both boundaries from the data and annotates whichever
case actually obtains, so it cannot assert a gap that a later run has closed.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
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

# (label stem, predicate over a record, colour, marker)
CLASSES = [
    ("Answerable\n(in-domain)",
     lambda r: r["answerable"] is True, BLUE, "o"),
    ("Uncertain\n(in-domain hard)",
     lambda r: r["answerable"] == "uncertain", ORANGE, "^"),
    ("Unanswerable\n(near-miss)",
     lambda r: r["answerable"] is False and r["category"] == "near_miss", VIOLET, "D"),
    ("Unanswerable\n(out-of-domain)",
     lambda r: r["answerable"] is False and r["category"] == "out_of_domain", AQUA, "s"),
]


def main():
    with open(config.PER_QUERY_FILE, encoding="utf-8") as f:
        records = json.load(f)
    with open(config.SELECTED_THRESHOLD_FILE, encoding="utf-8") as f:
        selected = json.load(f)["selected_threshold"]

    groups = [(label, [r["confidence"] for r in records if pred(r)], colour, marker)
              for label, pred, colour, marker in CLASSES]
    missing = [label.replace("\n", " ") for label, values, _, _ in groups if not values]
    if missing:
        raise SystemExit(
            "No queries found for: {}. Has experiments.py finished?".format(missing))

    answerable = [r["confidence"] for r in records if r["answerable"] is True]
    near_miss = [r["confidence"] for r in records
                 if r["answerable"] is False and r["category"] == "near_miss"]
    ood = [r["confidence"] for r in records
           if r["answerable"] is False and r["category"] == "out_of_domain"]

    fig, ax = plt.subplots(figsize=(7.6, 4.3))

    # Every annotation sits in the right-hand margin rather than over the
    # scatter. The answerable column is dense between 0.5 and 0.85, which is
    # exactly the band the threshold line and the overlap note want to occupy,
    # so anything placed there lands on top of the data and cannot be read.
    # RIGHT_EDGE is past the last column of points; xlim is widened to match.
    RIGHT_EDGE = 4.62
    X_MAX = 4.68
    boxed = dict(boxstyle="round,pad=0.28", facecolor="white",
                 edgecolor=GRID, linewidth=0.6, alpha=0.94)

    # -- the band between the answerable floor and the out-of-domain ceiling.
    # Any threshold inside it makes identical decisions on those two classes.
    ood_max, answerable_min = max(ood), min(answerable)
    if ood_max < answerable_min:
        ax.axhspan(ood_max, answerable_min, color=GRID, alpha=0.45, zorder=0)
        ax.annotate(
            "clean separation gap\nvs. out-of-domain: {:.2f}".format(
                answerable_min - ood_max),
            xy=(RIGHT_EDGE, (ood_max + answerable_min) / 2),
            ha="right", va="center", fontsize=8, color=INK_SOFT, bbox=boxed)

    # -- the overlap with the near-miss class, which is the harder test.
    near_miss_max = max(near_miss)
    if near_miss_max >= answerable_min:
        overlapped = sum(1 for v in answerable if v <= near_miss_max)
        ax.axhspan(answerable_min, near_miss_max, color=ORANGE, alpha=0.10, zorder=0)
        ax.annotate(
            "near-miss queries overlap\nthe answerable range: no\n"
            "threshold separates them\n({} of {} answerable queries\n"
            "sit below the highest-scoring\nnear-miss query)".format(
                overlapped, len(answerable)),
            xy=(RIGHT_EDGE, (answerable_min + near_miss_max) / 2 + 0.16),
            ha="right", va="center", fontsize=8, color=INK_SOFT, bbox=boxed)

    for i, (label, values, colour, marker) in enumerate(groups):
        # Deterministic spread so overlapping points stay readable.
        offsets = [(j - (len(values) - 1) / 2) * (0.55 / max(len(values), 1))
                   for j in range(len(values))]
        ax.scatter([i + o for o in offsets], values, s=26, color=colour,
                   marker=marker, edgecolor="white", linewidth=0.6,
                   zorder=3, label="{} (n={})".format(
                       label.replace("\n", " "), len(values)))

    # The threshold line is labelled below itself in the margin, clear of both
    # the scatter and the two notes above it.
    ax.axhline(selected, color=INK, linestyle=":", linewidth=1.5, zorder=2)
    ax.annotate(r"selected $\tau^*$ = {}".format(selected),
                xy=(RIGHT_EDGE, selected - 0.055), ha="right", va="top",
                fontsize=8, color=INK, fontweight="bold", bbox=boxed, zorder=4)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(["{}\n(n={})".format(label, len(values))
                        for label, values, _, _ in groups])
    ax.set_ylabel("Retrieval confidence  $c(q)$")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.5, X_MAX)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.6, color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Retrieval confidence separates out-of-domain queries cleanly, "
                 "near-miss queries not at all", pad=8)

    fig.savefig(config.CONFIDENCE_PLOT_FILE, dpi=300, bbox_inches="tight")
    print("Saved: {}".format(config.CONFIDENCE_PLOT_FILE))
    print("  answerable    min {:.3f}  max {:.3f}  (n={})".format(
        min(answerable), max(answerable), len(answerable)))
    print("  near-miss     min {:.3f}  max {:.3f}  (n={})".format(
        min(near_miss), near_miss_max, len(near_miss)))
    print("  out-of-domain min {:.3f}  max {:.3f}  (n={})".format(
        min(ood), ood_max, len(ood)))


if __name__ == "__main__":
    main()
