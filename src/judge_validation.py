"""Validate the LLM judge against human ratings.

Reports, for relevance and for faithfulness separately, the mean absolute
error, exact-match rate, Pearson r, Spearman rho and Cohen's quadratic-weighted
kappa between the judge's scores and a human rater's.

Pearson alone is misleading on this kind of data. Ratings cluster at the
ceiling of 5, which inflates the exact-match rate while leaving the
correlation to be determined by a handful of disagreements. Spearman measures
monotonic rather than linear association, and weighted kappa corrects for
chance agreement and penalises larger discrepancies more heavily.

Run with --worksheet to emit a blank rating sheet sampled from the run:

    python src/judge_validation.py --worksheet 15

Fill in human_relevance and human_faithfulness in data/human_ratings.json,
then run the script again with no arguments to compute the statistics.
"""

import json
import os
import random
import sys

import config


def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        # One rater gave the same score to everything; correlation undefined.
        return None
    return num / (dx * dy)


def _rank(values):
    """Average ranks, so ties -- which are the norm on a 1-5 scale -- are
    handled correctly."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def _spearman(xs, ys):
    return _pearson(_rank(xs), _rank(ys))


def _quadratic_weighted_kappa(xs, ys, min_score=1, max_score=5):
    categories = list(range(min_score, max_score + 1))
    index = {c: i for i, c in enumerate(categories)}
    n_cat = len(categories)
    n = len(xs)

    observed = [[0] * n_cat for _ in range(n_cat)]
    for x, y in zip(xs, ys):
        observed[index[int(round(x))]][index[int(round(y))]] += 1

    hist_x = [0] * n_cat
    hist_y = [0] * n_cat
    for x, y in zip(xs, ys):
        hist_x[index[int(round(x))]] += 1
        hist_y[index[int(round(y))]] += 1

    denominator = (n_cat - 1) ** 2
    num = den = 0.0
    for i in range(n_cat):
        for j in range(n_cat):
            weight = ((i - j) ** 2) / denominator
            expected = hist_x[i] * hist_y[j] / n
            num += weight * observed[i][j]
            den += weight * expected
    if den == 0:
        return None
    return 1 - num / den


def agreement(pairs):
    """pairs: list of (human, judge)."""
    human = [p[0] for p in pairs]
    judge = [p[1] for p in pairs]
    n = len(pairs)
    return {
        "n": n,
        "mae": sum(abs(h - j) for h, j in pairs) / n,
        "exact_match": sum(1 for h, j in pairs if h == j) / n,
        "pearson_r": _pearson(human, judge),
        "spearman_rho": _spearman(human, judge),
        "quadratic_kappa": _quadratic_weighted_kappa(human, judge),
    }


def make_worksheet(n_items):
    """Sample items for hand-rating, stratified by category.

    Two filters, both of which follow from what the ratings are for:

    Generator refusals are excluded. The faithfulness rubric scores whether an
    assertion is supported by the context, and a refusal asserts nothing, so
    the axis is undefined for it -- that is precisely the inconsistency the
    paper documents elsewhere, and asking a human to put a number on it would
    import that same undefined behaviour into the reference ratings.

    The remainder is stratified across query categories rather than sampled
    uniformly. A uniform draw over a query set this unbalanced would spend most
    of its items on the largest category and leave the interesting ones -- the
    near-miss and in-domain-hard answers, where judge and human are most likely
    to part company -- represented by nothing at all.
    """
    with open(config.PER_QUERY_FILE, encoding="utf-8") as f:
        records = json.load(f)

    from metrics import is_generator_refusal

    eligible = [r for r in records
                if r.get("relevance") is not None
                and r.get("faithfulness") is not None
                and not is_generator_refusal(r.get("rag_answer", ""))]
    if not eligible:
        raise SystemExit(
            "No substantive answers to rate in {}.".format(config.PER_QUERY_FILE))

    by_category = {}
    for r in eligible:
        by_category.setdefault(r["category"], []).append(r)

    rng = random.Random(config.RANDOM_SEED)
    for group in by_category.values():
        rng.shuffle(group)

    # Round-robin over the categories until the quota is filled, so every
    # category that has an eligible answer contributes before any contributes
    # a second item.
    sample = []
    order = sorted(by_category)
    while len(sample) < min(n_items, len(eligible)):
        drawn = False
        for category in order:
            if len(sample) >= n_items:
                break
            if by_category[category]:
                sample.append(by_category[category].pop())
                drawn = True
        if not drawn:
            break

    sample.sort(key=lambda r: r["id"])
    print("Sampled {} of {} substantive answers ({} refusals excluded), "
          "stratified over {} categories.".format(
              len(sample), len(eligible),
              len(records) - len(eligible), len(by_category)))

    items = []
    for r in sample:
        items.append({
            "id": r["id"],
            "category": r["category"],
            "question": r["question"],
            "context": "\n".join(c["text"] for c in r["retrieved"]),
            "answer": r["rag_answer"],
            "judge_relevance": r["relevance"],
            "judge_faithfulness": r["faithfulness"],
            "human_relevance": None,
            "human_faithfulness": None,
        })

    with open(config.HUMAN_RATINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print("Wrote a blank worksheet with {} items to {}".format(
        len(items), config.HUMAN_RATINGS_FILE))
    print("\nFor each item, read the question, the context and the answer, then")
    print("fill in human_relevance and human_faithfulness on the same 1-5 scale")
    print("the judge uses (see the rubric in src/evaluate.py). Rate them without")
    print("looking at the judge's scores first, or the comparison is worthless.")
    print("\nThen re-run: python src/judge_validation.py")


def main():
    if "--worksheet" in sys.argv:
        position = sys.argv.index("--worksheet")
        count = int(sys.argv[position + 1]) if len(sys.argv) > position + 1 else 15
        make_worksheet(count)
        return

    if not os.path.exists(config.HUMAN_RATINGS_FILE):
        raise SystemExit(
            "{} not found.\nGenerate one with:  python src/judge_validation.py "
            "--worksheet 15".format(config.HUMAN_RATINGS_FILE)
        )

    with open(config.HUMAN_RATINGS_FILE, encoding="utf-8") as f:
        items = json.load(f)

    relevance_pairs, faithfulness_pairs = [], []
    for item in items:
        if item.get("human_relevance") is not None and item.get("judge_relevance") is not None:
            relevance_pairs.append((float(item["human_relevance"]),
                                    float(item["judge_relevance"])))
        if item.get("human_faithfulness") is not None and item.get("judge_faithfulness") is not None:
            faithfulness_pairs.append((float(item["human_faithfulness"]),
                                       float(item["judge_faithfulness"])))

    if not relevance_pairs and not faithfulness_pairs:
        raise SystemExit(
            "No human ratings filled in yet -- every human_relevance and "
            "human_faithfulness field is still null."
        )

    results = {}
    print("{:14}{:>4}{:>7}{:>8}{:>11}{:>11}{:>9}".format(
        "metric", "n", "MAE", "Exact", "Pearson r", "Spearman", "kappa_w"))
    for name, pairs in [("Relevance", relevance_pairs),
                        ("Faithfulness", faithfulness_pairs)]:
        if not pairs:
            continue
        stats = agreement(pairs)
        results[name.lower()] = stats

        def fmt(value):
            return "  n/a" if value is None else "{:.2f}".format(value)

        print("{:14}{:>4}{:>7.2f}{:>8.2f}{:>11}{:>11}{:>9}".format(
            name, stats["n"], stats["mae"], stats["exact_match"],
            fmt(stats["pearson_r"]), fmt(stats["spearman_rho"]),
            fmt(stats["quadratic_kappa"])))

    # Show the disagreements: with ratings piled up at 5, a couple of items
    # usually account for the entire correlation.
    print("\nItems where judge and human disagree:")
    any_disagreement = False
    for item in items:
        for axis in ("relevance", "faithfulness"):
            human = item.get("human_" + axis)
            judge = item.get("judge_" + axis)
            if human is not None and judge is not None and float(human) != float(judge):
                any_disagreement = True
                print("  {:5} {:13} human={} judge={}".format(
                    item["id"], axis, human, judge))
    if not any_disagreement:
        print("  (none -- judge and human agree on every rated item)")

    with open(config.JUDGE_VALIDATION_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved {}".format(config.JUDGE_VALIDATION_FILE))


if __name__ == "__main__":
    main()
