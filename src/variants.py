"""Confidence-score variants and a chance-level abstention baseline.

Three questions the main experiment does not answer:

  A. Does rank-weighting the similarity scores beat a flat mean?
  B. Does the single best passage carry the signal better than the mean?
  C. How much of the mechanism's decision quality is better than chance?

All three are derived from results/per_query.json, which stores the per-passage
scores, so nothing here calls an API. The confidence definition changes what
the gate sees, but not what was retrieved or generated, so the cached answers
and judge scores remain valid under every variant.

The random baseline is the honest comparison for (C): at a given threshold the
system abstains on some number of queries, and the question is whether
abstaining on *those particular* queries beats abstaining on the same number
chosen at random. Averaged over many seeds, that is the chance level.
"""

import json
import os
import random

import config
from metrics import abstention_metrics, quality_score, response_metrics

# Rank weights for variant A. They sum to 1, so the weighted score stays on the
# same 0-1 scale as the mean and the swept thresholds remain comparable.
RANK_WEIGHTS = [0.5, 0.3, 0.2]
RANDOM_TRIALS = 100


def load_records():
    with open(config.PER_QUERY_FILE, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- variants
def confidence_mean(scores):
    return sum(scores) / len(scores) if scores else 0.0


def confidence_weighted(scores):
    if not scores:
        return 0.0
    weights = RANK_WEIGHTS[:len(scores)]
    total = sum(weights)
    return sum(w * s for w, s in zip(weights, scores)) / total


def confidence_max(scores):
    return max(scores) if scores else 0.0


VARIANTS = [
    ("mean (paper)", confidence_mean),
    ("rank-weighted", confidence_weighted),
    ("max", confidence_max),
]


def rescored(records, fn):
    """Copy the records with `confidence` recomputed under a variant."""
    out = []
    for r in records:
        scores = [p["score"] for p in r["retrieved"]]
        copy = dict(r)
        copy["confidence"] = fn(scores)
        out.append(copy)
    return out


def sweep_best(records):
    """Best threshold under a variant, by the paper's own selection rule."""
    rows = []
    for tau in config.THRESHOLDS:
        response = response_metrics(records, tau)
        abstention = abstention_metrics(records, tau)
        rows.append({"threshold": tau, **response, **abstention,
                     "quality": quality_score(response)})
    best_f1 = max(r["f1"] for r in rows)
    tied = [r for r in rows if r["f1"] == best_f1]
    best_q = max(r["quality"] for r in tied)
    chosen = min([r for r in tied if r["quality"] == best_q],
                 key=lambda r: r["threshold"])
    return chosen, rows


# ------------------------------------------------------- random baseline
def random_baseline(records, n_abstain, trials=RANDOM_TRIALS, seed=config.RANDOM_SEED):
    """Abstain on `n_abstain` randomly chosen queries, averaged over trials.

    Scored over the labelled queries only, exactly as the real metrics are.
    """
    labelled = [r for r in records if r["answerable"] != "uncertain"]
    rng = random.Random(seed)
    totals = {"precision": 0.0, "recall": 0.0, "f1": 0.0,
              "specificity": 0.0, "accuracy": 0.0, "mcc": 0.0}

    for _ in range(trials):
        chosen = set(rng.sample(range(len(labelled)), k=min(n_abstain, len(labelled))))
        # A sentinel threshold of 0.5 with confidences forced to 0 or 1 makes
        # the existing metric code express an arbitrary abstention decision.
        fake = []
        for i, r in enumerate(labelled):
            copy = dict(r)
            copy["confidence"] = 0.0 if i in chosen else 1.0
            fake.append(copy)
        m = abstention_metrics(fake, 0.5)
        for key in totals:
            totals[key] += m[key]

    return {k: v / trials for k, v in totals.items()}


def main():
    records = load_records()

    print("=== Confidence-score variants "
          "(selection rule unchanged: argmax F1, ties by Q then lowest tau) ===\n")
    print("  {:<16}{:>7}{:>7}{:>7}{:>7}{:>7}{:>7}{:>8}".format(
        "confidence c(q)", "tau*", "P", "R", "F1", "Spec", "Acc", "MCC"))

    summary = {}
    for label, fn in VARIANTS:
        variant_records = rescored(records, fn)
        chosen, rows = sweep_best(variant_records)
        print("  {:<16}{:>7.2f}{:>7.2f}{:>7.2f}{:>7.2f}{:>7.2f}{:>7.2f}{:>8.2f}".format(
            label, chosen["threshold"], chosen["precision"], chosen["recall"],
            chosen["f1"], chosen["specificity"], chosen["accuracy"], chosen["mcc"]))
        summary[label] = {
            "selected_threshold": chosen["threshold"],
            "precision": chosen["precision"], "recall": chosen["recall"],
            "f1": chosen["f1"], "specificity": chosen["specificity"],
            "accuracy": chosen["accuracy"], "mcc": chosen["mcc"],
            "abstain_rate": chosen["abstain_rate"],
            "sweep": rows,
        }

    # -- chance level, matched to the real system's abstain count at tau*
    with open(config.SELECTED_THRESHOLD_FILE, encoding="utf-8") as f:
        tau = json.load(f)["selected_threshold"]
    labelled = [r for r in records if r["answerable"] != "uncertain"]
    n_abstain = sum(1 for r in labelled if r["confidence"] < tau)
    real = abstention_metrics(records, tau)
    chance = random_baseline(records, n_abstain)

    print("\n=== Chance-level abstention at tau* = {} "
          "({} of {} labelled queries abstained, {} random trials) ===\n".format(
              tau, n_abstain, len(labelled), RANDOM_TRIALS))
    print("  {:<16}{:>7}{:>7}{:>7}{:>7}{:>7}{:>8}".format(
        "", "P", "R", "F1", "Spec", "Acc", "MCC"))
    for label, m in (("confidence gate", real), ("random", chance)):
        print("  {:<16}{:>7.2f}{:>7.2f}{:>7.2f}{:>7.2f}{:>7.2f}{:>8.2f}".format(
            label, m["precision"], m["recall"], m["f1"],
            m["specificity"], m["accuracy"], m["mcc"]))
    print("  {:<16}{:>7.2f}{:>7.2f}{:>7.2f}{:>7.2f}{:>7.2f}{:>8.2f}".format(
        "difference", real["precision"] - chance["precision"],
        real["recall"] - chance["recall"], real["f1"] - chance["f1"],
        real["specificity"] - chance["specificity"],
        real["accuracy"] - chance["accuracy"], real["mcc"] - chance["mcc"]))

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, "variants.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "rank_weights": RANK_WEIGHTS,
            "random_trials": RANDOM_TRIALS,
            "variants": summary,
            "chance_baseline": {
                "threshold": tau,
                "n_abstained": n_abstain,
                "n_labelled": len(labelled),
                "confidence_gate": {k: real[k] for k in
                                    ("precision", "recall", "f1", "specificity",
                                     "accuracy", "mcc")},
                "random": chance,
            },
        }, f, indent=2)
    print("\nSaved {}".format(path))


if __name__ == "__main__":
    main()
