"""Sweep the abstention threshold and select tau* by an explicit rule.

tau* = argmax_tau F1_abstain(tau), computed only over the queries carrying a
definite true/false answerability label. Ties -- which are expected, because
the two classes are well separated and a whole run of thresholds can score
identically -- are broken by the coverage-weighted quality score Q, and any
remaining tie by preferring the lowest tau, since a lower threshold answers
more questions for the same decision quality.

The sweep is derived from the cached per-query confidences, so this script
makes no API calls and can be re-run freely.
"""

import json
import os

import config
from metrics import abstention_metrics, quality_score, response_metrics


def load_records():
    if not os.path.exists(config.PER_QUERY_FILE):
        raise SystemExit(
            f"{config.PER_QUERY_FILE} not found -- run `python src/experiments.py` first."
        )
    with open(config.PER_QUERY_FILE, encoding="utf-8") as f:
        return json.load(f)


def sweep(records):
    rows = []
    for tau in config.THRESHOLDS:
        response = response_metrics(records, tau)
        abstention = abstention_metrics(records, tau)
        rows.append({
            "threshold": tau,
            **response,
            **abstention,
            "quality": quality_score(response),
        })
    return rows


def select(rows):
    best_f1 = max(r["f1"] for r in rows)
    tied = [r for r in rows if r["f1"] == best_f1]

    best_quality = max(r["quality"] for r in tied)
    tied_q = [r for r in tied if r["quality"] == best_quality]

    chosen = min(tied_q, key=lambda r: r["threshold"])
    return chosen, tied, tied_q


def check_labels(records):
    """F1 on an empty positive class is 0 at every threshold, which makes the
    selection meaningless rather than merely poor. Fail loudly instead."""
    positives = sum(1 for r in records if r["answerable"] is False)
    negatives = sum(1 for r in records if r["answerable"] is True)
    if positives == 0 or negatives == 0:
        raise SystemExit(
            "Cannot select a threshold: the labelled set has {} unanswerable "
            "and {} answerable queries, and F1 needs both classes present.\n"
            "Only {} queries are in {} -- has experiments.py finished?"
            .format(positives, negatives, len(records), config.PER_QUERY_FILE)
        )


def main():
    records = load_records()
    check_labels(records)
    rows = sweep(records)
    chosen, tied_f1, tied_q = select(rows)

    header = (f"{'tau':>6} {'Rel':>6} {'EffRel':>7} {'Faith':>6} {'FaithA':>7} "
              f"{'Abst':>6} {'Cov':>6} {'Hall':>6} {'P':>6} {'R':>6} {'F1':>6} "
              f"{'Spec':>6} {'Acc':>6} {'MCC':>7} {'Q':>6}")
    print(header)
    print("-" * len(header))
    for r in rows:
        mark = " *" if r["threshold"] == chosen["threshold"] else ""
        print(f"{r['threshold']:6.2f} {r['relevance']:6.2f} {r['eff_relevance']:7.2f} "
              f"{r['faithfulness']:6.2f} {r['eff_faithfulness']:7.2f} "
              f"{r['abstain_rate']:6.2f} {r['coverage']:6.2f} "
              f"{r['hallucination_rate']:6.2f} {r['precision']:6.2f} {r['recall']:6.2f} "
              f"{r['f1']:6.2f} {r['specificity']:6.2f} {r['accuracy']:6.2f} "
              f"{r['mcc']:7.2f} {r['quality']:6.2f}{mark}")

    print(f"\nBest F1 = {chosen['f1']:.2f}, attained at "
          f"{[r['threshold'] for r in tied_f1]}")
    if len(tied_f1) > 1:
        print(f"Tie broken by Q = {chosen['quality']:.3f}, attained at "
              f"{[r['threshold'] for r in tied_q]}")
        if len(tied_q) > 1:
            print("Still tied; lowest tau preferred (answers the most queries).")
    print(f"\nSelected tau* = {chosen['threshold']}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(config.THRESHOLD_EXPERIMENT_FILE, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(config.SELECTED_THRESHOLD_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "selected_threshold": chosen["threshold"],
            "rule": "argmax F1_abstain; ties broken by coverage-weighted Q, then lowest tau",
            "best_f1": chosen["f1"],
            "tied_on_f1": [r["threshold"] for r in tied_f1],
            "tied_on_quality": [r["threshold"] for r in tied_q],
            "metrics_at_selected": chosen,
        }, f, indent=2)

    print(f"Saved {config.THRESHOLD_EXPERIMENT_FILE}")
    print(f"Saved {config.SELECTED_THRESHOLD_FILE}")


if __name__ == "__main__":
    main()
