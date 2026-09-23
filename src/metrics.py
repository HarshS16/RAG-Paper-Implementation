"""Abstention-decision metrics.

The abstention decision is treated as binary classification with "should
abstain" (ground-truth answerable == False) as the positive class. Queries
labelled "uncertain" carry no reliable ground truth and are excluded from
every metric in this module.
"""

import math
import re

from config import HALLUCINATION_FAITHFULNESS_CUTOFF

# The system can decline in two distinct ways, and they must not be conflated.
# The confidence gate abstains before generating. Separately, the generator may
# answer "I don't know" of its own accord even when the gate let the query
# through. Both look like a refusal to a user, but only the first is the
# mechanism under study, so generator-side refusals are counted and reported
# separately rather than being folded into the abstain rate.
_REFUSAL = re.compile(
    r"^\s*(i\s+(?:do\s*n[o']?t|don'?t)\s+know"
    r"|i\s+am\s+not\s+sure"
    r"|the\s+context\s+does\s+not\s+(?:provide|contain|mention))",
    re.IGNORECASE,
)


def is_generator_refusal(answer):
    return bool(answer) and bool(_REFUSAL.match(answer.strip()))


def confusion(records, threshold):
    """Confusion matrix of the abstention decision at a given threshold.

    TP: should abstain and did.     FP: should have answered but abstained.
    FN: should abstain but answered. TN: should answer and did.
    """
    tp = fp = tn = fn = 0
    for r in records:
        if r["answerable"] == "uncertain":
            continue
        should_abstain = (r["answerable"] is False)
        did_abstain = r["confidence"] < threshold

        if should_abstain and did_abstain:
            tp += 1
        elif should_abstain and not did_abstain:
            fn += 1
        elif not should_abstain and did_abstain:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def abstention_metrics(records, threshold):
    c = confusion(records, threshold)
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0

    # Matthews correlation coefficient; the denominator collapses to zero when
    # a whole row or column of the matrix is empty, in which case MCC is
    # undefined and conventionally reported as 0.
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denominator) if denominator else 0.0

    return {
        **c,
        "n_labelled": total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "accuracy": accuracy,
        "mcc": mcc,
    }


def response_metrics(records, threshold):
    """Relevance / faithfulness / coverage over the whole 30-query set."""
    relevances, faithfulnesses = [], []
    answered_relevance, answered_faithfulness = [], []
    substantive_faithfulness = []
    abstained = 0
    generator_refusals = 0

    for r in records:
        did_abstain = r["confidence"] < threshold
        if did_abstain:
            abstained += 1
            relevances.append(r["abstain_relevance"])
            faithfulnesses.append(r["abstain_faithfulness"])
        else:
            relevances.append(r["relevance"])
            faithfulnesses.append(r["faithfulness"])
            answered_relevance.append(r["relevance"])
            answered_faithfulness.append(r["faithfulness"])
            # The gate let this query through, but the generator declined to
            # answer it anyway.
            if is_generator_refusal(r.get("rag_answer", "")):
                generator_refusals += 1
            else:
                substantive_faithfulness.append(r["faithfulness"])

    n = len(records)
    n_answered = len(answered_relevance)
    n_substantive = len(substantive_faithfulness)

    # A refusal asserts nothing, so it cannot be unfaithful -- it is counted
    # neither as a hallucination nor in the denominator. This matters: the
    # judge is not consistent about refusals, scoring the identical string
    # "I don't know." as faithfulness 1 on some queries and 5 on others, so
    # leaving them in would let that inconsistency masquerade as hallucination.
    hallucinations = sum(
        1 for f in substantive_faithfulness
        if f is not None and f < HALLUCINATION_FAITHFULNESS_CUTOFF
    )

    return {
        "n": n,
        "n_answered": n_answered,
        "n_substantive": n_substantive,
        "generator_refusals": generator_refusals,
        "abstain_rate": abstained / n if n else 0.0,
        "coverage": n_answered / n if n else 0.0,
        "relevance": _mean(relevances),
        "faithfulness": _mean(faithfulnesses),
        "eff_relevance": _mean(answered_relevance),
        "eff_faithfulness": _mean(answered_faithfulness),
        "hallucinations": hallucinations,
        "hallucination_rate": hallucinations / n_substantive if n_substantive else 0.0,
    }


def quality_score(response, threshold=None):
    """Coverage-weighted tie-break score Q: harmonic mean of coverage and
    effective relevance rescaled to [0, 1]."""
    coverage = response["coverage"]
    rel = response["eff_relevance"] / 5.0
    if coverage + rel == 0:
        return 0.0
    return 2 * coverage * rel / (coverage + rel)


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0
