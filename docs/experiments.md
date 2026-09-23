# Experiments

> **Superseded — numbers in this document are from the earlier run**
> (9 pages / 1,533 passages / 30 queries / tau* = 0.30).
> The corpus, the query set and every result have since been rebuilt:
> 62 pages, 2,732 passages, 160 queries including a near-miss
> unanswerable category, tau* = 0.62. Current numbers live in
> [`../REVISION_RESULTS.md`](../REVISION_RESULTS.md), generated from
> `results/`. The prose here still describes the method accurately; the
> figures quoted in it do not.

## 1. Objectives

- Compare confidence-gated RAG against a no-context baseline.
- Measure retrieval quality independently of the abstention mechanism.
- Score the **abstention decision itself** against ground truth, rather than
  reporting only how often the system happened to refuse.
- Characterise how the threshold τ changes both response quality and decision
  quality.
- Check whether the LLM judge measures what it is assumed to measure.

---

## 2. Query set

30 hand-written queries in [`data/queries.json`](../data/queries.json). Each
carries a category and a ground-truth answerability label.

| Category | n | `answerable` | Purpose |
|---|---:|---|---|
| Factual | 9 | `true` | retrieval / generation |
| Reasoning | 5 | `true` | retrieval / generation |
| Comparative | 2 | `true` | retrieval / generation |
| Numeric | 2 | `true` | retrieval / generation |
| In-domain hard | 4 | `"uncertain"` | stress test |
| Out-of-domain | 8 | `false` | abstention test |

The 18 `true` queries also carry ground-truth keywords, used for the
retrieval metrics.

**In-domain hard** queries ask for oddly specific numbers — an exact strike
rate, an attendance figure — that sound like reasonable IPL trivia but are not
reliably spelled out in prose-style Wikipedia text. They are labelled
`"uncertain"` and excluded from abstention scoring entirely, because there is
no sound way to confirm whether the corpus covers them.

**Out-of-domain** queries have nothing to do with the IPL. They are
unanswerable by construction, and they are the point of the design: they are
what makes it possible to score the abstention decision against an actual
ground truth.

Before use, every `true` query was checked to confirm that the corpus really
does answer it — all 18 retrieve at least one passage containing a
ground-truth keyword.

---

## 3. Systems compared

**System A — baseline.** The generator, given the question with no retrieved
context and no abstention option.

**System B — RAG with confidence-gated abstention.** The proposed system, at
each threshold in the sweep.

Both use the same generator model, so the comparison isolates grounding.

---

## 4. Metrics

### Response quality
- **Relevance**, **Faithfulness** — 1–5, LLM judge. An abstention is assigned
  relevance 1 and faithfulness 5 in code.
- **Effective relevance** — relevance averaged over non-abstained responses
  only. Always reported alongside its denominator, because it is computed over
  a set that shrinks as τ rises.
- **Hallucination rate** — the fraction of non-abstained responses scoring
  faithfulness < 3. A binary, more directly interpretable complement to the
  mean, which can be dragged around by ceiling effects at 5.

### Retrieval quality (18 in-domain queries, k = 3)
- **Hit@k** — did any retrieved passage contain a ground-truth keyword?
- **Precision@k** — what fraction of the k passages did?
- **MRR@k** — reciprocal rank of the first relevant passage.

None of these depend on τ: the threshold decides whether context is *used*,
not what gets *retrieved*.

### Abstention decision (26 queries with a definite label)
Treating "should abstain" as the positive class: **precision**, **recall**,
**F1**, **specificity** (TN/(TN+FP)), **accuracy**, and **Matthews
correlation coefficient**.

MCC is the strictest single number of the set. F1 is buoyed by recall and can
stay high while the answerable side is being destroyed; MCC weighs all four
cells of the confusion matrix and registers that damage.

---

## 5. Threshold sweep

τ is swept over 16 values from 0.10 to 0.80.

The seven values reported in the original write-up (0.45–0.70) all sit *above*
the out-of-domain score ceiling, so they can only ever show precision decaying
on the answerable side — they never reach the region where abstention recall
breaks down. The sweep is extended downward specifically to bracket **both**
edges of the operating region.

Extending it is free: because the sweep is derived from cached per-query
confidences rather than re-run, adding thresholds costs no API calls.

---

## 6. Judge validation

[`src/judge_validation.py`](../src/judge_validation.py) compares judge scores
against human ratings on a sample of the run, reporting MAE, exact-match rate,
Pearson *r*, Spearman *ρ* and quadratic-weighted *κ* separately for relevance
and faithfulness.

Pearson alone is not enough here. Ratings cluster at the ceiling of 5, which
inflates exact-match while leaving the correlation to be decided by a handful
of items. Spearman measures monotonic association; weighted kappa corrects for
chance agreement and penalises larger discrepancies more heavily.

The correlation implementations are verified against `scipy.stats` and
`sklearn.metrics` to within 1e-9.

To produce the rating sheet:

```bash
python src/judge_validation.py --worksheet 15
```

Fill in `human_relevance` and `human_faithfulness` in
`data/human_ratings.json` **before** looking at the judge's scores, then
re-run without arguments.

---

## 7. Reproduction

```bash
python src/scrape.py            # build the corpus      (network)
python src/preprocess.py        # chunk it
python src/experiments.py       # run every query once  (API calls; resumable)
python src/select_threshold.py  # sweep and pick tau*   (no API calls)
python src/analyze.py           # every number the paper reports
python src/plot.py              # Figure 1
python src/plot_threshold.py    # Figure 2
python src/plot_confidence.py   # Figure 3
```

`experiments.py` is the only script that calls the API, and it is resumable:
queries already in `results/per_query.json` are skipped, so hitting the rate
limit does not cost the whole run. It also throttles itself to stay under the
per-minute token ceiling.

---

## 8. Determinism

Generation and judging both run at temperature 0. Retrieval is deterministic
and preserves rank order.

One prior source of nondeterminism has been removed: the retrieved passages
used to be deduplicated through a Python `set`, which both destroyed the
ranking and varied between processes, because string hashing is randomised per
interpreter run. Results were therefore not reproducible across runs despite
the documentation claiming they were.

Hosted LLM inference is not bit-for-bit reproducible even at temperature 0, so
scores may shift slightly between runs. `results/per_query.json` records the
exact answers and scores behind every reported number.
