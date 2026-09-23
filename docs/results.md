# Results

> **Superseded — numbers in this document are from the earlier run**
> (9 pages / 1,533 passages / 30 queries / tau* = 0.30).
> The corpus, the query set and every result have since been rebuilt:
> 62 pages, 2,732 passages, 160 queries including a near-miss
> unanswerable category, tau* = 0.62. Current numbers live in
> [`../REVISION_RESULTS.md`](../REVISION_RESULTS.md), generated from
> `results/`. The prose here still describes the method accurately; the
> figures quoted in it do not.

All numbers below come from `results/paper_numbers.json` and
`results/threshold_experiment.json`, which are regenerated from
`results/per_query.json` by `analyze.py` and `select_threshold.py`. Nothing
here is typed in by hand.

Run configuration: generator `openai/gpt-oss-20b`, judge
`openai/gpt-oss-120b`, embeddings `all-mpnet-base-v2`, *k* = 3, temperature 0.

---

## 1. Threshold selection

τ* = **0.30**, by `argmax F1_abstain`.

F1 reaches 1.00 across τ ∈ [0.30, 0.62] — eight of the sixteen thresholds
tested. The tie is broken by the coverage-weighted quality score *Q*, which is
itself tied across [0.30, 0.55], and finally by preferring the lowest τ.

---

## 2. RAG with abstention vs. no-context baseline

| Metric | RAG (τ\* = 0.30) | Baseline |
|---|---:|---:|
| Relevance (all 30) | 3.40 | 4.60 |
| Relevance (answered only, n = 22) | 4.27 | — |
| Faithfulness (all 30) | **4.63** | **2.37** |
| Unfaithful answers (faithfulness < 3) | **1 / 18** | **16 / 30** |
| Abstain rate | 26.7 % | 0 % |

The baseline scores *higher* on raw relevance because it answers all 30
queries, including the 8 it cannot possibly ground — drawing on pretrained
knowledge rather than the passage it was shown. That is precisely the
behaviour the faithfulness metric penalises, and it does: baseline
faithfulness is 2.37 against RAG's 4.63.

The clearest statement of the difference is the count of unfaithful answers a
user would actually encounter: **1 in 18 for RAG, 16 in 30 for the baseline.**

Both denominators exclude refusals. A refusal asserts nothing, so it can be
neither faithful nor unfaithful, and counting it either way distorts the rate
(see §6).

---

## 3. Retrieval quality (18 in-domain queries, k = 3)

| Metric | Value |
|---|---:|
| Hit@k | 1.00 |
| Precision@k | 0.78 |
| MRR@k | 0.94 |

Hit@k is saturated: every in-domain query retrieves at least one passage
containing a ground-truth keyword. Precision@k = 0.78 says roughly 2.3 of the
3 retrieved passages are relevant, and MRR@k = 0.94 says the first relevant
passage is almost always ranked first.

**This is a change from earlier runs, and it came from fixing the corpus, not
from tuning retrieval.** The previous scraper walked only top-level Wikipedia
sections. On Wikipedia a heading like `History` or `Rivalries` holds no prose
of its own, so that walk discarded most of every page, and Hit@k sat at 0.72.
It also scraped `IPL playoffs`, which redirects to `Indian Premier League`,
storing that page twice, while `IPL auction` does not exist as a page at all
and was silently skipped. Recursive section collection with a redirect guard
took the corpus from 56 sections / 451 chunks to 171 sections / 1,533 chunks.

None of these metrics vary with τ: the threshold governs whether context is
*used*, not what is *retrieved*.

---

## 4. Threshold sweep

| τ | Rel. | Eff. Rel. | Faith. | Abstain | P | R | F1 | Spec. | Acc. | MCC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 3.47 | 3.47 | 3.57 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.69 | 0.00 |
| 0.15 | 3.47 | 3.85 | 4.10 | 0.13 | 1.00 | 0.50 | 0.67 | 1.00 | 0.85 | 0.64 |
| 0.20 | 3.40 | 4.00 | 4.37 | 0.20 | 1.00 | 0.75 | 0.86 | 1.00 | 0.92 | 0.82 |
| 0.25 | 3.40 | 4.13 | 4.50 | 0.23 | 1.00 | 0.88 | 0.93 | 1.00 | 0.96 | 0.91 |
| **0.30** | 3.40 | 4.27 | 4.63 | 0.27 | 1.00 | 1.00 | **1.00** | 1.00 | 1.00 | **1.00** |
| 0.35–0.55 | 3.40 | 4.27 | 4.63 | 0.27 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.60 | 3.40 | 4.43 | 4.63 | 0.30 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.62 | 3.40 | 4.60 | 4.77 | 0.33 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.65 | 3.27 | 5.00 | 4.77 | 0.43 | 0.89 | 1.00 | 0.94 | 0.94 | 0.96 | 0.92 |
| 0.70 | 2.20 | 5.00 | 5.00 | 0.70 | 0.47 | 1.00 | 0.64 | 0.50 | 0.65 | 0.49 |
| 0.75 | 1.53 | 5.00 | 5.00 | 0.87 | 0.36 | 1.00 | 0.53 | 0.22 | 0.46 | 0.28 |
| 0.80 | 1.27 | 5.00 | 5.00 | 0.93 | 0.33 | 1.00 | 0.50 | 0.11 | 0.38 | 0.19 |

The sweep has three regions:

1. **τ < 0.30 — recall climbing.** Abstention recall rises 0.00 → 0.50 → 0.75
   → 0.88 → 1.00. Out-of-domain queries are being answered that should have
   been refused. This is a genuine reliability cost, and it is the region an
   earlier sweep starting at 0.45 could not see.
2. **0.30 ≤ τ ≤ 0.62 — the plateau.** F1 = MCC = 1.00. Every one of the 26
   labelled queries is decided correctly. Any threshold in this band behaves
   identically on the labelled set.
3. **τ > 0.62 — precision decay.** Recall stays pinned at 1.00 while precision
   falls to 0.33 and specificity to 0.11: answerable queries are being refused
   for no gain in domain rejection.

**Effective relevance rises with τ purely as a selection effect.** It reaches
5.00 at τ ≥ 0.65 not because answers improve, but because the denominator
shrinks to the easiest queries. It must always be read with its *n*.

**F1 and MCC diverge above the plateau.** At τ = 0.70, F1 is still 0.64,
buoyed by a recall of 1.00, while MCC has fallen to 0.49. MCC weighs all four
cells of the confusion matrix and registers the damage on the answerable side
that F1 partly conceals. For choosing a threshold, MCC is the number least
likely to give a falsely reassuring picture.

---

## 5. Confidence separation

| Class | n | Confidence range |
|---|---:|---|
| Answerable (in-domain) | 18 | 0.649 – 0.835 |
| Uncertain (in-domain hard) | 4 | 0.598 – 0.643 |
| Unanswerable (out-of-domain) | 8 | 0.103 – 0.275 |

The two labelled classes are separated by a gap of roughly **0.37**, from
0.275 to 0.649, containing no labelled query at all. Any τ inside it makes
identical decisions on the labelled set — which is exactly why the plateau in
§4 is so wide, and why "the effective range of τ is narrow" would be the wrong
conclusion to draw for this corpus.

The four *uncertain* queries sit inside that gap, below every answerable query
and far above every out-of-domain one. With n = 4 this is suggestive, not
conclusive, but it indicates the confidence score carries some signal about
question specificity and not only about topic.

---

## 6. Results by category (τ\* = 0.30)

| Category | n | Abstain rate | Generator refusals | Relevance | Faithfulness | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|
| Factual | 9 | 0.00 | 0 | 5.00 | 4.44 | 0.737 |
| Reasoning | 5 | 0.00 | 0 | 5.00 | 4.60 | 0.710 |
| Comparative | 2 | 0.00 | 0 | 5.00 | 5.00 | 0.673 |
| Numeric | 2 | 0.00 | 0 | 5.00 | 5.00 | 0.683 |
| In-domain hard | 4 | 0.00 | **4** | 1.00 | 4.00 | 0.617 |
| Out-of-domain | 8 | **1.00** | 0 | 1.00 | 5.00 | 0.175 |

Every out-of-domain query is refused; no answerable query is. Relevance is 5.00
in all four answerable categories.

### The second failure mode, and what actually happens

The in-domain hard queries are never abstained on — their confidence (0.617
mean) sits far above τ* — so the confidence gate does not catch them. But the
model does not then answer them wrongly: **all four are declined by the
generator itself**, which replies "I don't know" rather than inventing a strike
rate or an attendance figure.

So the system has **two independent refusal channels**, and they cover
different failures:

- the **confidence gate** catches topic absence (out-of-domain), and
- the **generator** catches question specificity (in-domain hard).

This is why generator-side refusals are counted and reported separately rather
than folded into the abstain rate. It also means the honest description of the
second failure mode is not "these queries are answered incorrectly" but "the
gate misses them and something else catches them."

---

## 7. Abstention does not reduce hallucination here

Among substantive answers, the hallucination rate is **0.06 (1/18) at every
threshold in the operating region**. It does not improve as τ rises.

The single unfaithful answer is q07, *"Which IPL franchise does Virat Kohli
play for?"*, answered **"Virat Kohli plays for the Mumbai Indians"** — wrong;
he plays for Royal Challengers Bengaluru. Its confidence is 0.663, comfortably
above τ*, because retrieval returned passages about Rohit Sharma and MS Dhoni
that are topically close to the query without containing its answer.

This is the sharpest statement of the mechanism's limit: **the faithfulness
gain over the baseline comes from grounding, not from abstention.** Abstention
removes queries that would have been answered ungroundedly; it does not
improve the answers that remain, and it cannot catch a confident retrieval
failure on an in-domain question.

---

## 8. Judge reliability

The judge scores the **identical string** `"I don't know."` as faithfulness 1
on nine queries and faithfulness 5 on three others (q20, q21, q22) — same
model, same temperature 0, same rubric. The only thing that differs is the
surrounding context.

This is direct, reproducible evidence of the unreliability that the human
comparison is meant to probe, and it needs no annotation to observe. Its cause
is a gap in the rubric: faithfulness is defined for assertions, and a refusal
asserts nothing, so the judge has no defined behaviour to fall back on.

Two consequences:

1. Refusals are excluded from the hallucination rate and its denominator
   (§2). Left in, this judge inconsistency alone would have reported 2
   "hallucinations" instead of 1 — a 100 % overstatement.
2. Faithfulness scores on borderline items should not be taken at face value.

Human validation is **not yet done**. `data/human_ratings.json` holds a blank
15-item worksheet generated by `judge_validation.py --worksheet 15`; the
correlation statistics (Pearson *r*, Spearman *ρ*, quadratic-weighted *κ*) are
implemented and verified against `scipy`/`sklearn`, but they have nothing to
run on until a human fills the sheet in.

---

## 9. Limitations

- **Single, small domain.** Nine IPL Wikipedia pages. Whether any of this
  generalises to a larger or noisier corpus is untested.
- **Out-of-domain queries are unambiguous by construction.** They are entirely
  unrelated to the IPL, which is why the retriever separates them so cleanly by
  a 0.37 margin. A query at the *periphery* of the corpus would be far harder,
  and was not tested. The clean plateau in §4 is a property of this query set
  as much as of the method.
- **The confidence score is uncalibrated.** It is a raw mean similarity, and
  the thresholds reported are specific to this embedding model and corpus.
- **The faithfulness rubric is partly circular.** Abstentions are assigned
  faithfulness 5 by definition, so part of the rise in mean faithfulness with
  τ reflects that rule rather than an observed effect. This is why §7 reports
  the hallucination rate over substantive answers, which is not subject to it.
- **The rubric does not define faithfulness for refusals** (§8).
- **No human validation yet** (§8).
- **n = 30, single run.** No confidence intervals, no repeated trials. With 4
  uncertain queries and 8 out-of-domain, single-query changes move the rates
  visibly — one query is 12.5 % of the out-of-domain class.
- **No reranking or hybrid retrieval.** A single dense pass. With Hit@k
  already at 1.00, the remaining headroom is in Precision@k (0.78), which is
  where the q07 failure originates.
