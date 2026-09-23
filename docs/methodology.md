# Methodology

> **Superseded — numbers in this document are from the earlier run**
> (9 pages / 1,533 passages / 30 queries / tau* = 0.30).
> The corpus, the query set and every result have since been rebuilt:
> 62 pages, 2,732 passages, 160 queries including a near-miss
> unanswerable category, tau* = 0.62. Current numbers live in
> [`../REVISION_RESULTS.md`](../REVISION_RESULTS.md), generated from
> `results/`. The prose here still describes the method accurately; the
> figures quoted in it do not.

This project studies a confidence-gated abstention mechanism for
retrieval-augmented generation: the system declines to answer when the
retriever is not confident that it found relevant context. The mechanism is
deliberately simple; the work is in measuring its behaviour carefully.

All settings named below are defined in [`src/config.py`](../src/config.py),
which is the single source of truth for a run.

---

## 1. System overview

A query passes through five stages:

1. retrieve the top-*k* passages,
2. compute a confidence score from their similarity scores,
3. compare that score against a threshold τ,
4. either generate an answer from the retrieved passages, or abstain,
5. evaluate whatever came out.

The corpus is embedded and indexed once, offline.

---

## 2. Corpus

Nine English Wikipedia pages about the Indian Premier League: the league
itself, three franchises, three players, a season page, and a records page.
The domain is narrow on purpose — it has to be small enough that a human can
decide, by hand, which questions it can and cannot answer.

Sections are collected **recursively**. This matters more than it sounds: on
Wikipedia a heading such as `History` or `Rivalries` typically holds no prose
of its own, and every substantive paragraph sits one or two levels below it.
Walking only the top level discards most of each page. The page lead is kept
as its own `Summary` section, and boilerplate headings (references, external
links, see also, …) are skipped.

Page titles are resolved to their canonical form before scraping, and a title
that redirects onto a page already collected is skipped. Without that guard
`IPL playoffs` silently redirects to `Indian Premier League` and the corpus
ends up holding the same page twice.

Sections shorter than 100 characters are dropped. The remainder are split into
overlapping chunks of 200 characters with 50 characters of overlap, using
LangChain's `RecursiveCharacterTextSplitter`.

---

## 3. Embedding and retrieval

- Embeddings: `all-mpnet-base-v2` (sentence-transformers), L2-normalised.
- Index: `faiss.IndexFlatIP`.

Because the vectors are normalised, the inner product the index returns **is
the cosine similarity**. It is a similarity, not a distance: higher means more
similar, and no distance-to-similarity transform is applied or needed. (Earlier
versions of this documentation described the index as "FAISS (L2)", which was
wrong and would imply the opposite ordering.)

Retrieved passages are returned in descending score order, and exact duplicates
are removed in place so that ranking is preserved. This is required for
rank-sensitive metrics such as MRR@k.

---

## 4. Confidence score

For a query *q* retrieving passages *p₁ … p_k* with similarity scores
*s₁ … s_k*, confidence is their arithmetic mean:

```
c(q) = (1/k) · Σ sᵢ        for i = 1 … k
```

with *k* = 3. The score is uncalibrated: it is a raw mean similarity, not a
probability, and its scale is specific to this embedding model and corpus.

---

## 5. Abstention rule

If `c(q) < τ` the system abstains, returning

> I don't know based on the provided context.

Otherwise it generates an answer from the concatenated passages.

An abstention is **assigned** relevance 1 and faithfulness 5 directly in code,
rather than being sent to the judge: it did not answer the question, but it
also did not invent anything. Earlier versions sent abstentions to the judge
with a rubric instructing it to produce those two numbers, which spent tokens
and left the scores dependent on the judge obeying the instruction.

Note that this scoring rule makes part of the faithfulness improvement at high
abstain rates true by construction rather than observed — see the limitations
in [results.md](results.md).

---

## 6. Generation

- Generator: `openai/gpt-oss-20b`, hosted on Groq, temperature 0.
- Input: the retrieved passages, plus the query.
- The prompt instructs the model to treat the context as the primary source
  and to answer in at most three sentences.

`gpt-oss` models are reasoning models: they spend completion tokens in a
separate `reasoning` field before emitting `content`. `max_tokens` has to be
generous (1024 here) or `content` comes back empty.

### Baseline

The baseline is the **same generator model** given the **same question with no
retrieved context**, and with no abstention option. Holding the model fixed
isolates grounding as the variable: any faithfulness gap cannot be explained
by one model simply being stronger than the other.

The baseline answer is judged against the *same retrieved context* the RAG
system was given. That is what makes the two faithfulness columns comparable —
the question being asked is whether the answer is supported by that passage,
not whether it happens to be true in general.

---

## 7. Evaluation

An LLM judge (`openai/gpt-oss-120b`, temperature 0) scores each answer on two
axes, 1–5:

- **Relevance** — does the answer address the question?
- **Faithfulness** — is the answer supported by the provided context?

A single judge model is used for every experiment in this study, so all scores
are mutually comparable. The judge is a different model from the generator, so
the generator never grades its own output.

The judge's reply is parsed with a regular expression that returns `None` for
a field the judge did not emit, so a malformed reply is visible in the results
rather than being silently recorded as a zero.

Judge reliability is not assumed. [`src/judge_validation.py`](../src/judge_validation.py)
compares its scores against human ratings using MAE, exact-match rate,
Pearson *r*, Spearman *ρ* and Cohen's quadratic-weighted *κ*.

---

## 8. Threshold selection

τ is chosen by an explicit, re-runnable rule rather than by looking at a plot.

Every query carries a ground-truth answerability label
*a(q) ∈ {true, false, uncertain}*. Treating *a(q) = false* ("should abstain")
as the positive class, and restricting to queries with a definite true/false
label:

```
τ* = argmax_τ F1_abstain(τ)
```

Ties are expected, because the classes are well separated and long runs of
thresholds score identically. They are broken by a coverage-weighted quality
score

```
Q(τ) = 2 · cov(τ) · rel_eff(τ)/5 / ( cov(τ) + rel_eff(τ)/5 )
```

where `cov(τ) = 1 − abstain_rate(τ)`, and any remaining tie by preferring the
**lowest** τ, since a lower threshold answers more questions at the same
decision quality.

This lives in [`src/select_threshold.py`](../src/select_threshold.py) so the
choice can be re-run and checked.

---

## 9. Why the sweep is derived, not re-run

Retrieval and generation do not depend on τ. The same query retrieves the same
passages at every threshold and, whenever it is answered at all, produces the
same answer. So each query is run **once** — retrieve, generate, judge — and
the entire threshold sweep is derived afterwards by masking on the cached
confidence scores.

Besides being roughly *k* times cheaper in API calls, this makes the sweep
exact. Re-generating and re-judging at every threshold, as earlier versions
did, allowed judge nondeterminism to move the faithfulness column
independently of the variable under study.

---

## 10. Pipeline summary

```
scrape.py  →  preprocess.py  →  experiments.py  →  select_threshold.py
                                      ↓                     ↓
                              per_query.json      threshold_experiment.json
                                      ↓                     ↓
                                   analyze.py  →  paper_numbers.json  →  plots
```

Only `experiments.py` makes API calls. Every downstream script reads cached
results, so analysis and figures can be re-run freely.
