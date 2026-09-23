# Paper revision checklist

Every number in the draft now has a reproducible source. This file lists what
changes, what stays, and — more importantly — the three places where the new
data **contradicts a claim the paper currently makes**.

Source of truth: `results/paper_numbers.json`, `results/threshold_experiment.json`,
`results/per_query.json`. Regenerate with `analyze.py` / `select_threshold.py`.

Legend: 🔴 claim must change · 🟡 number must change · 🟢 reproduced, leave as is

---

## Part 1 — The three claims that must change

### 🔴 1. "Recall remains at 1.00 across every threshold evaluated"

*Abstract; §V-C; §VI; §VIII.*

This is false, and it was an artifact of where the sweep started. All seven
thresholds in the original sweep (0.45–0.70) sit **above** the decision
boundary, so recall was already saturated before the first data point.

Extending the sweep down to τ = 0.10 shows recall genuinely climbing:

| τ | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 |
|---|---:|---:|---:|---:|---:|
| Abstention recall | 0.00 | 0.50 | 0.75 | 0.88 | **1.00** |

**What to say instead.** Recall saturates at τ = 0.30 and holds at 1.00
thereafter. Below that there is a real coverage–reliability tradeoff of the
conventional kind; above it, the behaviour is precision decay. The paper's
"precision decay, not a tradeoff" framing is right *about the region it
measured*, and the corrected claim is stronger because it says where each
description applies.

This also rescues the paper from a reviewer's obvious objection — that a sweep
which never varies its headline metric has not established anything about that
metric.

### 🔴 2. "The effective range of τ for this corpus is narrow (approximately 0.45 to 0.50)"

*§VI.*

The opposite is true. **F1 = MCC = 1.00 for every τ in [0.30, 0.62]** — eight
of sixteen thresholds tested.

The reason is a wide separation gap: out-of-domain queries score 0.103–0.275,
answerable ones 0.649–0.835, and **no labelled query falls in the 0.37-wide
band between them**. Any threshold inside it makes identical decisions on the
labelled set.

**What to say instead.** The operating region is wide, not narrow, and its
width is a measurable property of the corpus (the class separation gap) rather
than something to be discovered by trial and error. The deployment advice
survives intact and gets sharper: measure the gap, then put τ in it.

New Figure 3 (`results/confidence_plot.png`) shows this directly.

### 🔴 3. "In-domain questions too specific for the corpus pass the confidence check and are answered incorrectly"

*Abstract; §V-D; §VIII.*

The first half holds — all 4 in-domain-hard queries pass the gate (mean
confidence 0.617, well above τ\*). The second half does not. **None of them is
answered incorrectly. All four are declined by the generator itself**, which
replies "I don't know" rather than inventing a strike rate.

Relevance for the category is still 1.00, so **Table V does not change** — but
the mechanism behind that 1.00 is not what the text says it is.

**What to say instead.** The system has two independent refusal channels that
cover different failures: the confidence gate catches *topic absence*
(out-of-domain), and the generator catches *question specificity* (in-domain
hard). This is a better result than the draft claims, and it turns the paper's
own future-work suggestion — "look into generator-side uncertainty" — into a
finding with evidence behind it.

The genuine version of the "confident but wrong" failure is elsewhere, and
there is exactly one instance of it: see the new §V-D material in Part 4.

---

## Part 2 — Numbers

### Abstract

| Claim | Old | New |
|---|---|---|
| Faithfulness, baseline → RAG | 1.77 → 4.70 | **2.37 → 4.63** |
| Corpus | 30 questions, small IPL corpus | unchanged (30 questions) |
| Precision and recall at lowest τ | both 1.00 | 🔴 recall is 0.00 at the lowest τ tested (0.10); both reach 1.00 at τ = 0.30 |
| Judge validation | r = 0.96 relevance, r = 0.06 faithfulness | 🔴 no human data exists — see Part 5 |

### §III-D Generation and evaluation models

- 🟢 Generator `openai/gpt-oss-20b` — correct and available.
- 🟡 Judge: the draft uses `openai/gpt-oss-120b` for the sweep and
  `qwen/qwen3.6-27b` for the baseline comparison.
  **`qwen/qwen3.6-27b` does not exist on Groq** (the account carries
  `qwen/qwen3.8-27b`). The new run uses **`openai/gpt-oss-120b` throughout.**
- ✅ **Delete the paragraph about switching judges**, and delete the
  corresponding limitation. All experiments now share one judge, so all scores
  are mutually comparable. The generator still never grades its own output
  (20b generates, 120b judges).
- ➕ Add: `gpt-oss` are reasoning models that emit a separate `reasoning` field;
  `max_tokens` must be generous (1024) or `content` returns empty.

### §IV-A Dataset

| Claim | Old | New |
|---|---|---|
| Pages | 9 (league, 3 franchises, 3 players, the auction, playoff format) | 9 (league, 3 franchises, 3 players, **a season page, a records page**) |
| Sections | 56 | **171** |
| Passages | 460 | **1,533** |
| Chunk size / overlap | 200 / 50 chars | 🟢 unchanged |

**This needs a sentence of explanation, because the old numbers were wrong in
a way that mattered.** Three corpus defects, all now fixed:

1. `IPL playoffs` **redirects to** `Indian Premier League` — the corpus held
   the same page twice.
2. `IPL auction` **does not exist** as a Wikipedia page and was silently
   skipped. So only 8 pages were ever scraped, not 9.
3. Only **top-level** sections were collected. On Wikipedia a heading like
   `History` or `Rivalries` has no prose of its own, so this discarded the
   bulk of every page — Royal Challengers Bengaluru contributed just two
   sections, `Fan support` and `See also`.

The old "56 sections" was a coincidence: the duplicated page almost exactly
offset the missing one. Sections are now collected recursively, the lead
summary is kept, boilerplate headings are dropped, and redirect collisions are
detected and skipped.

### §IV-B Query set / Table I

🟢 **Unchanged.** Composition reproduces exactly: 9 factual, 5 reasoning,
2 comparative, 2 numeric (18 `true`), 4 in-domain hard (`uncertain`),
8 out-of-domain (`false`). The queries themselves are new — the 30-query set
never existed in the repository and had to be written — but the design is the
one Table I describes, and all 18 answerable queries were verified to be
answerable from the corpus before use.

### Table II — RAG vs. baseline

τ\* is now **0.30**, not 0.45.

| Metric | Old | New |
|---|---:|---:|
| Relevance (overall) | 2.87 / 4.80 | **3.40** / **4.60** |
| Relevance (RAG, answered only) | 3.55 (n=22) | **4.27** (n=22) |
| Faithfulness | 4.70 / 1.77 | **4.63** / **2.37** |
| Hallucination rate (<3) | `[X/22]` / `[Y/30]` | **1/18** / **16/30** |
| Abstain rate | 26.67 % | 🟢 **26.67 %** |

Both hallucination denominators exclude refusals — see Part 3.

### Table III — Retrieval quality

| Metric | Old | New |
|---|---:|---:|
| Hit@k | 0.72 | **1.00** |
| Precision@k | `[A.AA]` | **0.78** |
| MRR@k | `[B.BB]` | **0.94** |

🔴 Hit@k is now saturated, so **the "retrieval ceiling" argument in §V-B no
longer holds** and must be cut. The 0.72 was a symptom of the broken scraper,
not a property of dense retrieval on this corpus.

Something better replaces it: Precision@k = 0.78 means ~2.3 of 3 retrieved
passages are relevant, and that remaining 22 % is exactly where the paper's
one real hallucination comes from. The bottleneck moved from *recall* of
relevant passages to *precision* of the retrieved set.

The sentence "Whatever the abstention mechanism does, it can't push RAG
quality past this ceiling" should go; the surviving point is that none of
these metrics vary with τ, which is still worth stating.

### Table IV — Threshold sweep

Now 16 thresholds. Full table in `docs/results.md` §4; the seven original rows
for direct comparison:

| τ | Rel. | Eff. Rel. | Faith. | Abstain | P | R | F1 | Spec. | Acc. | MCC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.45 | 3.40 | 4.27 | 4.63 | 0.27 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.50 | 3.40 | 4.27 | 4.63 | 0.27 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.55 | 3.40 | 4.27 | 4.63 | 0.27 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.60 | 3.40 | 4.43 | 4.63 | 0.30 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.62 | 3.40 | 4.60 | 4.77 | 0.33 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.65 | 3.27 | 5.00 | 4.77 | 0.43 | 0.89 | 1.00 | 0.94 | 0.94 | 0.96 | 0.92 |
| 0.70 | 2.20 | 5.00 | 5.00 | 0.70 | 0.47 | 1.00 | 0.64 | 0.50 | 0.65 | 0.49 |

🟢 **The τ = 0.70 row reproduces almost exactly** (old: P 0.33, F1 0.50,
Spec 0.11, Acc 0.38, MCC 0.19 — now those are the τ = 0.80 values, with 0.70
landing at P 0.47 / F1 0.64 / MCC 0.49, matching the old τ = 0.60 row). The
decay pattern is real; the corrected corpus shifts it right by about 0.10.

🟡 Add the new low-τ rows (0.10–0.25) — they carry claim #1.
🟡 Add a `Hall.` column: flat at 0.06 through the operating region (see Part 3).

### Table V — Results by category

🟢 Every number reproduces, including in-domain hard relevance = 1.00 and
out-of-domain abstain rate = 1.00. Add a **generator refusals** column (0 for
every category except in-domain hard, which is 4) — that column is the
evidence for claim #3.

| Category | n | Abstain | Gen. refusals | Relevance | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Comparative | 2 | 0.00 | 0 | 5.00 | 0.673 |
| Factual | 9 | 0.00 | 0 | 5.00 | 0.737 |
| Reasoning | 5 | 0.00 | 0 | 5.00 | 0.710 |
| Numeric | 2 | 0.00 | 0 | 5.00 | 0.683 |
| In-domain hard | 4 | 0.00 | **4** | 1.00 | 0.617 |
| Out-of-domain | 8 | 1.00 | 0 | 1.00 | 0.175 |

---

## Part 3 — Two methodological points to add

### A refusal is not a hallucination

The judge scores the **identical string** `"I don't know."` as faithfulness 1
on nine queries and 5 on three others — same model, temperature 0, same
rubric. The rubric defines faithfulness for assertions, and a refusal asserts
nothing, so the judge has no defined behaviour.

Refusals are therefore excluded from both the numerator and the denominator of
the hallucination rate, for RAG and baseline alike. Left in, this
inconsistency alone would report **2** hallucinations instead of **1** — a
100 % overstatement of the paper's headline reliability metric.

Worth a short methods paragraph and a limitation. It is also free evidence for
the paper's own thesis about judge unreliability, requiring no annotation.

### The sweep is derived, not re-run

Retrieval and generation do not depend on τ, so each query is run **once** and
the sweep is computed by masking on cached confidences.

Besides being ~*k*× cheaper, this makes the sweep exact. Re-generating and
re-judging at every threshold — as the original code did — lets judge
nondeterminism move the faithfulness column independently of the variable
under study. Some of the wobble in the draft's Table IV faithfulness column is
almost certainly this.

---

## Part 4 — Replacement material for §V-D

The draft's §V-D is about the wrong failure. Here is the one that actually
occurred, and it is a better example:

> **q07** — *"Which IPL franchise does Virat Kohli play for?"*
> **Answer:** *"Virat Kohli plays for the Mumbai Indians in the IPL."*
> **Confidence:** 0.663 (well above τ\*) · **Faithfulness:** 2 · **Relevance:** 5

Kohli plays for Royal Challengers Bengaluru. Retrieval returned passages about
Rohit Sharma and MS Dhoni — topically adjacent, high-scoring, and without the
answer. The gate saw a confident retrieval and let it through.

This supports a sharper version of the paper's conclusion:

> **The faithfulness gain over the baseline comes from grounding, not from
> abstention.** Among substantive answers the hallucination rate is a flat
> 0.06 (1/18) at *every* threshold in the operating region — abstention never
> improves it. What abstention removes is the set of queries that would have
> been answered ungroundedly; it does not improve the answers that remain, and
> it cannot catch a confident retrieval failure on an in-domain question.

That is a more defensible claim than "abstention reduces hallucination," and
the data supports it directly.

---

## Part 5 — §V-E and Table VI: the honest position

**The human validation described in §V-E was never run in this repository.**
There is no rating data, and there was no script. The reported r = 0.96 /
r = 0.06, MAE 0.29 / 0.79 and exact-match 0.79 / 0.71 cannot be reproduced,
and the Spearman and kappa cells are still `[C.CC]` placeholders.

What now exists:

- `src/judge_validation.py` — MAE, exact match, Pearson *r*, Spearman *ρ* and
  Cohen's quadratic-weighted *κ*, verified against `scipy`/`sklearn` to 1e-9.
- `data/human_ratings.json` — a blank 15-item worksheet (seeded sample, so it
  is reproducible), with `human_relevance` / `human_faithfulness` to fill in.

**Three options, in order of preference:**

1. **Rate the 15 items** (~30 minutes), run the script, report real numbers.
   This is the strongest option and the section then stands up.
2. **Cut §V-E and Table VI**, and move judge reliability to the evidence in
   Part 3 — which is real, reproducible, and needs no annotation.
3. If the original 14 ratings exist somewhere outside this repository, put
   them in `data/human_ratings.json` and the script will reproduce the table
   and fill in the missing ρ and κ cells.

**Do not ship the current numbers.** They have no source.

---

## Part 6 — Limitations section

**Delete:**
- "Two judge models were used out of necessity" — one judge throughout now.

**Keep:** single small domain · out-of-domain queries unambiguous by
construction · uncalibrated confidence · circular faithfulness rubric for
abstentions · small human-validation sample · no reranking or hybrid retrieval.

**Add:**
- The faithfulness rubric is **undefined for refusals**, and the judge is
  demonstrably inconsistent on them (Part 3).
- n = 30, single run, no confidence intervals. One out-of-domain query is
  12.5 % of its class; single-query changes move the rates visibly.
- The wide plateau is a property of *this query set* as much as of the method.
  Out-of-domain queries are wholly unrelated to the IPL, which is why the
  0.37 separation gap exists at all. A query at the corpus's periphery would
  be much harder, and was not tested — this is the single biggest threat to
  the result's generality, and is worth stating plainly rather than burying.

---

## Part 7 — Figures

| Figure | Status |
|---|---|
| Fig. 1 — RAG vs baseline | Regenerated: `results/final_plot.png` |
| Fig. 2 — threshold sweep | **Redrawn**: `results/threshold_plot.png` |
| Fig. 3 — confidence separation | **New**: `results/confidence_plot.png` |

Figure 2 was a single panel with a **twinned y axis** — scores on the left,
abstain rate on the right. That invites the reader to compare crossing points
of curves that share no units, and lets apparent steepness be set by the axis
limits. It is now two stacked panels sharing one x axis: judge scores (1–5)
above, rates and coefficients (0–1) below. It also read `Optimal ≈ 0.55`
hard-coded in the plotting script, which disagreed with the paper's own
τ\* = 0.45; the selected threshold is now read from
`results/selected_threshold.json` so the figure cannot drift from the
selection rule again.

Figure 3 is new and carries claims #1 and #2 — it shows the separation gap
directly, and is probably the most persuasive single image in the paper.

---

## Part 8 — Also worth mentioning in Reproducibility

`select_threshold.py` exists now. The paper cites it; the repository did not
contain it, nor any of the abstention metrics (precision, recall, F1,
specificity, accuracy, MCC), Precision@k, MRR@k, the hallucination rate, the
baseline runner, or the 30-query set. Those are all implemented and wired
together now, and every reported number regenerates from
`results/per_query.json`.

One reproducibility bug is worth a line in the paper, because the draft's
claim of determinism was not true: retrieved passages were deduplicated
through a Python `set`, which destroyed rank order and **varied between
processes**, since string hashing is randomised per interpreter run. MRR@k is
not computable without that fix.
