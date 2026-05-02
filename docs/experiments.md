# Experiments

This section describes the experimental setup used to evaluate the RAG system.

---

## 1. Objectives

The experiments aim to:

- Compare RAG vs baseline generation
- Evaluate retrieval quality
- Study the effect of threshold-based abstention
- Analyze tradeoffs between coverage and reliability

---

## 2. Query Set

A small set of factual and analytical queries was used, including:

- "Who has won the most IPL titles?"
- "What is IPL auction?"
- "Why has Mumbai Indians been successful in IPL?"

Each query includes expected keywords or ground-truth signals for evaluation.

---

## 3. Baseline Setup

Baseline model:
- Same LLM
- No retrieval context

Purpose:
- Measure hallucination tendency
- Compare grounding improvements from RAG

---

## 4. RAG Setup

- Top-k retrieval (k = 3)
- Context concatenation
- LLM generates answer using retrieved text

---

## 5. Metrics

### Generation Metrics

- Relevance (0–5)
- Faithfulness (0–5)

### Retrieval Metrics

- Hit@k:
  - 1 if correct information appears in retrieved chunks
  - 0 otherwise

### System Metrics

- Abstain Rate:
  - Fraction of queries where model abstains

- Effective Relevance:
  - Relevance computed only for non-abstained responses

---

## 6. Threshold Experiment

We vary the confidence threshold:
thresholds = [0.45, 0.5, 0.55, 0.6, 0.62, 0.65, 0.7]


For each threshold:
- Compute average retrieval score
- Apply abstention rule
- Generate and evaluate responses

---

## 7. Experimental Flow

For each query:

1. Retrieve top-k documents  
2. Compute retrieval confidence  
3. Apply threshold  
4. Generate or abstain  
5. Evaluate output  
6. Store results  

---

## 8. Output

Results are stored in:

- `results/output.json`
- `results/threshold_experiment.json`

These include:
- scores
- retrieved context
- abstention flags
- evaluation outputs

---

## Notes

- Small dataset used to isolate behavior
- Experiments are deterministic given fixed inputs
- Focus is on system behavior rather than large-scale benchmarking