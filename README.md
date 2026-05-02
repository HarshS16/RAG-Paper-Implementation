# RAG Evaluation and Hallucination Analysis

This repository contains an experimental implementation of a Retrieval-Augmented Generation (RAG) system focused on evaluating retrieval quality, response grounding, and hallucination control using an abstention mechanism.

The goal of this work is not just to build a RAG pipeline, but to study how retrieval confidence affects answer quality and when a model should refuse to answer.

---

## Motivation

LLMs often produce confident but incorrect answers when retrieval is weak or irrelevant. Standard RAG pipelines reduce hallucination, but they still generate responses even when the context is unreliable.

This project explores a simple but practical idea:

> A model should abstain when retrieval confidence is low.

We implement this idea and evaluate how it impacts answer quality and system behavior.

---

## System Overview

The pipeline follows a standard RAG structure with an additional decision layer:

Query → Retriever → Top-k Documents → Confidence Scoring  
→ If confidence ≥ threshold → Generate Answer  
→ If confidence < threshold → Abstain

Abstention output: "I don't know based on the provided context."


---

## Dataset

- Source: Wikipedia (Indian Premier League domain)
- Custom scraped and processed dataset
- Converted into chunked text passages for retrieval

This domain was chosen to:
- keep queries grounded
- allow factual evaluation
- control noise during experimentation

---

## Tech Stack

- Embeddings: `sentence-transformers (all-mpnet-base-v2)`
- Vector Index: FAISS (L2)
- LLM: Groq (LLaMA-based models)
- Evaluation: LLM-as-judge (relevance and faithfulness scoring)

---

## Evaluation Metrics

We evaluate both retrieval and generation:

- **Relevance**: Does the answer address the question?
- **Faithfulness**: Is the answer supported by retrieved context?
- **Hit@k**: Was the correct information retrieved?
- **Abstain Rate**: How often the system refuses to answer
- **Effective Relevance**: Relevance computed only on non-abstained responses

---

## Experiments

### 1. Baseline vs RAG

| Metric        | RAG  | Baseline |
|--------------|------|----------|
| Relevance     | 3.0  | 3.0      |
| Faithfulness  | 5.0  | 3.66     |

Observation:
- RAG significantly improves faithfulness (answers are more grounded)
- Relevance remains similar due to dataset and query scope

---

### 2. Retrieval Quality

Hit@k ≈ 0.66

Interpretation:
- Retrieval succeeds in ~2 out of 3 cases
- Retrieval errors directly impact generation quality

---

### 3. Threshold-Based Abstention

We vary a confidence threshold on retrieved documents:
thresholds = [0.45, 0.5, 0.55, 0.6, 0.62, 0.65, 0.7]


For each threshold:
- If average retrieval score < threshold → abstain
- Else → generate answer

---

## Results

The main result is shown below:

![Threshold Plot](results/threshold_plot.png)

---

## Key Observations

### 1. Coverage vs Reliability Tradeoff

- Lower threshold → more answers, but weaker reliability
- Higher threshold → fewer answers, but safer outputs

---

### 2. Abstention Behavior

| Threshold | Abstain Rate |
|----------|-------------|
| 0.45     | ~0.0        |
| 0.55     | ~0.33       |
| 0.70     | ~0.66       |

As threshold increases, the system becomes more conservative.

---

### 3. Effective Relevance

Effective relevance remains ~5.0 across thresholds.

Interpretation:
- When the model answers, it produces high-quality responses
- Errors are primarily avoided through abstention

---

### 4. Overall Relevance

- Highest at low thresholds (~4.3)
- Drops at higher thresholds (~2.3)

Reason:
- Increasing abstention reduces the number of evaluated answers

---

### 5. Optimal Threshold

Empirically observed:
Optimal threshold ≈ 0.55


This point balances:
- answer coverage
- response quality
- hallucination control

---

## Findings

- RAG improves grounding but is limited by retrieval quality
- Retrieval quality is the primary bottleneck in the pipeline
- Introducing abstention significantly reduces hallucination risk
- There is a clear and measurable tradeoff between:
  - answering more questions
  - answering them reliably

---

## How to Run

### 1. Setup

```bash
git clone <repo-url>
cd <repo>
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Build Dataset

```bash
python src/scrape.py
python src/preprocess.py
```

### 3. Run Experiments

```bash
python src/experiments.py
```

### 4. Analyze Results

```bash
python src/analyze.py
```

### 5. Generate Plot

```bash
python src/plot.py
```

## Project Structure

```
src/
├── scrape.py
├── preprocess.py
├── retriever.py
├── generator.py
├── evaluate.py
├── experiments.py
├── analyze.py
├── plot.py

data/
├── chunks.json
└── raw_dataset.json

results/
└── output.json
```

## Limitations

- Small dataset (single domain)
- Limited query set
- Evaluation relies on LLM scoring (not human-annotated)
- No reranking or hybrid retrieval

## Future Work

- Hybrid retrieval (BM25 + dense)
- Cross-encoder reranking
- Larger and multi-domain datasets
- Human evaluation benchmarks
- Better calibration of abstention thresholds

## Reproducibility

To reproduce results:
1. Build dataset: `python src/preprocess.py`
2. Run experiments: `python src/experiments.py`
3. Generate plots: `python src/plot.py`

All results are deterministic given the same dataset and model configuration.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Harsh Srivastava
https://github.com/harshs16