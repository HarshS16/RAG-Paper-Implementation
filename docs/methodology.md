# Methodology

This project implements a Retrieval-Augmented Generation (RAG) pipeline with an additional focus on evaluating retrieval quality and controlling hallucinations through abstention.

---

## 1. System Overview

The pipeline consists of three main components:

1. Retriever  
2. Generator  
3. Evaluation Module  

An additional decision layer is introduced between retrieval and generation to determine whether the model should answer or abstain.

---

## 2. Data Collection and Preprocessing

- Data source: Wikipedia (Indian Premier League domain)
- Raw pages are scraped and stored as structured text
- Text is split into smaller chunks to improve retrieval granularity

Chunking ensures:
- Better semantic matching
- Reduced noise in retrieved context
- Improved embedding quality

---

## 3. Embedding and Retrieval

We use a dense retrieval approach:

- Model: `all-mpnet-base-v2` (SentenceTransformers)
- Index: FAISS (L2 distance)

Steps:
1. Convert each chunk into a vector embedding
2. Store embeddings in FAISS index
3. Encode query into vector space
4. Retrieve top-k most similar chunks

---

## 4. Retrieval Scoring

Each retrieved document is associated with a similarity score.

We compute:

Average Score = mean(similarity scores of top-k results)

This score is used as a proxy for retrieval confidence.

---

## 5. Abstention Mechanism

A threshold-based decision rule is introduced:

- If average retrieval score < threshold → abstain
- If average retrieval score ≥ threshold → generate answer

Abstention response:
"I don't know based on the provided context."

This prevents the model from generating unsupported or hallucinated responses.

---

## 6. Generation

- Model: Groq-hosted LLaMA models
- Input:
  - Retrieved context
  - User query

The model is prompted to:
- Use only the provided context
- Avoid unsupported claims

---

## 7. Evaluation

We use an LLM-as-a-judge approach to evaluate responses.

Each answer is scored on:

- Relevance (0–5)
- Faithfulness (0–5)

Scores are extracted programmatically and stored for analysis.

---

## 8. Pipeline Summary

Query  
→ Retrieve top-k documents  
→ Compute confidence score  
→ Apply threshold  
→ Generate or abstain  
→ Evaluate output  

---

## Design Choices

- Dense retrieval chosen over keyword search for semantic matching
- Threshold-based abstention for simplicity and interpretability
- LLM-based evaluation for scalable automated scoring

---

## Limitations

- Retrieval confidence is approximated using similarity scores
- No reranking or hybrid retrieval used
- Evaluation depends on LLM judgment, not human annotation