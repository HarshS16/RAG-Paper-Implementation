"""LLM-as-judge scoring and keyword-based retrieval metrics."""

import re

from config import JUDGE_MODEL
from llm import complete


def evaluation_prompt(question, context, answer):
    return f"""You are a strict evaluator.

Score the answer on two axes, each from 1 to 5.

Relevance -- does the answer address the question that was asked?
  1 = does not answer the question at all
  3 = partially answers the question
  5 = fully answers the question

Faithfulness -- is the answer supported by the context provided?
  1 = contradicts the context, or asserts facts absent from it
  3 = partly supported, with some unsupported detail
  5 = fully grounded in the context

Judge faithfulness against the context only. An answer may be factually true
in the wider world and still score 1 for faithfulness if the context does not
support it.

Return ONLY these two lines and nothing else:
Relevance: <number>
Faithfulness: <number>

Question: {question}

Context: {context}

Answer: {answer}"""


def evaluate_answer(question, context, answer):
    return complete(evaluation_prompt(question, context, answer), model=JUDGE_MODEL)


def parse_scores(text):
    """Pull the two scores out of the judge's reply.

    Returns None for a field the judge did not emit, so a malformed reply is
    visible in the results rather than being silently recorded as a zero.
    """
    scores = {"relevance": None, "faithfulness": None}
    for key in scores:
        match = re.search(rf"{key}\s*[:=]\s*([0-5](?:\.\d+)?)", text, re.IGNORECASE)
        if match:
            scores[key] = float(match.group(1))
    return scores


# ------------------------------------------------------------ retrieval
def _contains_keyword(text, keywords):
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def compute_hit_at_k(retrieved_texts, keywords):
    """1 if any retrieved passage contains any ground-truth keyword."""
    if not keywords:
        return None
    return int(any(_contains_keyword(t, keywords) for t in retrieved_texts))


def compute_precision_at_k(retrieved_texts, keywords, k):
    """Fraction of the k retrieved passages containing a ground-truth keyword."""
    if not keywords or k == 0:
        return None
    relevant = sum(1 for t in retrieved_texts if _contains_keyword(t, keywords))
    return relevant / k


def compute_reciprocal_rank(retrieved_texts, keywords):
    """1/rank of the first relevant passage, or 0 if none is relevant.

    Depends on retrieved_texts being in descending score order.
    """
    if not keywords:
        return None
    for rank, text in enumerate(retrieved_texts, start=1):
        if _contains_keyword(text, keywords):
            return 1.0 / rank
    return 0.0
