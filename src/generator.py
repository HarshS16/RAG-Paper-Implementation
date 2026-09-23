"""Answer generation, with and without retrieved context."""

from config import GENERATOR_MODEL
from llm import complete


def build_prompt(context, query):
    return f"""You are answering questions using the provided context.

Rules:
- Use the context as the primary source
- If the answer is clearly present, answer confidently
- If the answer is partially present, try to infer carefully
- Only say "I don't know" if the context has NO relevant information
- Answer in at most three sentences

Context:
{context}

Question: {query}

Answer:"""


def build_baseline_prompt(query):
    """No-context baseline: the same generator, asked the question directly."""
    return f"""Answer the following question.

Answer in at most three sentences.

Question: {query}

Answer:"""


def generate_answer(context_chunks, query):
    context = "\n".join(context_chunks)
    return complete(build_prompt(context, query), model=GENERATOR_MODEL)


def generate_baseline_answer(query):
    return complete(build_baseline_prompt(query), model=GENERATOR_MODEL)
