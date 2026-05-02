import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def evaluation_prompt(question, context, answer):
    return f"""
You are a strict evaluator.

Scoring rules:
- If the model says "I don't know":
  → Relevance = 1 (it failed to answer)
  → Faithfulness = 5 (it did not hallucinate)

- Relevance:
  1 = did not answer
  3 = partial answer
  5 = fully answers the question

- Faithfulness:
  1 = hallucinated / unsupported
  5 = fully grounded in context

Return ONLY:
Relevance: <number>
Faithfulness: <number>

Question: {question}
Context: {context}
Answer: {answer}
"""
def evaluate_answer(question, context, answer):
    prompt = evaluation_prompt(question, context, answer)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content

def parse_scores(text):
    lines = text.split("\n")
    scores = {}
    
    for line in lines:
        if "Relevance" in line:
            scores["relevance"] = float(line.split(":")[1].strip())
        if "Faithfulness" in line:
            scores["faithfulness"] = float(line.split(":")[1].strip())
    
    return scores

def compute_hit_at_k(retrieved_chunks, keywords):
    for chunk in retrieved_chunks:
        text = chunk.lower()
        if any(keyword.lower() in text for keyword in keywords):
            return 1
    return 0