import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def evaluation_prompt(question, context, answer):
    return f"""
You are a strict evaluator for a research benchmark.

Scoring rules:
- 5 = fully correct, precise, and completely supported by context
- 4 = mostly correct but minor issues
- 3 = partially correct or incomplete
- 2 = mostly incorrect
- 1 = incorrect or hallucinated

IMPORTANT:
- Penalize ANY missing detail
- Penalize ANY unsupported claim
- Do NOT give 5 unless answer is perfect
- Most answers should NOT be 5

Evaluate:

Relevance: Does the answer fully address the question?
Faithfulness: Is every claim supported by the context?

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