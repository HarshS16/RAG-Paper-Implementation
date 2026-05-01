import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def build_prompt(context, query):
    return f"""
Answer the question using ONLY the context below.

If the answer is not in the context, say:
"I don't know based on the provided context."

Do NOT use external knowledge.

Context:
{context}

Question: {query}
Answer:
"""

def generate_answer(context_chunks, query):
    context = "\n".join(context_chunks)

    prompt = build_prompt(context, query)

    response = client.chat.completions.create(
        model = "llama-3.1-8b-instant",  # fast + good
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content


