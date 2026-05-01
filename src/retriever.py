import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def load_chunks():
    with open("data/chunks.json") as f:
        return json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_chunks(chunks):
    texts = [c["text"] for c in chunks]
    return model.encode(texts), texts

def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

def retrieve(query, model, index, texts, k=3):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k)
    
    return [texts[i] for i in I[0]]


