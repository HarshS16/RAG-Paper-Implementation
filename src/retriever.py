import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from config import EMBED_MODEL, CHUNKS_FILE, TOP_K

def load_chunks():
    with open(CHUNKS_FILE) as f:
        return json.load(f)

model = SentenceTransformer(EMBED_MODEL)
def embed_chunks(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings, chunks

def build_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    return index


def retrieve(query, model, index, chunks, k=TOP_K):
    query_embedding = model.encode([query], normalize_embeddings=True)
    D, I = index.search(query_embedding, k)

    results = []
    for score, idx in zip(D[0], I[0]):
        results.append({
            "text": chunks[idx]["text"],
            "score": float(score)
        })

    return results