"""Dense retrieval over the chunked corpus.

The index is faiss.IndexFlatIP over L2-normalised embeddings, so the score
returned is the cosine similarity between query and chunk. It is a similarity,
not a distance: higher is better, and no distance-to-similarity transform is
applied.
"""

import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import CHUNKS_FILE, EMBED_MODEL, TOP_K

model = SentenceTransformer(EMBED_MODEL)


def load_chunks():
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        return json.load(f)


def embed_chunks(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)
    return np.asarray(embeddings, dtype="float32"), chunks


def build_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    return index


def retrieve(query, model, index, chunks, k=TOP_K):
    """Return the top-k chunks in descending score order.

    Rank order is preserved and exact duplicates are dropped in place, so the
    caller can compute rank-sensitive metrics such as MRR@k. Earlier versions
    deduplicated through a set, which destroyed the ranking and varied between
    processes because string hashing is randomised.
    """
    query_embedding = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(np.asarray(query_embedding, dtype="float32"), k)

    results = []
    seen = set()
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        text = chunks[idx]["text"]
        if text in seen:
            continue
        seen.add(text)
        results.append({
            "text": text,
            "score": float(score),
            "source": chunks[idx]["source"],
            "section": chunks[idx]["section"],
        })

    return results


def build_pipeline():
    """Convenience helper: load chunks, embed, index. Returns (model, index, chunks)."""
    chunks = load_chunks()
    embeddings, chunks = embed_chunks(chunks)
    index = build_index(embeddings)
    return model, index, chunks
