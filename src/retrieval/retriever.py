import json
import faiss

from src.embeddings.embedder import generate_embeddings
from src.retrieval.faiss_index import load_index


with open("data/processed/all_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

index = load_index("data/index/faiss.index")


def retrieve_chunks(query, top_k=5):

    query_embedding = generate_embeddings([
        {"text": query}
    ]).astype("float32")

    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for distance, idx in zip(distances[0], indices[0]):
        results.append({
            "chunk": chunks[idx],
            "score": float(distance)
        })

    return results
