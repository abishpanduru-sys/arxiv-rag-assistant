import json
import faiss
from src.embeddings.embedder import generate_embeddings
from src.retrieval.faiss_index import load_index


# Load chunks
with open("data/processed/paper_01_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Load already-built FAISS index
index = load_index("data/index/faiss.index")


# User query
query = "What is retrieval augmented generation?"


# Embed ONLY the query
query_embedding = generate_embeddings([
    {"text": query}
]).astype("float32")

faiss.normalize_L2(query_embedding)


# Search FAISS
distances, indices = index.search(query_embedding.astype("float32"), 5)


# Display results
for rank, (distance, idx) in enumerate(zip(distances[0], indices[0]), start=1):
    chunk = chunks[idx]

    print(f"\n--- Result {rank} ---")
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"Page: {chunk['page_number']}")
    print(f"Distance: {distance}")
    print(chunk["text"])