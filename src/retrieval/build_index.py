import json

from src.embeddings.embedder import generate_embeddings
from src.retrieval.faiss_index import create_index, save_index


# Load all chunks
with open(
    "data/processed/all_chunks.json",
    "r",
    encoding="utf-8"
) as f:
    chunks = json.load(f)


# Generate embeddings
embeddings = generate_embeddings(chunks)

print(f"Number of chunks: {len(chunks)}")
print(f"Embedding shape: {embeddings.shape}")


# Create FAISS index
index = create_index(embeddings)


# Save FAISS index
save_index(
    index,
    "data/index/faiss.index"
)

print("FAISS index built successfully.")