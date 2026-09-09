import json
from sentence_transformers import SentenceTransformer

from src.retrieval.faiss_index import create_index, save_index


with open("data/processed/paper_01_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [chunk["text"] for chunk in chunks]
embeddings = model.encode(texts)

index = create_index(embeddings)

save_index(index, "data/index/faiss.index")

print(f"Indexed {len(chunks)} chunks")