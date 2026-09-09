import json
import re
from pathlib import Path

from src.ingestion.pdf_loader import extract_pages


def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def create_chunks(pages, paper_name, chunk_size=500, overlap=100):
    chunks = []
    chunk_id = 0

    for page in pages:
        sentences = split_into_sentences(page["text"])
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += " " + sentence

            else:
                if current_chunk:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "page_number": page["page_number"],
                        "paper": paper_name,
                        "text": current_chunk.strip()
                    })
                    chunk_id += 1

                current_chunk = current_chunk[-overlap:] + " " + sentence

        if current_chunk:
            chunks.append({
                "chunk_id": chunk_id,
                "page_number": page["page_number"],
                "paper": paper_name,
                "text": current_chunk.strip()
            })
            chunk_id += 1

    return chunks


# Process every PDF in data/raw/
raw_folder = Path("data/raw")

all_chunks = []
global_chunk_id = 0

for pdf_path in raw_folder.glob("*.pdf"):

    print(f"Processing: {pdf_path.name}")

    pages = extract_pages(pdf_path)

    chunks = create_chunks(
        pages,
        paper_name=pdf_path.name
    )

    for chunk in chunks:
        chunk["chunk_id"] = global_chunk_id
        global_chunk_id += 1

    all_chunks.extend(chunks)


# Save all chunks
with open(
    "data/processed/all_chunks.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)


print(f"\nTotal chunks: {len(all_chunks)}")