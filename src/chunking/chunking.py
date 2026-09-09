import json
import re

from src.ingestion.pdf_loader import extract_pages


def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def create_chunks(pages, chunk_size=500, overlap=100):
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
                        "text": current_chunk.strip()
                    })
                    chunk_id += 1

                current_chunk = current_chunk[-overlap:] + " " + sentence

        if current_chunk:
            chunks.append({
                "chunk_id": chunk_id,
                "page_number": page["page_number"],
                "text": current_chunk.strip()
            })
            chunk_id += 1

    return chunks


pages = extract_pages("data/raw/paper_01.pdf")
chunks = create_chunks(pages)

with open("data/processed/paper_01_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

print(f"Saved {len(chunks)} chunks")