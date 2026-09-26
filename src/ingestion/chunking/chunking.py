import argparse
import json
import re
from pathlib import Path

from src.ingestion.pdf_loader import extract_pages

# Common academic abbreviations that end in a period but do NOT mark a
# real sentence boundary. Without this, a naive regex split would break
# "et al. Smith found..." into two fake sentences: "et al." and "Smith
# found...", or "Fig. 3 shows..." into "Fig." and "3 shows...".
ABBREVIATIONS = {
    "e.g.", "i.e.", "et al.", "etc.", "vs.", "fig.", "figs.", "eq.",
    "eqs.", "ref.", "refs.", "no.", "nos.", "dr.", "mr.", "mrs.", "ms.",
    "prof.", "approx.", "cf.",
}

# Chunks shorter than this are almost always junk (page-header/footer
# artifacts, stray page numbers, etc.) and shouldn't be embedded/indexed.
MIN_CHUNK_LENGTH = 40


def split_into_sentences(text):
    """
    Split text into sentences, merging fragments back together when the
    split happened right after a known abbreviation rather than a real
    sentence boundary.
    """
    raw_parts = re.split(r'(?<=[.!?])\s+', text)

    sentences = []
    buffer = ""

    for part in raw_parts:
        buffer = f"{buffer} {part}".strip() if buffer else part

        words = buffer.split()
        last_word = words[-1].lower() if words else ""
        last_two_words = " ".join(words[-2:]).lower() if len(words) >= 2 else ""

        if last_word in ABBREVIATIONS or last_two_words in ABBREVIATIONS:
            # Don't finalize the sentence yet -- keep merging with the
            # next part, since this period didn't end a real sentence.
            continue

        sentences.append(buffer)
        buffer = ""

    if buffer:
        sentences.append(buffer)

    return [s.strip() for s in sentences if s.strip()]


def flatten_pages_to_sentences(pages):
    """
    Flatten a document's pages into a single ordered list of
    (sentence, page_number) tuples. This lets chunking flow continuously
    across page boundaries (fixing the lost-continuity flaw) while still
    tracking which page each sentence came from, so chunks can cite an
    accurate page number.
    """
    sentence_page_pairs = []

    for page in pages:
        sentences = split_into_sentences(page["text"])
        for sentence in sentences:
            sentence_page_pairs.append((sentence, page["page_number"]))

    return sentence_page_pairs


def _take_sentence_overlap(chunk_sentences, overlap_chars):
    """
    Return the trailing whole sentences from chunk_sentences whose
    combined length is within overlap_chars, WITHOUT ever cutting a
    sentence (let alone a word) in half. Character-slicing overlap text
    (the old approach) can start mid-word; this keeps overlap readable.
    """
    overlap_sentences = []
    running_length = 0

    for sentence in reversed(chunk_sentences):
        if running_length + len(sentence) > overlap_chars and overlap_sentences:
            break
        overlap_sentences.insert(0, sentence)
        running_length += len(sentence)

    return overlap_sentences


def create_chunks(pages, paper_name, chunk_size=500, overlap=100):
    sentence_page_pairs = flatten_pages_to_sentences(pages)

    chunks = []
    chunk_id = 0

    current_sentences = []
    current_pages = []
    current_length = 0

    def flush_chunk():
        nonlocal chunk_id
        text = " ".join(current_sentences).strip()
        if len(text) < MIN_CHUNK_LENGTH:
            return  # skip junk fragments (e.g. stray headers/footers)

        chunks.append({
            "chunk_id": chunk_id,
            "page_number": current_pages[0],
            "page_number_end": current_pages[-1],
            "paper": paper_name,
            "text": text,
        })
        chunk_id += 1

    for sentence, page_number in sentence_page_pairs:
        if current_length + len(sentence) <= chunk_size or not current_sentences:
            current_sentences.append(sentence)
            current_pages.append(page_number)
            current_length += len(sentence)
        else:
            flush_chunk()

            # Carry whole trailing sentences forward as overlap, instead
            # of blind character-slicing that can cut mid-word.
            overlap_sentences = _take_sentence_overlap(current_sentences, overlap)
            overlap_pages = current_pages[-len(overlap_sentences):] if overlap_sentences else []

            current_sentences = overlap_sentences + [sentence]
            current_pages = overlap_pages + [page_number]
            current_length = sum(len(s) for s in current_sentences)

    if current_sentences:
        flush_chunk()

    return chunks


def process_all_papers(raw_folder="data/raw", output_path="data/processed/all_chunks.json",
                        rebuild=False):
    """
    Process PDFs in raw_folder into chunks and save them to output_path.

    By default (rebuild=False) this is INCREMENTAL: papers already
    present in output_path (matched by "paper" filename) are skipped
    entirely, and newly-found papers are chunked and APPENDED with
    chunk_ids continuing from the current max. This is required for
    build_index.py's incremental indexing to be safe -- if chunk_ids
    were reassigned from scratch on every run, adding one new paper
    could renumber every existing chunk and silently desync the FAISS
    index (which stores vectors by insertion position) from this file.

    Pass rebuild=True to reprocess everything from scratch (e.g. after
    changing chunk_size/overlap, or fixing an extraction bug).
    """
    raw_folder = Path(raw_folder)
    output_path = Path(output_path)

    existing_chunks = []
    already_processed_papers = set()

    if not rebuild and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing_chunks = json.load(f)
        already_processed_papers = {c["paper"] for c in existing_chunks}

    next_chunk_id = (max((c["chunk_id"] for c in existing_chunks), default=-1) + 1)

    new_chunks = []
    pdf_paths = sorted(raw_folder.glob("*.pdf"))
    papers_to_process = [p for p in pdf_paths if p.name not in already_processed_papers]

    if not rebuild and already_processed_papers:
        skipped = len(pdf_paths) - len(papers_to_process)
        print(f"Skipping {skipped} already-processed paper(s); "
              f"{len(papers_to_process)} new paper(s) to chunk.")

    for pdf_path in papers_to_process:

        print(f"Processing: {pdf_path.name}")

        pages = extract_pages(pdf_path)
        if not pages:
            continue  # extract_pages already logged why it was skipped

        chunks = create_chunks(pages, paper_name=pdf_path.name)

        for chunk in chunks:
            chunk["chunk_id"] = next_chunk_id
            next_chunk_id += 1

        new_chunks.extend(chunks)

    all_chunks = existing_chunks + new_chunks

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\nNew chunks this run: {len(new_chunks)}")
    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chunk downloaded papers for the RAG pipeline."
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Reprocess every PDF from scratch instead of only new ones "
             "(use after changing chunk_size/overlap or fixing extraction).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_all_papers(rebuild=args.rebuild)