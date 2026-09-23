"""Split the scraped sections into overlapping character chunks."""

import json

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_FILE, RAW_DATASET_FILE

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def load_data():
    with open(RAW_DATASET_FILE, encoding="utf-8") as f:
        return json.load(f)


def clean_text(text):
    return text.replace("\n", " ").strip()


def create_chunks(data):
    chunks = []
    for item in data:
        for chunk in splitter.split_text(clean_text(item["content"])):
            chunks.append({
                "text": chunk,
                "source": item["page"],
                "section": item["section"],
            })
    return chunks


def save_chunks(chunks):
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    data = load_data()
    chunks = create_chunks(data)
    save_chunks(chunks)
    print(f"Saved {len(chunks)} chunks from {len(data)} sections to {CHUNKS_FILE}")
