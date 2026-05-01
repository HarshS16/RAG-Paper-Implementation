import json

def load_data():
    with open("data/raw_dataset.json") as f:
        return json.load(f)
    


def clean_text(text):
    return text.replace("\n", " ").strip()


from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)


def create_chunks(data):
    chunks = []
    
    for item in data:
        cleaned = clean_text(item["content"])
        split_texts = splitter.split_text(cleaned)
        
        for chunk in split_texts:
            chunks.append({
                "text": chunk,
                "source": item["page"],
                "section": item["section"]
            })
    
    return chunks



def save_chunks(chunks):
    with open("data/chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)

if __name__ == "__main__":
    data = load_data()
    chunks = create_chunks(data)
    save_chunks(chunks)