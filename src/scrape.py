import wikipedia
import json
import os



wiki = wikipedia.Wikipedia(user_agent='RAG-Paper-Bot/1.0 (https://github.com/harsh)', language='en')

IPL_PAGES = [
    "Indian Premier League",
    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bangalore",
    "Virat Kohli",
    "MS Dhoni",
    "Rohit Sharma",
    "IPL auction",
    "IPL playoffs"
]


def extract_sections(page):
    sections = []
    
    for section in page.sections:
        if len(section.text.strip()) > 100:
            sections.append({
                "section": section.title,
                "content": section.text
            })
    
    return sections



def build_dataset():
    dataset = []
    
    for title in IPL_PAGES:
        page = wiki.page(title)
        
        if not page.exists():
            continue
        
        sections = extract_sections(page)
        
        for sec in sections:
            dataset.append({
                "page": title,
                "section": sec["section"],
                "content": sec["content"]
            })


    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data")

    os.makedirs(DATA_PATH, exist_ok=True)

    file_path = os.path.join(DATA_PATH, "raw_dataset.json")

    with open(file_path, "w") as f:
        json.dump(dataset, f, indent=2)

if __name__ == "__main__":
    build_dataset()