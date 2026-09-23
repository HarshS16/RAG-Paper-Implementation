"""Scrape the IPL Wikipedia corpus into data/raw_dataset.json.

Sections are collected recursively: on Wikipedia a heading such as "History"
or "Rivalries" usually holds no prose of its own, and everything substantive
sits one or two levels down, so a top-level-only walk discards most of the
page. The lead summary is kept as its own section, and boilerplate headings
(references, external links, ...) are skipped.

Titles are resolved to their canonical form before scraping, and a page that
redirects onto one already collected is skipped -- without that guard
"IPL playoffs" silently redirects to "Indian Premier League" and the corpus
ends up holding the same page twice.
"""

import json
import os

import wikipediaapi

from config import (
    DATA_DIR,
    IPL_PAGES,
    MIN_SECTION_CHARS,
    RAW_DATASET_FILE,
    SKIP_SECTIONS,
)

wiki = wikipediaapi.Wikipedia(
    user_agent="RAG-Paper-Bot/1.0 (https://github.com/harshs16)",
    language="en",
)


def extract_sections(page):
    """Depth-first walk of the section tree, keeping sections with prose."""
    sections = []

    if page.summary and len(page.summary.strip()) > MIN_SECTION_CHARS:
        sections.append({"section": "Summary", "content": page.summary})

    def visit(section):
        if section.title.strip().lower() in SKIP_SECTIONS:
            return
        text = section.text.strip()
        if len(text) > MIN_SECTION_CHARS:
            sections.append({"section": section.title, "content": text})
        for child in section.sections:
            visit(child)

    for section in page.sections:
        visit(section)

    return sections


def build_dataset():
    dataset = []
    seen_titles = {}

    for title in IPL_PAGES:
        print(f"Fetching: {title}")
        try:
            page = wiki.page(title)
        except Exception as e:
            print(f"  Error fetching '{title}': {e}")
            continue

        if not page.exists():
            print(f"  SKIPPED: page '{title}' does not exist on Wikipedia.")
            continue

        canonical = page.title
        if canonical in seen_titles:
            print(f"  SKIPPED: '{title}' redirects to '{canonical}', "
                  f"already collected via '{seen_titles[canonical]}'.")
            continue
        seen_titles[canonical] = title

        if canonical != title:
            print(f"  (redirects to '{canonical}')")

        sections = extract_sections(page)
        print(f"  {len(sections)} sections kept")

        for sec in sections:
            dataset.append({
                "page": canonical,
                "section": sec["section"],
                "content": sec["content"],
            })

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(RAW_DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(dataset)} sections from {len(seen_titles)} pages "
          f"to {RAW_DATASET_FILE}")


if __name__ == "__main__":
    build_dataset()
