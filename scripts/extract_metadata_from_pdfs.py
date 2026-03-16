

import os
import re
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.core.ingestion import RAW_DATA_DIR
import arxiv

def extract_metadata_from_filename(filename):
    # Extract arXiv ID from filename (e.g., 1706.03762.pdf)
    match = re.match(r"([\d.]+)\.pdf", filename)
    if match:
        return match.group(1)
    return None


def fetch_arxiv_metadata(arxiv_id):
    try:
        search = arxiv.Search(id_list=[arxiv_id])
        for result in search.results():
            return {
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "abstract": result.summary,
                "category": result.primary_category,
                "pdf_url": result.pdf_url,
                "published_date": result.published.strftime("%Y-%m-%d")
            }
    except Exception as e:
        print(f"Failed to fetch arXiv metadata for {arxiv_id}: {e}")
    return None

def main():
    pdf_dir = os.path.abspath(RAW_DATA_DIR)
    metadata_list = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            arxiv_id = extract_metadata_from_filename(filename)
            meta = fetch_arxiv_metadata(arxiv_id)
            if meta:
                metadata = {
                    "title": meta["title"],
                    "authors": meta["authors"],
                    "abstract": meta["abstract"],
                    "category": meta["category"],
                    "pdf_url": meta["pdf_url"],
                    "published_date": meta["published_date"]
                }
                metadata_list.append(metadata)
            else:
                print(f"No metadata found for {arxiv_id}")
    # Save metadata to JSON
    out_path = os.path.join(pdf_dir, "papers_metadata.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)
    print(f"Extracted metadata for {len(metadata_list)} papers. Saved to {out_path}")

if __name__ == "__main__":
    main()
