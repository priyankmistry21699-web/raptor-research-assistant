import os
import json
import re

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
CHUNKS_PATH = os.path.join(RAW_DATA_DIR, 'papers_chunks.jsonl')
METADATA_PATH = os.path.join(RAW_DATA_DIR, 'papers_metadata.json')
OUT_PATH = os.path.join(RAW_DATA_DIR, 'papers_metadata_with_id.json')

def extract_arxiv_id_from_url(url):
    # e.g. https://arxiv.org/pdf/1312.4400v3 -> 1312.4400
    m = re.search(r'arxiv.org/pdf/([\d.]+)', url)
    if m:
        return m.group(1)
    return None

def main():
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    for paper in papers:
        arxiv_id = extract_arxiv_id_from_url(paper.get('pdf_url', ''))
        if arxiv_id:
            paper['arxiv_id'] = arxiv_id
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    print(f"Added arxiv_id to metadata. Output: {OUT_PATH}")

if __name__ == "__main__":
    main()
