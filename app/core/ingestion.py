# arXiv data ingestion logic
import os
from typing import List, Dict
import arxiv
import requests

# Configuration
CATEGORIES = ["cs.LG", "cs.AI", "stat.ML", "cs.CV", "cs.CL"]
DATE_FROM = "2017-01-01"
DATE_TO = "2026-03-13"
MAX_RESULTS = 100
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/processed")

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


def fetch_arxiv_papers(
    categories: List[str], max_results: int, date_from: str, date_to: str
) -> List[Dict]:
    query = "cat:(" + " OR ".join(categories) + ")"
    query += f" AND submittedDate:[{date_from} TO {date_to}]"
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    papers = []
    for result in search.results():
        paper = {
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "abstract": result.summary,
            "category": result.primary_category,
            "pdf_url": result.pdf_url,
            "published_date": result.published.strftime("%Y-%m-%d"),
            "arxiv_id": result.get_short_id(),
        }
        papers.append(paper)
    return papers


def download_pdf(pdf_url: str, save_path: str):
    response = requests.get(pdf_url)
    with open(save_path, "wb") as f:
        f.write(response.content)


def save_metadata(papers: List[Dict], save_path: str):
    import json

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
