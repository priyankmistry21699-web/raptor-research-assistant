# Script to ingest papers from arXiv
import os
import sys
import json

# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.core.ingestion import fetch_arxiv_papers, download_pdf, save_metadata, CATEGORIES, MAX_RESULTS, DATE_FROM, DATE_TO, RAW_DATA_DIR, PROCESSED_DATA_DIR

def main():
	print("Fetching arXiv papers...")
	papers = fetch_arxiv_papers(CATEGORIES, MAX_RESULTS, DATE_FROM, DATE_TO)
	print(f"Fetched {len(papers)} papers. Downloading PDFs...")
	for paper in papers:
		pdf_url = paper["pdf_url"]
		arxiv_id = paper["arxiv_id"]
		pdf_path = os.path.join(RAW_DATA_DIR, f"{arxiv_id}.pdf")
		try:
			download_pdf(pdf_url, pdf_path)
			print(f"Downloaded {arxiv_id}")
		except Exception as e:
			print(f"Failed to download {arxiv_id}: {e}")
	metadata_path = os.path.join(PROCESSED_DATA_DIR, "papers_metadata.json")
	save_metadata(papers, metadata_path)
	print(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
	main()
