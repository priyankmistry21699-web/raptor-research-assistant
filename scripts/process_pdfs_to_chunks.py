import os
import re
import json
import sys
import fitz  # PyMuPDF
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.core.ingestion import RAW_DATA_DIR

CHUNK_SIZE = 400  # tokens (approximate, by words)
CHUNK_OVERLAP = 50  # overlap between chunks

# Helper: simple whitespace tokenizer (can be replaced with tiktoken or nltk)
def tokenize(text):
    return text.split()

def detokenize(tokens):
    return ' '.join(tokens)

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    tokens = tokenize(text)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        chunks.append(detokenize(chunk))
        i += chunk_size - overlap
    return chunks

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def main():
    pdf_dir = os.path.abspath(RAW_DATA_DIR)
    out_path = os.path.join(pdf_dir, "papers_chunks.jsonl")
    chunk_records = []
    for filename in tqdm(os.listdir(pdf_dir)):
        if filename.endswith('.pdf'):
            arxiv_id = re.match(r"([\d.]+)\.pdf", filename)
            if not arxiv_id:
                continue
            arxiv_id = arxiv_id.group(1)
            pdf_path = os.path.join(pdf_dir, filename)
            try:
                text = extract_text_from_pdf(pdf_path)
                # Split into paragraphs (simple split, can be improved)
                paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
                # Chunk each paragraph
                chunk_idx = 0
                for para in paragraphs:
                    chunks = chunk_text(para)
                    for chunk in chunks:
                        record = {
                            "arxiv_id": arxiv_id,
                            "chunk_index": chunk_idx,
                            "text": chunk
                        }
                        chunk_records.append(record)
                        chunk_idx += 1
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
    # Write all chunks to JSONL
    with open(out_path, 'w', encoding='utf-8') as f:
        for rec in chunk_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f"Processed {len(chunk_records)} chunks. Saved to {out_path}")

if __name__ == "__main__":
    main()
