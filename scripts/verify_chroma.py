import os
import sys
import chromadb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.core.embedding import EmbeddingModel

e = EmbeddingModel()
c = chromadb.PersistentClient(path='data/raw/chroma_db')
col = c.get_or_create_collection('paper_chunks')
q = e.encode('What are the advantages of deep learning?')
r = col.query(query_embeddings=[q], n_results=5, include=['metadatas', 'documents'])

print("Search results with enriched metadata:\n")
for i in range(5):
    meta = r['metadatas'][0][i]
    doc = r['documents'][0][i][:100]
    print(f"{i+1}. [{meta.get('arxiv_id')}] Title: {meta.get('paper_title', 'NO TITLE')}")
    print(f"   Keys: {list(meta.keys())}")
    print(f"   Text: {doc}...")
    print()
