import os
import sys
import chromadb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.core.embedding import EmbeddingModel

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
CHROMA_DIR = os.path.join(RAW_DATA_DIR, 'chroma_db')

# ✅ Fix 1: PersistentClient reads from disk correctly
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection('paper_chunks')

embedder = EmbeddingModel()

def semantic_search(query, top_k=5, output_path=None, preview_len=200):
    # ✅ Fix 2: encode returns a list already
    query_vec = embedder.encode(query)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=['documents', 'metadatas']
    )
    output_lines = []
    for i, doc in enumerate(results['documents'][0]):
        meta = results['metadatas'][0][i]
        title = meta.get('paper_title', meta.get('title', ''))
        # ✅ Fix 4: preview_len parameter
        line = f"Rank {i+1} | arXiv: {meta['arxiv_id']} | Title: {title}\nChunk: {doc[:preview_len]}...\n"
        print(line)
        output_lines.append(line)
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(output_lines)

if __name__ == "__main__":
    count = collection.count()
    print(f"Chroma collection contains {count} documents.")
    # ✅ Fix 3: early exit if empty
    if count == 0:
        print("Collection is empty. Run the ingestion script first.")
        sys.exit(1)

    query = "transformer architecture for natural language processing"
    print(f"Semantic search for: {query}\n")
    output_file = os.path.join(os.path.dirname(__file__), 'semantic_search_output.txt')
    semantic_search(query, output_path=output_file)
    print(f"\nResults saved to {output_file}")