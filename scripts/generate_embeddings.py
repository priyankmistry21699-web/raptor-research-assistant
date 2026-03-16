import os
import sys
import json
import pickle
from tqdm import tqdm
import networkx as nx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.core.embedding import EmbeddingModel

# Path to chunked data
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

CHUNKS_PATH = os.path.join(RAW_DATA_DIR, 'papers_chunks.jsonl')
METADATA_PATH = os.path.join(RAW_DATA_DIR, 'papers_metadata_with_id.json')
EMBEDDINGS_PATH = os.path.join(RAW_DATA_DIR, 'papers_embeddings.jsonl')
TREE_DIR = os.path.join(RAW_DATA_DIR, 'paper_trees')


# Load embedding model
MODEL_NAME = 'all-MiniLM-L6-v2'  # Change to BGE if needed
embedder = EmbeddingModel(MODEL_NAME)

# Load metadata for lookup
with open(METADATA_PATH, 'r', encoding='utf-8') as f:
    paper_metadata = {p['arxiv_id']: p for p in json.load(f)}

# Build fallback title lookup from RAPTOR trees for papers without arXiv metadata
tree_titles = {}
if os.path.isdir(TREE_DIR):
    for fname in os.listdir(TREE_DIR):
        if fname.endswith('_tree.gpickle'):
            aid = fname.replace('_tree.gpickle', '')
            if aid not in paper_metadata:
                tree_path = os.path.join(TREE_DIR, fname)
                with open(tree_path, 'rb') as tf:
                    G = pickle.load(tf)
                title = G.nodes['root'].get('title', '')
                if title:
                    tree_titles[aid] = title
    print(f"Loaded {len(tree_titles)} fallback titles from RAPTOR trees")

# Generate embeddings and store
with open(CHUNKS_PATH, 'r', encoding='utf-8') as f_in, open(EMBEDDINGS_PATH, 'w', encoding='utf-8') as f_out:
    for line in tqdm(f_in, desc='Embedding chunks'):
        rec = json.loads(line)
        chunk_text = rec['text']
        arxiv_id = rec['arxiv_id']
        chunk_index = rec['chunk_index']
        embedding = embedder.encode(chunk_text)
        # Use arXiv metadata if available, otherwise build minimal metadata from tree title
        meta = paper_metadata.get(arxiv_id, {})
        if not meta and arxiv_id in tree_titles:
            meta = {'title': tree_titles[arxiv_id], 'arxiv_id': arxiv_id}
        # Compose output record
        out_rec = {
            'arxiv_id': arxiv_id,
            'chunk_index': chunk_index,
            'chunk_text': chunk_text,
            'embedding_vector': embedding,
            'paper_metadata': meta
        }
        f_out.write(json.dumps(out_rec, ensure_ascii=False) + '\n')
print(f"Embeddings saved to {EMBEDDINGS_PATH}")
