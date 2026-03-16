"""
Fast metadata update for ChromaDB.
Enriches existing chunk records with paper_title and metadata from RAPTOR trees,
WITHOUT re-computing embeddings.
"""
import os
import pickle
import chromadb
from tqdm import tqdm

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
CHROMA_DIR = os.path.join(RAW_DATA_DIR, 'chroma_db')
TREE_DIR = os.path.join(RAW_DATA_DIR, 'paper_trees')

# Connect to ChromaDB
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection('paper_chunks')
print(f"Collection count: {collection.count()}")

# Load all paper titles and metadata from RAPTOR trees
paper_info = {}
for fname in os.listdir(TREE_DIR):
    if fname.endswith('_tree.gpickle'):
        arxiv_id = fname.replace('_tree.gpickle', '')
        tree_path = os.path.join(TREE_DIR, fname)
        with open(tree_path, 'rb') as f:
            G = pickle.load(f)
        root = G.nodes['root']
        title = root.get('title', '')
        metadata = root.get('metadata', {})
        paper_info[arxiv_id] = {'title': title, 'metadata': metadata}

print(f"Loaded info for {len(paper_info)} papers from RAPTOR trees")
papers_with_title = sum(1 for v in paper_info.values() if v['title'])
papers_with_meta = sum(1 for v in paper_info.values() if v['metadata'])
print(f"  Papers with title: {papers_with_title}")
print(f"  Papers with full metadata: {papers_with_meta}")

# Update per-paper (204 updates instead of iterating all 148K records)
updated = 0
for arxiv_id, info in tqdm(paper_info.items(), desc='Updating papers'):
    # Get all chunks for this paper
    results = collection.get(
        where={'arxiv_id': arxiv_id},
        include=['metadatas']
    )
    if not results['ids']:
        continue

    ids_to_update = []
    metas_to_update = []
    for i, uid in enumerate(results['ids']):
        meta = results['metadatas'][i].copy()
        # Add paper_title
        if info['title']:
            meta['paper_title'] = info['title']
        # Add primitive metadata fields from tree
        for k, v in info['metadata'].items():
            if isinstance(v, (str, int, float, bool)) and k not in meta:
                meta[k] = v
        ids_to_update.append(uid)
        metas_to_update.append(meta)

    # Batch update all chunks for this paper at once
    collection.update(ids=ids_to_update, metadatas=metas_to_update)
    updated += len(ids_to_update)

print(f"\nDone! Updated metadata for {updated} records across {len(paper_info)} papers")

# Verify with a sample
sample = collection.peek(3)
print("\nSample metadata after update:")
for i, uid in enumerate(sample['ids']):
    meta = sample['metadatas'][i]
    print(f"  {uid}: paper_title='{meta.get('paper_title', 'N/A')}', keys={list(meta.keys())}")
