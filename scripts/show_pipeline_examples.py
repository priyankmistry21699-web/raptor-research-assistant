"""Show concrete examples from each pipeline step."""
import json, os, pickle, chromadb, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.core.embedding import EmbeddingModel

RAW = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

# ========== STEP 1: Data Ingestion ==========
print("=" * 60)
print("STEP 1: DATA INGESTION & COLLECTION")
print("=" * 60)
meta = json.load(open(os.path.join(RAW, 'papers_metadata_with_id.json'), 'r', encoding='utf-8'))
pdfs = [f for f in os.listdir(RAW) if f.endswith('.pdf')]
print(f"Total PDFs downloaded: {len(pdfs)}")
print(f"Papers with arXiv metadata: {len(meta)}")
print(f"\nExample metadata (1312.4400):")
sample_meta = [p for p in meta if p['arxiv_id'] == '1312.4400'][0]
for k, v in sample_meta.items():
    if k == 'abstract':
        print(f"  {k}: {str(v)[:120]}...")
    else:
        print(f"  {k}: {v}")

# ========== STEP 2: PDF Processing & Text Chunking ==========
print("\n" + "=" * 60)
print("STEP 2: PDF PROCESSING & TEXT CHUNKING")
print("=" * 60)
chunks = []
with open(os.path.join(RAW, 'papers_chunks.jsonl'), 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        chunks.append(json.loads(line))
        if i >= 10000:
            break
total_chunks = sum(1 for _ in open(os.path.join(RAW, 'papers_chunks.jsonl'), 'r', encoding='utf-8'))
papers_in_chunks = set(c['arxiv_id'] for c in chunks)
print(f"Total chunks generated: {total_chunks}")
print(f"Unique papers: 204")

# Show chunks for one specific paper
paper_id = '1706.03762'  # Attention Is All You Need
paper_chunks = [c for c in chunks if c['arxiv_id'] == paper_id]
print(f"\nExample: Paper {paper_id} has {len(paper_chunks)} chunks")
for c in paper_chunks[:3]:
    print(f"  Chunk {c['chunk_index']}: \"{c['text'][:100]}...\"")

# ========== STEP 3: Embedding Generation ==========
print("\n" + "=" * 60)
print("STEP 3: EMBEDDING GENERATION")
print("=" * 60)
emb_count = sum(1 for _ in open(os.path.join(RAW, 'papers_embeddings.jsonl'), 'r', encoding='utf-8'))
print(f"Total embeddings generated: {emb_count}")

# Show one embedding record
with open(os.path.join(RAW, 'papers_embeddings.jsonl'), 'r', encoding='utf-8') as f:
    sample_emb = json.loads(f.readline())
print(f"\nExample embedding record:")
print(f"  arxiv_id: {sample_emb['arxiv_id']}")
print(f"  chunk_index: {sample_emb['chunk_index']}")
print(f"  chunk_text: \"{sample_emb['chunk_text'][:100]}...\"")
print(f"  embedding_vector: [{sample_emb['embedding_vector'][0]:.6f}, {sample_emb['embedding_vector'][1]:.6f}, ... ] (dim={len(sample_emb['embedding_vector'])})")
print(f"  paper_metadata keys: {list(sample_emb['paper_metadata'].keys()) if sample_emb['paper_metadata'] else '{}'}")

# ========== STEP 4: Vector Database ==========
print("\n" + "=" * 60)
print("STEP 4: VECTOR DATABASE (ChromaDB)")
print("=" * 60)
client = chromadb.PersistentClient(path=os.path.join(RAW, 'chroma_db'))
col = client.get_or_create_collection('paper_chunks')
print(f"Collection name: paper_chunks")
print(f"Total documents stored: {col.count()}")

# Sample metadata
sample = col.get(ids=[f'{paper_id}_0', f'{paper_id}_1'], include=['metadatas', 'documents'])
print(f"\nExample records for {paper_id}:")
for i, uid in enumerate(sample['ids']):
    m = sample['metadatas'][i]
    d = sample['documents'][i][:100]
    print(f"  ID: {uid}")
    print(f"  Metadata: {m}")
    print(f"  Document: \"{d}...\"")

# Semantic search demo
print(f"\n--- Semantic Search Demo ---")
embedder = EmbeddingModel()
query = "attention mechanism in transformers"
qvec = embedder.encode(query)
results = col.query(query_embeddings=[qvec], n_results=3, include=['metadatas', 'documents'])
print(f"Query: \"{query}\"")
for i in range(3):
    m = results['metadatas'][0][i]
    d = results['documents'][0][i][:120]
    print(f"  Result {i+1}: [{m['arxiv_id']}] {m.get('paper_title','')} | \"{d}...\"")

# ========== STEP 5: RAPTOR Trees ==========
print("\n" + "=" * 60)
print("STEP 5: RAPTOR HIERARCHICAL INDEX")
print("=" * 60)
trees = [f for f in os.listdir(os.path.join(RAW, 'paper_trees')) if f.endswith('.gpickle')]
print(f"Total trees built: {len(trees)}")

tree_path = os.path.join(RAW, 'paper_trees', f'{paper_id}_tree.gpickle')
with open(tree_path, 'rb') as f:
    G = pickle.load(f)
root = G.nodes['root']
sections = [n for n in G.nodes if G.nodes[n].get('type') == 'section']
chunk_nodes = [n for n in G.nodes if G.nodes[n].get('type') == 'chunk']
print(f"\nExample tree for {paper_id}:")
print(f"  Root title: {root.get('title', '')}")
print(f"  Sections: {len(sections)}")
print(f"  Chunks: {len(chunk_nodes)}")
print(f"  Sample sections:")
for s in sections[:5]:
    n = G.nodes[s]
    children = list(G.successors(s))
    print(f"    {n.get('section_num','')} {n.get('title','')} ({len(children)} chunks)")
