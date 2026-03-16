"""Quick script to gather example data for the walkthrough."""
import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === STEP 1: Metadata ===
print("=" * 60)
print("STEP 1: Data Ingestion & Collection")
print("=" * 60)
with open('data/raw/papers_metadata_with_id.json', 'r', encoding='utf-8') as f:
    all_meta = json.load(f)
    meta = [p for p in all_meta if p.get('arxiv_id') == '1706.03762']

if meta:
    m = meta[0]
    print("Metadata for 1706.03762:")
    print("  arxiv_id:", m.get('arxiv_id'))
    print("  title:", m.get('title'))
    authors = m.get('authors', [])
    print("  authors:", authors[:3], "..." if len(authors) > 3 else "")
    print("  category:", m.get('category'))
    print("  pdf_url:", m.get('pdf_url'))
    print("  published_date:", m.get('published_date'))

pdf_path = 'data/raw/1706.03762.pdf'
print("  PDF exists on disk:", os.path.exists(pdf_path))
if os.path.exists(pdf_path):
    size = os.path.getsize(pdf_path)
    print(f"  PDF size: {size / 1024:.0f} KB")

# === STEP 2: Chunks ===
print()
print("=" * 60)
print("STEP 2: PDF Processing & Text Chunking")
print("=" * 60)
chunks = []
with open('data/raw/papers_chunks.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        rec = json.loads(line)
        if rec['arxiv_id'] == '1706.03762':
            chunks.append(rec)
print(f"Total chunks for 1706.03762: {len(chunks)}")
print()
print("Sample chunks:")
for i in [0, 5, 100, len(chunks)-1]:
    if i < len(chunks):
        c = chunks[i]
        text = c['text'][:120].replace('\n', ' ')
        print(f"  chunk_{c['chunk_index']}: {text}...")
print()

# === STEP 3: Embeddings ===
print("=" * 60)
print("STEP 3: Embedding Generation")
print("=" * 60)
emb_count = 0
sample_emb = None
with open('data/raw/papers_embeddings.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        rec = json.loads(line)
        if rec['arxiv_id'] == '1706.03762':
            emb_count += 1
            if emb_count == 1:
                sample_emb = rec
print(f"Total embeddings for 1706.03762: {emb_count}")
if sample_emb:
    vec = sample_emb['embedding_vector']
    print(f"  Embedding dimension: {len(vec)}")
    print(f"  First 5 values: {vec[:5]}")
    print(f"  chunk_text preview: {sample_emb['chunk_text'][:100]}...")
    pm = sample_emb.get('paper_metadata', {})
    print(f"  paper_metadata keys: {list(pm.keys())}")

# === STEP 4: Vector DB ===
print()
print("=" * 60)
print("STEP 4: Vector Database (ChromaDB)")
print("=" * 60)
import chromadb
chroma_dir = 'data/raw/chroma_db'
client = chromadb.PersistentClient(path=chroma_dir)
collection = client.get_or_create_collection('paper_chunks')
total = collection.count()
print(f"Total documents in Chroma: {total}")

# Count docs for our paper
results = collection.get(where={"arxiv_id": "1706.03762"}, limit=1, include=["metadatas"])
# Use query to count
from app.core.embedding import EmbeddingModel
embedder = EmbeddingModel()
qvec = embedder.encode("attention mechanism")
paper_results = collection.query(
    query_embeddings=[qvec],
    n_results=3,
    where={"arxiv_id": "1706.03762"},
    include=["documents", "metadatas", "distances"]
)
print(f"Sample search within 1706.03762 for 'attention mechanism':")
for i, doc in enumerate(paper_results['documents'][0]):
    meta = paper_results['metadatas'][0][i]
    dist = paper_results['distances'][0][i]
    print(f"  Result {i+1} (distance={dist:.4f}): {doc[:100]}...")

# === STEP 5: RAPTOR Tree ===
print()
print("=" * 60)
print("STEP 5: RAPTOR Hierarchical Index")
print("=" * 60)
from app.core.raptor_index import load_tree, get_tree_structure, get_context_for_chunk

G = load_tree('1706.03762')
print(f"Tree nodes: {len(G.nodes)}")
print(f"Tree edges: {len(G.edges)}")

# Count by type
types = {}
for n in G.nodes:
    t = G.nodes[n].get('type', '?')
    types[t] = types.get(t, 0) + 1
print(f"Node types: {types}")

struct = get_tree_structure('1706.03762')
print(f"Tree levels: {struct['tree_levels']}")
print()
print("Full tree hierarchy:")
print(f"ROOT: {struct['title']}")
for topic in struct['topics']:
    scount = topic['section_count']
    print(f"  TOPIC: {topic['title']} ({scount} sections)")
    for sec in topic['sections']:
        ccount = sec['chunk_count']
        print(f"    SECTION {sec['section_num']}: {sec['title']} ({ccount} chunks)")

# Show context for a specific chunk
print()
chunk_nodes = [n for n in G.nodes if G.nodes[n].get('type') == 'chunk']
ctx = get_context_for_chunk(G, chunk_nodes[100])
chunk_text = G.nodes[chunk_nodes[100]].get('text', '')[:100]
print(f"Context for chunk_100:")
print(f"  Paper: {ctx['paper_title']}")
print(f"  Topic: {ctx['topic']}")
print(f"  Section: {ctx['section_title']} (num: {ctx['section_num']})")
print(f"  Chunk text: {chunk_text}...")
