import os
import json
import chromadb
from tqdm import tqdm

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
EMBEDDINGS_PATH = os.path.join(RAW_DATA_DIR, 'papers_embeddings.jsonl')
CHROMA_DIR = os.path.join(RAW_DATA_DIR, 'chroma_db')

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection('paper_chunks')

# Count total lines for progress bar
total_lines = 0
with open(EMBEDDINGS_PATH, 'r', encoding='utf-8') as f:
    for _ in f:
        total_lines += 1
print(f"Total records to upsert: {total_lines}")

# Batch upsert for performance
BATCH_SIZE = 500
batch_ids, batch_embeddings, batch_documents, batch_metadatas = [], [], [], []
upserted = 0

with open(EMBEDDINGS_PATH, 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc='Storing in Chroma', total=total_lines):
        rec = json.loads(line)
        uid = f"{rec['arxiv_id']}_{rec['chunk_index']}"
        metadata = rec['paper_metadata'].copy()
        metadata.update({
            'arxiv_id': rec['arxiv_id'],
            'chunk_index': rec['chunk_index']
        })
        # Ensure paper_title is always present in metadata for fast filtering
        if 'title' in metadata:
            metadata['paper_title'] = metadata['title']
        # Ensure metadata values are only str/int/float/bool (ChromaDB requirement)
        metadata = {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))}

        batch_ids.append(uid)
        batch_embeddings.append(rec['embedding_vector'])
        batch_documents.append(rec['chunk_text'])
        batch_metadatas.append(metadata)

        if len(batch_ids) >= BATCH_SIZE:
            collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas
            )
            upserted += len(batch_ids)
            batch_ids, batch_embeddings, batch_documents, batch_metadatas = [], [], [], []

# Flush remaining
if batch_ids:
    collection.upsert(
        ids=batch_ids,
        embeddings=batch_embeddings,
        documents=batch_documents,
        metadatas=batch_metadatas
    )
    upserted += len(batch_ids)

print(f"Upserted {upserted} embeddings in Chroma at {CHROMA_DIR}")
print(f"Collection count: {collection.count()}")