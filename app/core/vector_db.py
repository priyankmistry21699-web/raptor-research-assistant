"""
Vector DB interface for RAPTOR Research Assistant.
Wraps ChromaDB for semantic search over paper chunks.
"""

import os
from typing import List, Dict, Any, Optional
import chromadb

# Default Chroma path — can be overridden
DEFAULT_CHROMA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "raw", "chroma_db"
)
COLLECTION_NAME = "paper_chunks"


class VectorDB:
    def __init__(self, chroma_dir: str = None):
        self.chroma_dir = chroma_dir or DEFAULT_CHROMA_DIR
        self.client = chromadb.PersistentClient(path=self.chroma_dir)
        self.collection = self.client.get_or_create_collection(COLLECTION_NAME)

    def count(self) -> int:
        return self.collection.count()

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search by embedding vector. Returns ranked results."""
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        items = []
        for i in range(len(results["ids"][0])):
            items.append(
                {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "arxiv_id": results["metadatas"][0][i].get("arxiv_id", ""),
                    "chunk_index": results["metadatas"][0][i].get("chunk_index", 0),
                }
            )
        return items

    def search_by_paper(
        self,
        query_embedding: List[float],
        arxiv_id: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search within a specific paper."""
        return self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            where={"arxiv_id": arxiv_id},
        )

    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single document by ID."""
        result = self.collection.get(ids=[doc_id], include=["documents", "metadatas"])
        if not result["ids"]:
            return None
        return {
            "id": result["ids"][0],
            "text": result["documents"][0],
            "metadata": result["metadatas"][0],
        }

    def upsert_chunks(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        batch_size: int = 500,
    ) -> int:
        """Upsert chunks with embeddings into ChromaDB. Returns count upserted."""
        upserted = 0
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            self.collection.upsert(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end],
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )
            upserted += batch_end - i
        return upserted
