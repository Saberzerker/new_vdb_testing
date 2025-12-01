# src/cloud_client.py
"""
Qdrant Cloud integration for hybrid VDB system.
Handles cloud storage, search, and vector retrieval.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple
import time
from src.config import CLOUD_TIMEOUT_SECONDS
from src.config import *


class QdrantCloudClient:
    """Real Qdrant Cloud client for canonical vector storage."""
    
    def __init__(self, url: str, api_key: str, collection_name: str, dimension: int = None):
        self.collection_name = collection_name
        self.dimension = dimension or VECTOR_DIMENSION  # Use config default
        
        print(f"[QDRANT] Connecting to {url}...")
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=CLOUD_TIMEOUT_SECONDS
        )
        
        self._ensure_collection_exists()
        print(f"[QDRANT] ✅ Connected to collection '{collection_name}'")
    
    def _ensure_collection_exists(self):
        """Create collection if doesn't exist."""
        try:
            self.client.get_collection(self.collection_name)
            print(f"[QDRANT] Collection '{self.collection_name}' exists")
        except Exception:
            print(f"[QDRANT] Creating collection '{self.collection_name}'...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE
                )
            )
            print(f"[QDRANT] Collection created")
    
    def populate_with_documents(self, documents: List[str], embedder: SentenceTransformer):
        """
        Populate cloud with document corpus.
        This is the canonical knowledge base.
        """
        print(f"\n[QDRANT] Populating cloud with {len(documents)} documents...")
        
        # Generate embeddings
        print("[QDRANT] Generating embeddings...")
        embeddings = embedder.encode(documents, show_progress_bar=True)
        
        # Create points
        points = [
            PointStruct(
                id=i,
                vector=embeddings[i].tolist(),
                payload={"text": documents[i], "doc_id": f"doc_{i}"}
            )
            for i in range(len(documents))
        ]
        
        # Upload
        print("[QDRANT] Uploading to cloud...")
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"[QDRANT] ✅ Uploaded {len(documents)} vectors")
        
        # Verify
        collection_info = self.client.get_collection(self.collection_name)
        print(f"[QDRANT] Cloud contains {collection_info.points_count} vectors\n")
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[List[str], List[float], float]:
        """
        Search cloud for similar vectors.
        
        Returns:
            (ids, scores, latency_ms)
        """
        start_time = time.time()
        
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=k
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            ids = [str(result.id) for result in results]
            scores = [result.score for result in results]
            
            return ids, scores, latency_ms
            
        except Exception as e:
            print(f"[QDRANT] Search error: {e}")
            return [], [], 0.0
    
    def get_vectors_by_ids(self, ids: List[str]) -> List[np.ndarray]:
        """
        Get vectors by IDs (batch operation).
        
        OPTIMIZED: Single API call for multiple IDs
        
        Args:
            ids: List of vector IDs to fetch
            
        Returns:
            List of numpy arrays (vectors)
        """
        try:
            # ✅ BATCH RETRIEVE (single API call)
            records = self.client.retrieve(
                collection_name=self.collection_name,
                ids=ids,
                with_vectors=True
            )
            
            vectors = []
            for record in records:
                if record.vector:
                    vectors.append(np.array(record.vector, dtype='float32'))
            
            logger.debug(f"[CLOUD] Batch retrieved {len(vectors)} vectors")
            return vectors
            
        except Exception as e:
            logger.error(f"[CLOUD] Batch retrieve failed: {e}")
            return []
    
    
    def get_collection_stats(self) -> dict:
        """Get cloud collection statistics."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "points_count": info.points_count,
                "status": info.status,
                "vectors_count": info.vectors_count
            }
        except Exception as e:
            return {"error": str(e)}
    
    def clear_collection(self):
        """Clear all vectors (for testing)."""
        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection_exists()
            print("[QDRANT] Collection cleared")
        except Exception as e:
            print(f"[QDRANT] Clear error: {e}")