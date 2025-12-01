# src/simple_wrappers.py
"""
Simple wrappers for components that might not have all expected methods.
Provides backward compatibility for demo.

Author: Saberzerker
Date: 2025-11-16
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class SimpleEmbeddingModel:
    """Simple wrapper for sentence transformer."""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def encode(self, texts):
        """Encode texts to vectors."""
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts)


def get_embedding_model():
    """Get embedding model instance."""
    return SimpleEmbeddingModel()


class SimpleLocalVDB:
    """Simplified local VDB wrapper."""
    
    def __init__(self):
        self.permanent_count = 300  # We seeded 300 facts
    
    def get_permanent_layer_count(self):
        """Get count of permanent layer vectors."""
        return self.permanent_count
    
    def search(self, query_vector, k=5):
        """Placeholder search - returns empty for now."""
        return []


class SimpleCloudClient:
    """Simplified Qdrant cloud wrapper."""
    
    def __init__(self, url, api_key, collection_name, dimension):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
    
    def get_collection_stats(self):
        """Get collection statistics."""
        info = self.client.get_collection(self.collection_name)
        return {'points_count': info.points_count}
    
    def search(self, query_vector, k=5):
        """Search cloud VDB."""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector,
            limit=k
        )
        return [
            {
                'id': hit.id,
                'score': hit.score,
                'payload': {'text': hit.payload.get('text', '')}
            }
            for hit in results
        ]