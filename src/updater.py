# src/updater.py
"""
Cloud-to-Local Cache Updater with Semantic Usefulness Heuristics.

Decides which cloud results should be cached locally based on:
- Confidence scores
- Semantic fit with current context
- Cache capacity

Author: Saberzerker
Date: 2025-11-16
"""

import numpy as np
import logging
from typing import List, Optional

from src.local_vdb import LocalVDB
from src.semantic_cache import SemanticClusterCache
from src.cloud_client import QdrantCloudClient

logger = logging.getLogger(__name__)


class Updater:
    """
    Manages the flow of data from cloud to local cache.
    
    Key Responsibility:
    After a cloud fallback, decide if the result should be added to
    local cache for faster future access.
    """
    
    def __init__(
        self,
        local_vdb: LocalVDB,
        cloud_vdb: QdrantCloudClient,
        semantic_cache: SemanticClusterCache,
        config
    ):
        """
        Initialize updater.
        
        Args:
            local_vdb: Local vector database
            cloud_vdb: Cloud vector database client
            semantic_cache: Semantic clustering manager
            config: Configuration object
        """
        self.local_vdb = local_vdb
        self.cloud_vdb = cloud_vdb
        self.semantic_cache = semantic_cache
        self.config = config
        
        # Heuristics thresholds
        self.min_score = getattr(config, 'MIN_CACHE_SCORE', 0.85)
        self.min_context_similarity = getattr(config, 'MIN_CONTEXT_SIMILARITY', 0.70)
        
        logger.info("[UPDATER] Initialized with semantic usefulness heuristics")
    
    def add_to_local_if_useful(
        self,
        cloud_ids: List[str],
        cloud_scores: List[float],
        query_vector: np.ndarray,
        cluster_id: int,
        anchor_id: Optional[int] = None
    ):
        """
        Decide if cloud results should be cached locally.
        
        Decision Criteria:
        1. Score threshold: Only cache high-confidence results
        2. Semantic fit: Result must fit current semantic context
        3. Capacity: Don't cache if local is full (handled by storage engine)
        
        Args:
            cloud_ids: IDs from cloud search
            cloud_scores: Scores from cloud search
            query_vector: Query embedding
            cluster_id: Current semantic cluster ID
            anchor_id: Optional anchor ID for metadata
        """
        if not cloud_ids or not cloud_scores:
            logger.debug("[UPDATER] No cloud results to cache")
            return
        
        # Heuristic 1: Score threshold
        if cloud_scores[0] < self.min_score:
            logger.debug(f"[UPDATER] Cloud result score too low ({cloud_scores[0]:.3f} < {self.min_score}), not caching")
            return
        
        # Heuristic 2: Semantic fit check
        centroid = self.semantic_cache.get_centroid(cluster_id)
        
        if centroid is not None:
            # Calculate similarity between query and cluster centroid
            similarity = np.dot(query_vector, centroid) / (
                np.linalg.norm(query_vector) * np.linalg.norm(centroid)
            )
            
            if similarity < self.min_context_similarity:
                logger.debug(f"[UPDATER] Cloud result doesn't fit current context "
                           f"(similarity={similarity:.3f} < {self.min_context_similarity})")
                return
        
        # Decision: CACHE IT
        try:
            # Fetch actual vector from cloud
            vectors = self.cloud_vdb.get_vectors_by_ids([cloud_ids[0]])
            
            if vectors:
                # Add to local dynamic layer
                self.local_vdb.insert_vector(
                    vectors=vectors[0],
                    ids=[cloud_ids[0]],
                    layer="dynamic",
                    anchor_id=anchor_id,
                    metadata={
                        "source": "cloud_fallback",
                        "cluster_id": cluster_id,
                        "score": cloud_scores[0]
                    }
                )
                
                logger.info(f"[UPDATER] ✅ Added cloud result {cloud_ids[0]} to local cache "
                          f"(score={cloud_scores[0]:.3f})")
        
        except Exception as e:
            logger.error(f"[UPDATER] Failed to cache cloud result: {e}")