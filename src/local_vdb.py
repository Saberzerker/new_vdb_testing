# src/local_vdb.py
"""
Local VDB Interface - Clean API for Three-Tier System

Wrapper around StorageEngine that provides:
- Easy tier separation (permanent vs dynamic)
- Smart neighborhood checking
- Weight management
- Statistics tracking

Author: Saberzerker
Date: 2025-11-17 00:10 UTC
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict

from src.storage_engine import StorageEngine
from src.config import Config

logger = logging.getLogger(__name__)


class LocalVDB:
    """
    User-friendly interface for three-tier local VDB.

    Hides complexity of storage engine.
    Makes it easy to:
    - Search specific tiers
    - Check if neighborhood exists
    - Manage capacity
    - Track weights
    """

    def __init__(self, config: Config = None):
        """Initialize local VDB with storage engine."""
        self.config = config or Config()
        self.storage = StorageEngine(self.config)

        logger.info("[LOCAL VDB] Initialized")
        logger.info(f"[LOCAL VDB] Permanent: {self.get_permanent_count()} vectors")
        logger.info(
            f"[LOCAL VDB] Dynamic: {self.get_dynamic_count()}/{self.storage.dynamic_capacity} vectors"
        )

    # ═══════════════════════════════════════════════════════════
    # TIER 1: PERMANENT LAYER (Kitchen)
    # ═══════════════════════════════════════════════════════════

    def search_permanent(
        self, query_vector: np.ndarray, k: int = 5
    ) -> Tuple[List[str], List[float]]:
        """
        Search only permanent layer.

        Use when:
        - Want privacy-guaranteed results
        - Checking baseline knowledge
        """
        return self.storage.search_permanent(query_vector, k)

    def get_permanent_count(self) -> int:
        """How many vectors in permanent layer?"""
        return self.storage._count_permanent()

    def get_permanent_stats(self) -> Dict:
        """Get permanent layer statistics."""
        return {
            "vector_count": self.get_permanent_count(),
            "read_only": True,
            "layer": "permanent",
        }

    # ═══════════════════════════════════════════════════════════
    # TIER 2: DYNAMIC LAYER (Backpack)
    # ═══════════════════════════════════════════════════════════

    def search_dynamic(
        self, query_vector: np.ndarray, k: int = 5
    ) -> Tuple[List[str], List[float]]:
        """
        Search only dynamic layer.

        Use when:
        - Checking learned knowledge
        - Verifying predictions
        """
        return self.storage.search_dynamic(query_vector, k)

    def insert_dynamic(
        self, vectors: np.ndarray, ids: List[str], metadata: Optional[Dict] = None
    ):
        """
        Insert vectors into dynamic layer.

        Raises:
            ValueError if dynamic is full (must evict first)
        """
        try:
            self.storage.insert_dynamic(vectors, ids, metadata)
            logger.debug(f"[LOCAL VDB] Inserted {len(ids)} vectors to dynamic")
        except ValueError as e:
            logger.error(f"[LOCAL VDB] Insert failed: {e}")
            raise

    def delete_dynamic(self, vec_id: str):
        """Delete vector from dynamic layer."""
        self.storage.delete_dynamic(vec_id)

    def get_dynamic_count(self) -> int:
        """How many vectors in dynamic layer?"""
        return self.storage._count_dynamic()

    def is_dynamic_full(self) -> bool:
        """Is dynamic layer at capacity?"""
        return self.storage.is_dynamic_full()

    def has_dynamic_space(self, n: int) -> bool:
        """Does dynamic have space for n vectors?"""
        return self.storage.has_dynamic_space(n)

    def get_dynamic_stats(self) -> Dict:
        """Get dynamic layer statistics."""
        return self.storage.get_dynamic_stats()

    # ═══════════════════════════════════════════════════════════
    # SMART OPERATIONS
    # ═══════════════════════════════════════════════════════════

    def exists_in_dynamic_neighborhood(
        self, query_vector: np.ndarray, threshold: float = 0.90
    ) -> bool:
        """
        🔍 SMART CHECK: Does similar vector exist in dynamic?

        This is THE KEY innovation!

        Before fetching from cloud:
          if local_vdb.exists_in_dynamic_neighborhood(prediction):
              skip_fetch()  # Save 900ms!

        Args:
            query_vector: Vector to check
            threshold: Similarity threshold (0.90 = 90% similar)

        Returns:
            True if neighborhood exists (skip fetch!)
            False if new neighborhood (go fetch!)
        """
        return self.storage.exists_in_dynamic_neighborhood(query_vector, threshold)

    def get_weakest_dynamic_vector(self) -> Optional[str]:
        """
        Find vector with lowest weight (least stars).

        Used for eviction when dynamic is full.

        Returns:
            Vector ID of weakest vector, or None if empty
        """
        return self.storage.get_weakest_dynamic_vector()

    def update_dynamic_weight(self, vec_id: str, delta: float):
        """
        Update vector weight (add/remove stars).

        Use:
          - After prediction hit: update_weight(id, +1.0)
          - After prediction miss: update_weight(id, -0.5)
        """
        self.storage.update_dynamic_weight(vec_id, delta)

    # ═══════════════════════════════════════════════════════════
    # UNIFIED OPERATIONS
    # ═══════════════════════════════════════════════════════════

    def search(
        self, query_vector: np.ndarray, k: int = 5
    ) -> Tuple[List[str], List[float]]:
        """
        Search BOTH tiers (permanent + dynamic).

        Merges results and returns top-k.
        """
        # Search both
        perm_ids, perm_scores = self.search_permanent(query_vector, k)
        dyn_ids, dyn_scores = self.search_dynamic(query_vector, k)

        # Merge
        all_ids = perm_ids + dyn_ids
        all_scores = perm_scores + dyn_scores

        # Sort by score
        combined = list(zip(all_ids, all_scores))
        combined.sort(key=lambda x: x[1])

        # Return top-k
        top_k = combined[:k]
        return [vid for vid, _ in top_k], [s for _, s in top_k]

    def get_total_vectors(self) -> int:
        """Total vectors across both tiers."""
        return self.get_permanent_count() + self.get_dynamic_count()

    def get_stats(self) -> Dict:
        """
        Get comprehensive statistics.

        Returns:
            Dict with permanent, dynamic, and total stats
        """
        return {
            "permanent_layer_vectors": self.get_permanent_count(),
            "dynamic_layer_vectors": self.get_dynamic_count(),
            "total_vectors": self.get_total_vectors(),
            "dynamic_capacity": self.storage.dynamic_capacity,
            "dynamic_fill_rate": self.get_dynamic_count()
            / self.storage.dynamic_capacity
            * 100,
            "dynamic_stats": self.get_dynamic_stats(),
        }

    def save_state(self):
        """Save dynamic layer to disk."""
        self.storage.save_dynamic_state()
        logger.info("[LOCAL VDB] State saved")
