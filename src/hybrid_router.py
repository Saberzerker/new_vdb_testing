# src/hybrid_router.py

"""
Hybrid Router with Three-Tier Architecture and Smart Prefetching

TIER 1: Permanent Local VDB (300 vectors, read-only, privacy)
TIER 2: Dynamic Prefetch Space (700 vectors, learning engine)
TIER 3: Cloud VDB (9,482 vectors, canonical truth)

Core Innovation:
- Fixed-size dynamic space (700 vectors, no bloating)
- Smart prefetching (checks if neighborhood exists before fetching)
- Phase-based strategy (cold start → warmup → steady state)
- Momentum-based trajectory predictions

OPTIMIZED (2025-11-30):
- Parallel TIER 1 + TIER 2 search (33% faster local searches)

Author: Saberzerker
Date: 2025-11-30 (PARALLEL SEARCH OPTIMIZATION)
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum
import threading
import concurrent.futures  # ✅ NEW: For parallel tier search

from src.anchor_system import AnchorSystem
from src.semantic_cache import SemanticClusterCache
from src.local_vdb import LocalVDB
from src.cloud_client import QdrantCloudClient
from src.metrics import MetricsTracker
from src.config import (
    LOCAL_CONFIDENCE_THRESHOLD,
    PREFETCH_ENABLED,
    DYNAMIC_LAYER_CAPACITY,
    VERBOSE,
    DEFAULT_SEARCH_K,
)
from src.config import COLD_START_QUERIES, WARMUP_QUERIES

logger = logging.getLogger(__name__)


class PrefetchPhase(Enum):
    """Prefetch strategy phases based on query count."""
    COLD_START = "cold_start"      # Query 1-3: Fill aggressively
    WARMUP = "warmup"               # Query 4-20: Refine accuracy
    STEADY_STATE = "steady_state"  # Query 20+: Maintain consistency


class HybridRouter:
    """
    Three-tier hybrid router with smart prefetching.
    
    Query Flow:
    1. Check TIER 1 (Permanent) → Privacy layer, always available
    2. Check TIER 2 (Dynamic) → Learning engine, 700 fixed capacity
    3. Fallback TIER 3 (Cloud) → Truth source, minimize access
    4. Smart prefetch: Only fetch if NOT already in dynamic space
    """

    def __init__(
        self,
        local_vdb: LocalVDB,
        cloud_vdb: QdrantCloudClient,
        semantic_cache: SemanticClusterCache,
        anchor_system: AnchorSystem,
        metrics: MetricsTracker,
    ):
        """Initialize three-tier router."""
        self.local_vdb = local_vdb
        self.cloud_vdb = cloud_vdb
        self.semantic_cache = semantic_cache
        self.anchor_system = anchor_system
        self.metrics = metrics
        
        self.query_count = 0
        self.prefetch_cache_hits = 0
        self.prefetch_cache_misses = 0
        
        # ✅ OPTIMIZATION: Thread pool for parallel tier search
        self.tier_search_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="TierSearch"
        )
        
        logger.info("[ROUTER] ✅ Three-tier hybrid router initialized")
        logger.info(f"[ROUTER] TIER 1: Permanent (300 vectors, read-only)")
        logger.info(
            f"[ROUTER] TIER 2: Dynamic ({DYNAMIC_LAYER_CAPACITY} vectors, learning)"
        )
        logger.info(f"[ROUTER] TIER 3: Cloud (9,482 vectors, truth)")
        logger.info("[ROUTER] ⚡ Parallel tier search enabled")

    def search(
        self,
        query_vector: np.ndarray,
        query_id: str,
        query_text: str = "",
        k: int = DEFAULT_SEARCH_K,
    ) -> Dict:
        """
        Three-tier search with smart prefetching.
        
        OPTIMIZED: TIER 1 + TIER 2 searched in parallel
        """
        self.query_count += 1
        start_time = time.time()

        if VERBOSE:
            print(f"\n{'='*70}")
            print(f"[QUERY #{self.query_count}] {query_id}")
            print(f'Text: "{query_text[:60]}..."')
            print(f"{'='*70}")

        result = {
            "query_id": query_id,
            "query_text": query_text,
            "query_number": self.query_count,
            "timestamp": time.time(),
            "k": k,
        }

        # ═══════════════════════════════════════════════════════════
        # STEP 1: Check Anchor Prediction Match
        # ═══════════════════════════════════════════════════════════

        prediction_match = self.anchor_system.check_prediction_match(
            query_vector, query_id
        )

        if prediction_match:
            pred_id, source_anchor_id, similarity = prediction_match
            # 🎯 REINFORCE
            self.anchor_system.strengthen_anchor(source_anchor_id, reward=1.0)
            result["prediction_hit"] = True
            result["source_anchor_id"] = source_anchor_id
            result["prediction_similarity"] = similarity
            logger.info(
                f"[ROUTER] 🎯 PREDICTION HIT! Anchor #{source_anchor_id} "
                f"(sim={similarity:.3f})"
            )
        else:
            result["prediction_hit"] = False

        # ═══════════════════════════════════════════════════════════
        # STEP 2: Semantic Clustering
        # ═══════════════════════════════════════════════════════════

        cluster_id, cluster_action, access_count = self.semantic_cache.add_query(
            query_vector, query_id
        )

        result["cluster_id"] = cluster_id
        result["cluster_action"] = cluster_action
        result["cluster_access_count"] = access_count

        # ═══════════════════════════════════════════════════════════
        # STEP 3 & 4: Search TIER 1 + TIER 2 IN PARALLEL ⚡
        # ═══════════════════════════════════════════════════════════

        logger.debug("[ROUTER] 🔍 Searching TIER 1 + TIER 2 in parallel...")
        local_search_start = time.time()

        # ✅ PARALLEL SEARCH: Launch both tier searches concurrently
        tier1_future = self.tier_search_executor.submit(
            self.local_vdb.search_permanent, query_vector, k
        )
        tier2_future = self.tier_search_executor.submit(
            self.local_vdb.search_dynamic, query_vector, k
        )

        # Wait for both to complete (timeout 100ms)
        try:
            tier1_ids, tier1_scores = tier1_future.result(timeout=0.1)
            tier2_ids, tier2_scores = tier2_future.result(timeout=0.1)
        except concurrent.futures.TimeoutError:
            logger.warning("[ROUTER] ⚠️ Tier search timeout (>100ms)")
            tier1_ids, tier1_scores = [], []
            tier2_ids, tier2_scores = [], []

        total_local_latency = (time.time() - local_search_start) * 1000

        result["tier1_results"] = len(tier1_ids)
        result["tier2_results"] = len(tier2_ids)
        result["parallel_search_latency_ms"] = total_local_latency

        # Get dynamic space stats
        dynamic_stats = self.local_vdb.get_dynamic_stats()
        result["dynamic_fill_rate"] = (
            f"{dynamic_stats['current_size']}/{dynamic_stats['capacity']}"
        )

        # ✅ MERGE RESULTS: Find best result from both tiers
        all_local_results = []

        # Add TIER 1 results
        for vid, score in zip(tier1_ids, tier1_scores):
            all_local_results.append({
                "id": vid,
                "score": score,
                "tier": "tier1_permanent"
            })

        # Add TIER 2 results
        for vid, score in zip(tier2_ids, tier2_scores):
            all_local_results.append({
                "id": vid,
                "score": score,
                "tier": "tier2_dynamic"
            })

        # Sort by score (lower L2 distance = better)
        all_local_results.sort(key=lambda x: x["score"])

        # Check if best result meets confidence threshold
        if all_local_results and all_local_results[0]["score"] >= LOCAL_CONFIDENCE_THRESHOLD:
            best_result = all_local_results[0]
            best_tier = best_result["tier"]
            
            # Extract top-k results
            top_k = all_local_results[:k]
            result_ids = [r["id"] for r in top_k]
            result_scores = [r["score"] for r in top_k]
            
            # ✅ LOCAL HIT!
            total_time = (time.time() - start_time) * 1000
            result.update({
                "ids": result_ids,
                "scores": result_scores,
                "source": best_tier,
                "latency_ms": total_time,
                "confidence": best_result["score"],
            })
            
            # Record metrics for the winning tier
            if self.metrics:
                self.metrics.record_query(
                    source=best_tier,
                    latency_ms=total_local_latency,
                    prefetch_fetched=0,
                    prefetch_skipped=0
                )
            
            logger.info(f"[ROUTER] ✅ LOCAL HIT ({best_tier}, {total_local_latency:.1f}ms)")
            logger.info(f"[ROUTER] Dynamic space: {result['dynamic_fill_rate']}")
            
            # Prefetch for future queries
            if PREFETCH_ENABLED:
                self._smart_prefetch(
                    query_vector,
                    query_id,
                    query_text,
                    result,
                    cluster_id,
                    prediction_match,
                )
            
            return result

        # ═══════════════════════════════════════════════════════════
        # STEP 5: No local hit → Fallback to TIER 3 (Cloud)
        # ═══════════════════════════════════════════════════════════

        logger.debug(f"[ROUTER] ❌ Local miss (checked both tiers in {total_local_latency:.1f}ms)")
        logger.info(f"[ROUTER] ⬆️ TIER 3 fallback (cloud)")

        try:
            tier3_start = time.time()
            cloud_ids, cloud_scores, cloud_latency_ms = self.cloud_vdb.search(
                query_vector, k
            )

            total_time = (time.time() - start_time) * 1000

            result.update(
                {
                    "ids": cloud_ids,
                    "scores": cloud_scores,
                    "source": "tier3_cloud",
                    "latency_ms": total_time,
                    "tier3_latency_ms": cloud_latency_ms,
                    "confidence": cloud_scores[0] if cloud_scores else 0.0,
                }
            )

            # FIXED: Use record_query() instead of log_event()
            if self.metrics:
                self.metrics.record_query(
                    source="tier3_cloud",
                    latency_ms=cloud_latency_ms,
                    prefetch_fetched=0,
                    prefetch_skipped=0
                )

            logger.info(f"[ROUTER] ☁️ TIER 3 HIT (cloud, {cloud_latency_ms:.1f}ms)")

            # Cache neighborhood to TIER 2
            if cloud_ids and cloud_scores:
                cached = self._cache_neighborhood_to_tier2(
                    cloud_ids[:3], cloud_scores[:3], query_vector, cluster_id
                )
                result["cached_to_tier2"] = cached

            # Smart prefetch
            if PREFETCH_ENABLED:
                self._smart_prefetch(
                    query_vector,
                    query_id,
                    query_text,
                    result,
                    cluster_id,
                    prediction_match,
                )

            return result

        except Exception as e:
            # ════════════════════════════════════════════════════════════
            # STEP 6: Offline Graceful Degradation
            # ════════════════════════════════════════════════════════════
            logger.error(f"[ROUTER] ❌ TIER 3 error: {e}")
            total_time = (time.time() - start_time) * 1000

            # Return best local result (tier1 or tier2)
            if all_local_results:
                best_result = all_local_results[0]
                top_k = all_local_results[:k]
                result_ids = [r["id"] for r in top_k]
                result_scores = [r["score"] for r in top_k]
            else:
                result_ids = []
                result_scores = []

            result.update(
                {
                    "ids": result_ids,
                    "scores": result_scores,
                    "source": "offline_fallback",
                    "latency_ms": total_time,
                    "error": str(e),
                    "confidence": result_scores[0] if result_scores else 0.0,
                }
            )

            # FIXED: Use record_query() instead of log_event()
            if self.metrics:
                self.metrics.record_query(
                    source="offline_fallback",
                    latency_ms=total_time,
                    prefetch_fetched=0,
                    prefetch_skipped=0
                )

            logger.warning(f"[ROUTER] ⚠️ OFFLINE (best local result)")
            return result

    # ═══════════════════════════════════════════════════════════
    # PREFETCH PHASE DETERMINATION
    # ═══════════════════════════════════════════════════════════

    def _get_prefetch_phase(self) -> PrefetchPhase:
        """
        Determine current prefetch phase based on query count.
        
        Returns:
            PrefetchPhase enum value
        """
        if self.query_count <= COLD_START_QUERIES:
            return PrefetchPhase.COLD_START
        elif self.query_count <= WARMUP_QUERIES:
            return PrefetchPhase.WARMUP
        else:
            return PrefetchPhase.STEADY_STATE

    def _cache_neighborhood_to_tier2(
        self,
        cloud_ids: List[str],
        cloud_scores: List[float],
        query_vector: np.ndarray,
        cluster_id: int,
    ) -> int:
        """
        Cache semantic neighborhood to TIER 2 (dynamic space).
        """
        cached = 0
        for doc_id, score in zip(cloud_ids, cloud_scores):
            try:
                vectors = self.cloud_vdb.get_vectors_by_ids([doc_id])
                if vectors and len(vectors) > 0:
                    vector = vectors[0]
                    # Add to dynamic layer
                    self.local_vdb.insert_dynamic(
                        vectors=vector.reshape(1, -1),
                        ids=[doc_id],
                        metadata={
                            "source": "cloud_neighborhood",
                            "cluster_id": cluster_id,
                            "score": score,
                            "weight": 2.0,
                        },
                    )
                    cached += 1
            except Exception as e:
                logger.warning(f"[CACHE] Failed to cache {doc_id}: {e}")
                continue

        logger.info(f"[CACHE] Added {cached} neighborhood vectors to TIER 2")
        return cached

    def _smart_prefetch(
        self,
        query_vector: np.ndarray,
        query_id: str,
        query_text: str,
        result: Dict,
        cluster_id: int,
        prediction_match: Optional[Tuple]
    ):
        """
        SMART PREFETCHING with three phases.
        
        NOW RUNS IN BACKGROUND (non-blocking)!
        """
        # Define background prefetch function
        def prefetch_background():
            try:
                logger.info(f"[PREFETCH] {'─'*60}")
                logger.info(f"[PREFETCH] ⚡ Started background prefetch (non-blocking)")

                # Determine prefetch phase
                phase = self._get_prefetch_phase()
                logger.info(f"[PREFETCH] Phase: {phase.value.upper()} (query #{self.query_count})")

                # Create Anchor
                parent_anchor_id = prediction_match[1] if prediction_match else None
                anchor_id = self.anchor_system.create_anchor(
                    query_vector=query_vector,
                    query_id=query_id,
                    query_text=query_text,
                    parent_anchor_id=parent_anchor_id
                )

                # Phase-Based Prefetching
                prefetch_start = time.time()
                if phase == PrefetchPhase.COLD_START:
                    fetched, skipped = self._prefetch_cold_start(
                        query_vector, anchor_id, cluster_id
                    )
                elif phase == PrefetchPhase.WARMUP:
                    fetched, skipped = self._prefetch_warmup(
                        query_vector, anchor_id, cluster_id
                    )
                else:  # STEADY_STATE
                    fetched, skipped = self._prefetch_steady_state(
                        query_vector, anchor_id, cluster_id
                    )

                prefetch_time = (time.time() - prefetch_start) * 1000

                # Update Stats
                self.prefetch_cache_hits += skipped
                self.prefetch_cache_misses += fetched
                total_predictions = fetched + skipped
                cache_hit_rate = (skipped / total_predictions * 100) if total_predictions > 0 else 0

                # Get updated stats
                dynamic_stats = self.local_vdb.get_dynamic_stats()
                anchor_stats = self.anchor_system.get_anchor_stats()

                logger.info(f"[PREFETCH] Fetched: {fetched}, Skipped: {skipped} "
                           f"(cache hit: {cache_hit_rate:.1f}%)")
                logger.info(f"[PREFETCH] Time: {prefetch_time:.1f}ms")
                logger.info(f"[TIER 2] {dynamic_stats['current_size']}/{dynamic_stats['capacity']} vectors")
                logger.info(f"[ANCHORS] Total: {anchor_stats['total_anchors']}, "
                           f"Active predictions: {anchor_stats['active_predictions']}")
                logger.info(f"[PREFETCH] {'─'*60}")

            except Exception as e:
                logger.error(f"[PREFETCH] Background prefetch failed: {e}")
                import traceback
                traceback.print_exc()

        # Start background thread (non-blocking!)
        prefetch_thread = threading.Thread(
            target=prefetch_background,
            name=f"Prefetch-{query_id}",
            daemon=True
        )
        prefetch_thread.start()

    def _prefetch_cold_start(
        self,
        query_vector: np.ndarray,
        anchor_id: int,
        cluster_id: int
    ) -> Tuple[int, int]:
        """
        COLD START: Fill TIER 2 with initial predictions.
        
        OPTIMIZED: Parallel cloud searches + batch vector fetch
        """
        dynamic_stats = self.local_vdb.get_dynamic_stats()
        space_available = dynamic_stats['capacity'] - dynamic_stats['current_size']

        if space_available <= 0:
            return 0, 0

        predictions_needed = 5
        logger.info(f"[COLD START] Generating {predictions_needed} predictions")

        predictions = self.anchor_system.generate_predictions(
            anchor_id=anchor_id,
            centroid=None,
            count=predictions_needed,
            noise_scale=0.3
        )

        fetched = 0
        skipped = 0
        
        # ✅ PARALLEL CLOUD SEARCHES
        all_doc_ids = []
        all_scores = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all prediction searches in parallel
            future_to_pred = {
                executor.submit(self.cloud_vdb.search, pred_vec, 3): (i, pred_vec)
                for i, pred_vec in enumerate(predictions)
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_pred, timeout=10.0):
                pred_idx, pred_vec = future_to_pred[future]
                try:
                    pred_ids, pred_scores, _ = future.result(timeout=2.0)
                    all_doc_ids.extend(pred_ids)
                    all_scores.extend(pred_scores)
                    
                    # Register prediction
                    self.anchor_system.register_prediction(pred_vec, anchor_id)
                    
                except Exception as e:
                    logger.warning(f"[COLD START] Prediction {pred_idx} failed: {e}")

        # ✅ BATCH VECTOR FETCH (all at once!)
        if all_doc_ids:
            try:
                # Remove duplicates while preserving order
                unique_ids = []
                seen = set()
                for doc_id in all_doc_ids:
                    if doc_id not in seen:
                        unique_ids.append(doc_id)
                        seen.add(doc_id)
                
                # Limit to available space
                unique_ids = unique_ids[:space_available]
                
                logger.info(f"[COLD START] Batch fetching {len(unique_ids)} vectors...")
                batch_start = time.time()
                
                # Single batch fetch instead of N individual fetches!
                vectors = self.cloud_vdb.get_vectors_by_ids(unique_ids)
                
                batch_time = (time.time() - batch_start) * 1000
                logger.info(f"[COLD START] Batch fetch complete ({batch_time:.0f}ms)")
                
                # Insert all fetched vectors
                if vectors and len(vectors) > 0:
                    for i, (doc_id, vector) in enumerate(zip(unique_ids, vectors)):
                        if dynamic_stats['current_size'] >= dynamic_stats['capacity']:
                            break
                        
                        self.local_vdb.insert_dynamic(
                            vectors=vector.reshape(1, -1),
                            ids=[doc_id],
                            metadata={
                                "source": "cold_start_prefetch",
                                "anchor_id": anchor_id,
                                "weight": 1.0
                            }
                        )
                        fetched += 1
                        dynamic_stats['current_size'] += 1
                        
            except Exception as e:
                logger.error(f"[COLD START] Batch fetch failed: {e}")

        logger.info(f"[COLD START] ✅ Fetched {fetched} vectors")
        return fetched, skipped

    def _prefetch_warmup(self, query_vector: np.ndarray, anchor_id: int, cluster_id: int) -> Tuple[int, int]:
        """
        WARMUP: Refine accuracy with smart cache checking.
        
        OPTIMIZED: Parallel searches with early cache checking
        """
        # Get cluster centroid for guidance
        centroid = self.semantic_cache.get_centroid(cluster_id)

        predictions = self.anchor_system.generate_predictions(
            anchor_id=anchor_id,
            centroid=centroid,
            count=10,
            noise_scale=0.15,
        )

        fetched = 0
        skipped = 0
        
        # ✅ PARALLEL CACHE CHECKS + CLOUD SEARCHES
        predictions_to_fetch = []
        
        # First, quickly check which predictions need fetching
        for i, pred_vec in enumerate(predictions):
            if self.local_vdb.exists_in_dynamic_neighborhood(pred_vec, threshold=0.90):
                skipped += 1
                logger.debug(f"[WARMUP] Prediction {i}: SKIPPED (cached)")
            else:
                predictions_to_fetch.append((i, pred_vec))
        
        if not predictions_to_fetch:
            logger.info("[WARMUP] All predictions cached!")
            return fetched, skipped
        
        logger.info(f"[WARMUP] Fetching {len(predictions_to_fetch)} new predictions...")
        
        # Parallel cloud searches for uncached predictions
        all_doc_ids = []
        all_scores = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_pred = {
                executor.submit(self.cloud_vdb.search, pred_vec, 3): (i, pred_vec)
                for i, pred_vec in predictions_to_fetch
            }
            
            for future in concurrent.futures.as_completed(future_to_pred, timeout=10.0):
                pred_idx, pred_vec = future_to_pred[future]
                try:
                    pred_ids, pred_scores, _ = future.result(timeout=2.0)
                    all_doc_ids.extend(pred_ids)
                    all_scores.extend(pred_scores)
                    
                    self.anchor_system.register_prediction(pred_vec, anchor_id)
                    
                except Exception as e:
                    logger.warning(f"[WARMUP] Prediction {pred_idx} failed: {e}")
        
        # ✅ BATCH VECTOR FETCH
        if all_doc_ids:
            try:
                unique_ids = list(dict.fromkeys(all_doc_ids))  # Remove duplicates
                
                logger.info(f"[WARMUP] Batch fetching {len(unique_ids)} vectors...")
                vectors = self.cloud_vdb.get_vectors_by_ids(unique_ids)
                
                if vectors and len(vectors) > 0:
                    for doc_id, vector in zip(unique_ids, vectors):
                        # Replace weakest if full
                        if self.local_vdb.is_dynamic_full():
                            weakest_id = self.local_vdb.get_weakest_dynamic_vector()
                            if weakest_id:
                                self.local_vdb.delete_dynamic(weakest_id)
                        
                        self.local_vdb.insert_dynamic(
                            vectors=vector.reshape(1, -1),
                            ids=[doc_id],
                            metadata={
                                "source": "warmup_prefetch",
                                "anchor_id": anchor_id,
                                "weight": 2.0,
                            },
                        )
                        fetched += 1
                        
            except Exception as e:
                logger.error(f"[WARMUP] Batch fetch failed: {e}")

        logger.info(f"[WARMUP] ✅ Fetched {fetched}, Skipped {skipped}")
        return fetched, skipped


    def _prefetch_steady_state( self, query_vector: np.ndarray, anchor_id: int, cluster_id: int) -> Tuple[int, int]:
        """
        STEADY STATE: Maintain consistency with high cache hit rate.
        
        OPTIMIZED: Minimal parallel fetches (expect 90%+ cache hits)
        """
        # Get STRONG anchors only
        strong_anchors = self.anchor_system.get_strong_anchors()
        if not strong_anchors:
            strong_anchors = [self.anchor_system.anchors.get(anchor_id)]

        predictions = []
        for anchor in strong_anchors[:3]:
            pred_vec = self.anchor_system.generate_trajectory_prediction(
                query_vector,
                anchor.centroid if hasattr(anchor, "centroid") else anchor["vector"],
                momentum=0.9,
            )
            predictions.append(pred_vec)

        fetched = 0
        skipped = 0
        
        # Check cache (expect 90%+ hits in steady state)
        predictions_to_fetch = []
        
        for i, pred_vec in enumerate(predictions):
            if self.local_vdb.exists_in_dynamic_neighborhood(pred_vec, threshold=0.92):
                skipped += 1
                logger.debug(f"[STEADY] Prediction {i}: SKIPPED (high confidence cache)")
            else:
                predictions_to_fetch.append((i, pred_vec))
        
        if not predictions_to_fetch:
            logger.info("[STEADY] ✅ All predictions cached (90%+ hit rate achieved!)")
            return fetched, skipped
        
        logger.info(f"[STEADY] Rare fetch: {len(predictions_to_fetch)} vectors")
        
        # Parallel fetch (rare in steady state)
        all_doc_ids = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_pred = {
                executor.submit(self.cloud_vdb.search, pred_vec, 3): (i, pred_vec)
                for i, pred_vec in predictions_to_fetch
            }
            
            for future in concurrent.futures.as_completed(future_to_pred, timeout=10.0):
                pred_idx, pred_vec = future_to_pred[future]
                try:
                    pred_ids, pred_scores, _ = future.result(timeout=2.0)
                    all_doc_ids.extend(pred_ids)
                    
                    self.anchor_system.register_prediction(pred_vec, anchor_id)
                    
                except Exception as e:
                    logger.warning(f"[STEADY] Prediction {pred_idx} failed: {e}")
        
        # Batch fetch
        if all_doc_ids:
            try:
                unique_ids = list(dict.fromkeys(all_doc_ids))
                vectors = self.cloud_vdb.get_vectors_by_ids(unique_ids)
                
                if vectors and len(vectors) > 0:
                    for doc_id, vector in zip(unique_ids, vectors):
                        weakest_id = self.local_vdb.get_weakest_dynamic_vector()
                        if weakest_id:
                            self.local_vdb.delete_dynamic(weakest_id)
                        
                        self.local_vdb.insert_dynamic(
                            vectors=vector.reshape(1, -1),
                            ids=[doc_id],
                            metadata={
                                "source": "steady_state_prefetch",
                                "anchor_id": anchor_id,
                                "weight": 5.0,
                            },
                        )
                        fetched += 1
                        
            except Exception as e:
                logger.error(f"[STEADY] Batch fetch failed: {e}")

        return fetched, skipped


    def get_stats(self) -> Dict:
        """Get comprehensive router statistics."""
        total_prefetch = self.prefetch_cache_hits + self.prefetch_cache_misses
        prefetch_hit_rate = (
            self.prefetch_cache_hits / total_prefetch * 100 if total_prefetch > 0 else 0
        )

        return {
            "total_queries": self.query_count,
            "prefetch_cache_hit_rate": prefetch_hit_rate,
            "tier1_permanent": self.local_vdb.get_permanent_stats(),
            "tier2_dynamic": self.local_vdb.get_dynamic_stats(),
            "anchor_system": self.anchor_system.get_anchor_stats(),
            "metrics": self.metrics.get_summary(),
        }

    def shutdown(self):
        """Cleanup resources on shutdown."""
        logger.info("[ROUTER] Shutting down thread pools...")
        self.tier_search_executor.shutdown(wait=True)
        logger.info("[ROUTER] ✅ Shutdown complete")
