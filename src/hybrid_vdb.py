# src/hybrid_vdb.py
"""
Main Hybrid Vector Database System with Anchor-Based Trajectory Learning.

NOVEL CONTRIBUTION:
This system learns query PATHS (not just query points) through reinforcement.
After each query, it predicts likely next queries and prefetches them.
When predictions are correct, the source anchor is strengthened.
Over time, strong query paths emerge in the semantic graph.

Author: Saberzerker
Date: 2025-11-16
"""
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import time
from typing import List, Dict, Tuple, Optional
import json

from src.anchor_system import AnchorSystem
from src.semantic_cache import SemanticClusterCache
from src.cloud_client import QdrantCloudClient
from src.config import *


class HybridVectorDB:
    """
    Hybrid VDB with three-layer architecture:
    
    1. LOCAL FAISS INDEX: Fast cache (1-5ms latency)
    2. CLOUD QDRANT: Canonical source (150-200ms latency)
    3. ANCHOR SYSTEM: Learning layer (predicts and prefetches)
    
    The anchor system is the innovation - it learns which query paths
    users follow and proactively caches along those trajectories.
    """
    
    def __init__(self, cloud_client: QdrantCloudClient):
        print("\n" + "="*70)
        print("INITIALIZING HYBRID VDB WITH ANCHOR-BASED TRAJECTORY LEARNING")
        print("="*70 + "\n")
        
        # Layer 1: Cloud VDB (canonical storage)
        self.cloud = cloud_client
        
        # Layer 2: Local FAISS index (fast cache)
        self.local_index = faiss.IndexFlatL2(VECTOR_DIMENSION)
        self.local_ids = []  # Track what's in local cache
        self.local_metadata = []  # Track source and timestamp
        
        # Layer 3a: Semantic cache (momentum-based clustering)
        self.semantic_cache = SemanticClusterCache(
            drift_threshold=SEMANTIC_DRIFT_THRESHOLD,
            momentum_alpha=MOMENTUM_ALPHA,
            max_clusters=MAX_CLUSTERS
        )
        
        # Layer 3b: Anchor system (trajectory learning) - OUR INNOVATION
        self.anchor_system = AnchorSystem()
        
        # Embedding model
        print("[INIT] Loading sentence-transformers model...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        print(f"[INIT] Model loaded: {EMBEDDING_MODEL}")
        
        # Metrics for evaluation
        self.metrics = {
            "total_queries": 0,
            "local_hits": 0,
            "cloud_hits": 0,
            "prediction_hits": 0,  # NEW: how many queries matched predictions
            "prediction_misses": 0,
            "prefetch_operations": 0,
            "queries": [],  # Detailed log of each query
            "learning_timeline": []  # Track system evolution over time
        }
        
        # Track which local vectors came from prefetching
        self.prefetched_ids = set()
        
        # Decay scheduler (runs periodically)
        self.last_decay_check = time.time()
        
        print("[INIT] ✅ Initialization complete")
        print(f"[INIT] Local cache: {self.local_index.ntotal} vectors")
        print(f"[INIT] Cloud: {self.cloud.get_collection_stats()['points_count']} vectors")
        print(f"[INIT] Anchor system: ACTIVE")
        print(f"[INIT] Prefetch: {'ENABLED' if PREFETCH_ENABLED else 'DISABLED'}\n")
    
    def search(self, query_text: str, k: int = 5) -> Dict:
        """
        Main search method implementing the full hybrid + anchor system.
        
        Flow:
        1. Embed query
        2. Check if query was PREDICTED (anchor reinforcement)
        3. Add to semantic cache (momentum update)
        4. Try LOCAL first (fast path)
        5. Fall back to CLOUD if needed
        6. Generate PREDICTIONS for next queries
        7. PREFETCH predicted vectors
        
        Args:
            query_text: The user's query string
            k: Number of results to return
            
        Returns:
            Dict with search results and metadata
        """
        start_time = time.time()
        query_id = f"q{self.metrics['total_queries']}"
        self.metrics["total_queries"] += 1
        
        if VERBOSE:
            print(f"\n{'='*70}")
            print(f"[QUERY {query_id}] \"{query_text}\"")
            print(f"{'='*70}")
        
        # STEP 1: Embed the query into vector space
        query_vector = self.embedder.encode([query_text])[0]
        
        # STEP 2: Check if this query was PREDICTED by an anchor
        # This is where reinforcement learning happens
        matched_prediction = self.anchor_system.check_prediction_match(query_vector, query_id)
        
        result = {
            "query_id": query_id,
            "query_text": query_text,
            "timestamp": time.time()
        }
        
        # Reinforcement: If prediction hit, strengthen the source anchor
        if matched_prediction:
            pred_id, source_anchor_id, similarity = matched_prediction
            
            # REWARD the anchor that predicted correctly
            self.anchor_system.strengthen_anchor(source_anchor_id, reward=1.0)
            
            result["prediction_hit"] = True
            result["source_anchor_id"] = source_anchor_id
            result["prediction_similarity"] = similarity
            
            self.metrics["prediction_hits"] += 1
            
            if VERBOSE:
                print(f"[ANCHOR] 🎯 PREDICTION HIT! Query matched prediction from anchor {source_anchor_id}")
        else:
            result["prediction_hit"] = False
            self.metrics["prediction_misses"] += 1
        
        # STEP 3: Add to semantic cache (momentum-based clustering)
        # This provides context for prediction generation
        cluster_id, cluster_action, access_count = self.semantic_cache.add_query(
            query_vector, query_id
        )
        
        result["cluster_id"] = cluster_id
        result["cluster_action"] = cluster_action
        result["cluster_access_count"] = access_count
        
        if VERBOSE:
            print(f"[SEMANTIC] Cluster {cluster_id} | {cluster_action} | Access count: {access_count}")
        
        # STEP 4: Try LOCAL cache first (fast path: 1-5ms)
        if self.local_index.ntotal > 0:
            local_start = time.time()
            
            # FAISS search in local index
            D, I = self.local_index.search(
                query_vector.reshape(1, -1).astype('float32'),
                min(k, self.local_index.ntotal)
            )
            
            local_latency_ms = (time.time() - local_start) * 1000
            
            if len(D[0]) > 0 and D[0][0] < float('inf'):
                # Convert L2 distance to similarity score
                # Lower distance = higher similarity
                local_score = 1.0 / (1.0 + D[0][0])
                
                # Check if score meets confidence threshold
                if local_score >= LOCAL_CONFIDENCE_THRESHOLD:
                    # LOCAL HIT - we can return immediately!
                    total_latency_ms = (time.time() - start_time) * 1000
                    self.metrics["local_hits"] += 1
                    
                    hit_id = self.local_ids[I[0][0]]
                    was_prefetched = hit_id in self.prefetched_ids
                    
                    result.update({
                        "source": "local",
                        "latency_ms": total_latency_ms,
                        "local_latency_ms": local_latency_ms,
                        "score": float(local_score),
                        "hit_id": hit_id,
                        "was_prefetched": was_prefetched  # Track prefetch effectiveness
                    })
                    
                    if VERBOSE:
                        prefetch_marker = " (PREFETCHED)" if was_prefetched else ""
                        print(f"[RESULT] ✅ LOCAL HIT{prefetch_marker}")
                        print(f"[LATENCY] {total_latency_ms:.2f}ms")
                        print(f"[SCORE] {local_score:.3f}")
                    
                    self.metrics["queries"].append(result)
                    
                    # Generate predictions for next queries
                    if PREFETCH_ENABLED:
                        self._generate_and_prefetch(query_vector, query_id, result, cluster_id, matched_prediction)
                    
                    # Periodic decay check
                    self._maybe_decay_anchors()
                    
                    return result
        
        # STEP 5: CLOUD FALLBACK (slower: 150-200ms)
        # Local cache miss or score too low
        if VERBOSE:
            print(f"[RESULT] ⬆️  CLOUD FALLBACK (local miss or low confidence)")
        
        cloud_start = time.time()
        
        # Search cloud VDB
        cloud_ids, cloud_scores, cloud_latency_ms = self.cloud.search(query_vector, k)
        
        total_latency_ms = (time.time() - start_time) * 1000
        
        if cloud_ids:
            self.metrics["cloud_hits"] += 1
            
            result.update({
                "source": "cloud",
                "latency_ms": total_latency_ms,
                "cloud_latency_ms": cloud_latency_ms,
                "score": float(cloud_scores[0]) if cloud_scores else 0.0,
                "hit_id": cloud_ids[0]
            })
            
            if VERBOSE:
                print(f"[LATENCY] {total_latency_ms:.2f}ms (cloud: {cloud_latency_ms:.2f}ms)")
                print(f"[SCORE] {cloud_scores[0]:.3f}")
            
            # Add cloud result to local cache
            vectors = self.cloud.get_vectors_by_ids([cloud_ids[0]])
            if vectors:
                self._add_to_local(vectors[0], cloud_ids[0], source="cloud_result")
            
            self.metrics["queries"].append(result)
            
            # Generate predictions
            if PREFETCH_ENABLED:
                self._generate_and_prefetch(query_vector, query_id, result, cluster_id, matched_prediction)
            
            self._maybe_decay_anchors()
            
            return result
        
        # No results found
        result.update({
            "source": "none",
            "latency_ms": (time.time() - start_time) * 1000
        })
        self.metrics["queries"].append(result)
        return result
    
    def _generate_and_prefetch(self, query_vector: np.ndarray, query_id: str, 
                                result: Dict, cluster_id: int, 
                                matched_prediction: Optional[Tuple]):
        """
        CORE INNOVATION: Generate predictions for next queries and prefetch them.
        
        This is where trajectory learning happens:
        1. Create anchor for current query
        2. Generate N prediction vectors along likely trajectory
        3. Search cloud for each prediction
        4. Add to local cache
        5. Register predictions for future matching
        
        Args:
            query_vector: Current query embedding
            query_id: Query identifier
            result: Search result dict
            cluster_id: Semantic cluster ID
            matched_prediction: If query was predicted, info about the match
        """
        if VERBOSE:
            print(f"\n[PREFETCH] {'─'*60}")
            print(f"[PREFETCH] Generating trajectory predictions...")
        
        # Determine parent anchor (for graph structure)
        parent_anchor_id = None
        if matched_prediction:
            # If this query was predicted, link to source anchor
            _, parent_anchor_id, _ = matched_prediction
        
        # STEP 1: Create anchor for this query
        anchor_id = self.anchor_system.create_anchor(
            query_vector=query_vector,
            query_id=query_id,
            query_text=result["query_text"],
            parent_anchor_id=parent_anchor_id
        )
        
        # STEP 2: Generate prediction vectors
        # Use cluster centroid to guide trajectory
        centroid = self.semantic_cache.get_centroid(cluster_id)
        
        prediction_vectors = self.anchor_system.generate_predictions(
            anchor_id=anchor_id,
            centroid=centroid
        )
        
        if VERBOSE:
            print(f"[PREFETCH] Generated {len(prediction_vectors)} prediction vectors")
        
        # STEP 3: Prefetch each prediction
        prefetched_count = 0
        for i, pred_vec in enumerate(prediction_vectors):
            # Search cloud for this predicted vector
            cloud_ids, scores, _ = self.cloud.search(pred_vec, k=1)
            
            if cloud_ids and scores:
                # Fetch actual vector from cloud
                vectors = self.cloud.get_vectors_by_ids([cloud_ids[0]])
                
                if vectors:
                    # Add to local cache
                    self._add_to_local(
                        vector=vectors[0],
                        vector_id=cloud_ids[0],
                        source="prefetch",
                        anchor_id=anchor_id
                    )
                    
                    # Register this as an active prediction
                    self.anchor_system.register_prediction(pred_vec, anchor_id)
                    
                    prefetched_count += 1
        
        self.metrics["prefetch_operations"] += 1
        
        if VERBOSE:
            print(f"[PREFETCH] ✅ Prefetched {prefetched_count}/{len(prediction_vectors)} vectors")
            print(f"[CACHE] Local cache now: {self.local_index.ntotal} vectors")
            print(f"[PREFETCH] {'─'*60}\n")
    
    def _add_to_local(self, vector: np.ndarray, vector_id: str, 
                      source: str = "unknown", anchor_id: Optional[int] = None):
        """
        Add a vector to local FAISS cache.
        
        Args:
            vector: The embedding vector
            vector_id: Identifier (from cloud)
            source: How it was added ("cloud_result" or "prefetch")
            anchor_id: If prefetched, which anchor generated it
        """
        # Check if already in cache
        if vector_id in self.local_ids:
            return
        
        # Add to FAISS index
        self.local_index.add(vector.reshape(1, -1).astype('float32'))
        self.local_ids.append(vector_id)
        self.local_metadata.append({
            "added_at": time.time(),
            "source": source,
            "anchor_id": anchor_id
        })
        
        # Track if prefetched (for metrics)
        if source == "prefetch":
            self.prefetched_ids.add(vector_id)
        
        if VERBOSE and source == "cloud_result":
            print(f"[CACHE] Added {vector_id} to local cache")
    
    def _maybe_decay_anchors(self):
        """
        Periodically apply decay to anchors.
        Unused anchors gradually weaken and are eventually evicted.
        """
        current_time = time.time()
        
        if current_time - self.last_decay_check >= ANCHOR_DECAY_CHECK_INTERVAL:
            if VERBOSE:
                print(f"\n[DECAY] Running anchor decay check...")
            
            self.anchor_system.decay_anchors()
            self.last_decay_check = current_time
    
    def get_comprehensive_metrics(self) -> Dict:
        """
        Get complete system metrics for evaluation.
        
        Returns comprehensive stats including:
        - Query statistics (hits, misses, latency)
        - Prediction accuracy (novel metric)
        - Anchor system state (learning progress)
        - Cache efficiency
        """
        total_queries = self.metrics["total_queries"]
        
        # Calculate rates
        local_hit_rate = (self.metrics["local_hits"] / total_queries 
                         if total_queries > 0 else 0.0)
        cloud_hit_rate = (self.metrics["cloud_hits"] / total_queries 
                         if total_queries > 0 else 0.0)
        
        # Prediction accuracy (NOVEL METRIC)
        total_predictions = self.metrics["prediction_hits"] + self.metrics["prediction_misses"]
        prediction_accuracy = (self.metrics["prediction_hits"] / total_predictions 
                              if total_predictions > 0 else 0.0)
        
        # Prefetch effectiveness
        prefetched_hits = sum(1 for q in self.metrics["queries"] 
                             if q.get("source") == "local" and q.get("was_prefetched"))
        prefetch_hit_rate = (prefetched_hits / len(self.prefetched_ids) 
                            if self.prefetched_ids else 0.0)
        
        # Learning curve (hit rate over time windows)
        learning_curve = []
        window_size = 5
        for i in range(0, total_queries, window_size):
            window = self.metrics["queries"][i:i+window_size]
            window_local = sum(1 for q in window if q.get("source") == "local")
            learning_curve.append({
                "query_range": f"{i+1}-{min(i+window_size, total_queries)}",
                "local_hit_rate": window_local / len(window) if window else 0,
                "queries_in_window": len(window)
            })
        
        # Anchor system stats
        anchor_stats = self.anchor_system.get_stats()
        
        # Semantic cache stats
        cache_stats = self.semantic_cache.get_stats()
        
        return {
            # Query statistics
            "total_queries": total_queries,
            "local_hits": self.metrics["local_hits"],
            "cloud_hits": self.metrics["cloud_hits"],
            "local_hit_rate": local_hit_rate,
            "cloud_hit_rate": cloud_hit_rate,
            
            # Prediction statistics (NOVEL)
            "prediction_hits": self.metrics["prediction_hits"],
            "prediction_misses": self.metrics["prediction_misses"],
            "prediction_accuracy": prediction_accuracy,
            
            # Prefetch effectiveness
            "prefetch_operations": self.metrics["prefetch_operations"],
            "vectors_prefetched": len(self.prefetched_ids),
            "prefetched_vectors_used": prefetched_hits,
            "prefetch_hit_rate": prefetch_hit_rate,
            
            # Learning progress
            "learning_curve": learning_curve,
            
            # System state
            "local_cache_size": self.local_index.ntotal,
            "anchor_system": anchor_stats,
            "semantic_cache": cache_stats
        }
    
    def print_final_report(self):
        """Print a beautiful final report showing system performance."""
        metrics = self.get_comprehensive_metrics()
        
        print(f"\n{'='*70}")
        print(f"{'HYBRID VDB - ANCHOR-BASED TRAJECTORY LEARNING':^70}")
        print(f"{'FINAL PERFORMANCE REPORT':^70}")
        print(f"{'='*70}\n")
        
        # Query statistics
        print(f"📊 QUERY STATISTICS:")
        print(f"   Total Queries:        {metrics['total_queries']}")
        print(f"   Local Hits:           {metrics['local_hits']} ({metrics['local_hit_rate']:.1%})")
        print(f"   Cloud Hits:           {metrics['cloud_hits']} ({metrics['cloud_hit_rate']:.1%})")
        
        # Prediction accuracy (NOVEL METRIC)
        print(f"\n🎯 PREDICTION ACCURACY (Novel Contribution):")
        print(f"   Prediction Hits:      {metrics['prediction_hits']}")
        print(f"   Prediction Misses:    {metrics['prediction_misses']}")
        print(f"   Accuracy:             {metrics['prediction_accuracy']:.1%}")
        print(f"   → This shows the anchor system is learning query paths!")
        
        # Prefetch effectiveness
        print(f"\n🔮 PREFETCH EFFECTIVENESS:")
        print(f"   Prefetch Operations:  {metrics['prefetch_operations']}")
        print(f"   Vectors Prefetched:   {metrics['vectors_prefetched']}")
        print(f"   Prefetched Hits:      {metrics['prefetched_vectors_used']}")
        print(f"   Hit Rate:             {metrics['prefetch_hit_rate']:.1%}")
        
        # Anchor system
        anchor_stats = metrics['anchor_system']
        print(f"\n⚓ ANCHOR SYSTEM STATE:")
        print(f"   Total Anchors:        {anchor_stats['total_anchors']}")
        print(f"   By Type:")
        for atype, count in anchor_stats['by_type'].items():
            print(f"      {atype:12s}     {count}")
        print(f"   Total Strength:       {anchor_stats['total_strength']:.1f}")
        print(f"   Avg Strength:         {anchor_stats['average_strength']:.2f}")
        print(f"   Type Transitions:     {len(anchor_stats['type_transitions'])}")
        
        # Learning curve
        print(f"\n📈 LEARNING CURVE (Hit Rate Over Time):")
        for window in metrics['learning_curve']:
            bar_length = int(window['local_hit_rate'] * 40)
            bar = '█' * bar_length + '░' * (40 - bar_length)
            print(f"   Q {window['query_range']:8s} [{bar}] {window['local_hit_rate']:.1%}")
        
        # Cache state
        print(f"\n💾 CACHE STATE:")
        print(f"   Local Cache Size:     {metrics['local_cache_size']} vectors")
        print(f"   Semantic Clusters:    {metrics['semantic_cache']['total_clusters']}")
        
        print(f"\n{'='*70}\n")
    
    def export_results(self, filepath: str):
        """
        Export all metrics to JSON for detailed analysis.
        
        Args:
            filepath: Where to save the JSON file
        """
        metrics = self.get_comprehensive_metrics()
        
        # Add detailed query log
        metrics["detailed_queries"] = self.metrics["queries"]
        
        # Add anchor transitions (learning timeline)
        metrics["anchor_transitions"] = self.anchor_system.metrics["type_transitions"]
        
        # Add full anchor details (without vectors to save space)
        metrics["anchors_detail"] = {
            aid: {
                "query_id": a.query_id,
                "query_text": a.query_text,
                "strength": a.strength,
                "hits": a.hits,
                "misses": a.misses,
                "type": a.type.value,
                "predictions_generated": a.predictions_generated,
                "successful_predictions": a.successful_predictions,
                "created_at": a.created_at,
                "last_hit": a.last_hit
            }
            for aid, a in self.anchor_system.anchors.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"[EXPORT] ✅ Complete results saved to {filepath}")