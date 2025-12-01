# src/scheduler.py
"""
Background Scheduler for Maintenance Tasks.

Runs periodic jobs:
- Dynamic layer compaction
- Anchor decay
- Cluster eviction
- Prediction expiry

Author: Saberzerker
Date: 2025-11-16
"""

import time
import threading
import logging
from typing import Optional

from src.local_vdb import LocalVDB
from src.anchor_system import AnchorSystem
from src.semantic_cache import SemanticClusterCache
from src.config import (
    COMPACTION_INTERVAL_SECONDS,
    ANCHOR_DECAY_CHECK_INTERVAL,
    CLUSTER_TTL_SECONDS
)
# Add this alias
# COMPACTION_INTERVAL_SECONDS = COMPACTION_INTERVAL

logger = logging.getLogger(__name__)


class BackgroundScheduler:
    """
    Manages background maintenance jobs for the hybrid VDB system.
    
    Runs in separate threads to avoid blocking main query path.
    """
    
    def __init__(
        self,
        local_vdb: LocalVDB,
        anchor_system: AnchorSystem,
        semantic_cache: SemanticClusterCache
    ):
        """
        Initialize scheduler with system components.
        
        Args:
            local_vdb: Local vector database
            anchor_system: Anchor-based learning system
            semantic_cache: Semantic clustering manager
        """
        self.local_vdb = local_vdb
        self.anchor_system = anchor_system
        self.semantic_cache = semantic_cache
        
        self.running = False
        self.threads = []
        
        logger.info("[SCHEDULER] Initialized")
    
    def start(self):
        """
        Start all background jobs.
        
        Creates separate threads for:
        - Compaction job (every COMPACTION_INTERVAL_SECONDS)
        - Anchor decay job (every ANCHOR_DECAY_CHECK_INTERVAL)
        - Cluster eviction job (every CLUSTER_TTL_SECONDS / 2)
        - Prediction expiry job (every 60 seconds)
        """
        if self.running:
            logger.warning("[SCHEDULER] Already running")
            return
        
        self.running = True
        
        # Job 1: Dynamic layer compaction
        compaction_thread = threading.Thread(
            target=self._compaction_job,
            name="CompactionJob",
            daemon=True
        )
        compaction_thread.start()
        self.threads.append(compaction_thread)
        
        # Job 2: Anchor decay
        decay_thread = threading.Thread(
            target=self._anchor_decay_job,
            name="AnchorDecayJob",
            daemon=True
        )
        decay_thread.start()
        self.threads.append(decay_thread)
        
        # Job 3: Cluster eviction
        cluster_thread = threading.Thread(
            target=self._cluster_eviction_job,
            name="ClusterEvictionJob",
            daemon=True
        )
        cluster_thread.start()
        self.threads.append(cluster_thread)
        
        # Job 4: Prediction expiry
        prediction_thread = threading.Thread(
            target=self._prediction_expiry_job,
            name="PredictionExpiryJob",
            daemon=True
        )
        prediction_thread.start()
        self.threads.append(prediction_thread)
        
        logger.info(f"[SCHEDULER] ✅ Started {len(self.threads)} background jobs")
    
    def stop(self):
        """Stop all background jobs."""
        if not self.running:
            logger.warning("[SCHEDULER] Not running")
            return
        
        logger.info("[SCHEDULER] Stopping background jobs...")
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=5.0)
        
        logger.info("[SCHEDULER] ✅ All jobs stopped")
    
    def _compaction_job(self):
        """
        Background job: Compact dynamic layer.
        
        Runs every COMPACTION_INTERVAL_SECONDS.
        Merges hot partition + delta segments into optimized index.
        """
        logger.info(f"[COMPACTION JOB] Started (interval: {COMPACTION_INTERVAL_SECONDS}s)")
        
        while self.running:
            try:
                time.sleep(COMPACTION_INTERVAL_SECONDS)
                
                if not self.running:
                    break
                
                logger.info("[COMPACTION JOB] Running compaction...")
                self.local_vdb.trigger_compaction()
                
            except Exception as e:
                logger.error(f"[COMPACTION JOB] Error: {e}")
        
        logger.info("[COMPACTION JOB] Stopped")
    
    def _anchor_decay_job(self):
        """
        Background job: Apply decay to anchors.
        
        Runs every ANCHOR_DECAY_CHECK_INTERVAL.
        Weakens unused anchors and evicts expired ones.
        """
        logger.info(f"[ANCHOR DECAY JOB] Started (interval: {ANCHOR_DECAY_CHECK_INTERVAL}s)")
        
        while self.running:
            try:
                time.sleep(ANCHOR_DECAY_CHECK_INTERVAL)
                
                if not self.running:
                    break
                
                logger.debug("[ANCHOR DECAY JOB] Running decay check...")
                self.anchor_system.decay_anchors()
                
            except Exception as e:
                logger.error(f"[ANCHOR DECAY JOB] Error: {e}")
        
        logger.info("[ANCHOR DECAY JOB] Stopped")
    
    def _cluster_eviction_job(self):
        """
        Background job: Evict stale semantic clusters.
        
        Runs every CLUSTER_TTL_SECONDS / 2.
        Removes clusters that haven't been accessed recently.
        """
        interval = CLUSTER_TTL_SECONDS // 2
        logger.info(f"[CLUSTER EVICTION JOB] Started (interval: {interval}s)")
        
        while self.running:
            try:
                time.sleep(interval)
                
                if not self.running:
                    break
                
                logger.debug("[CLUSTER EVICTION JOB] Running eviction check...")
                evicted = self.semantic_cache.evict_stale_clusters()
                
                if evicted:
                    logger.info(f"[CLUSTER EVICTION JOB] Evicted {len(evicted)} clusters")
                
            except Exception as e:
                logger.error(f"[CLUSTER EVICTION JOB] Error: {e}")
        
        logger.info("[CLUSTER EVICTION JOB] Stopped")
    
    def _prediction_expiry_job(self):
        """
        Background job: Expire old predictions.
        
        Runs every 60 seconds.
        Marks predictions older than 5 minutes as expired.
        """
        logger.info("[PREDICTION EXPIRY JOB] Started (interval: 60s)")
        
        while self.running:
            try:
                time.sleep(60)
                
                if not self.running:
                    break
                
                logger.debug("[PREDICTION EXPIRY JOB] Expiring old predictions...")
                self.anchor_system.expire_old_predictions(max_age_seconds=300)
                
            except Exception as e:
                logger.error(f"[PREDICTION EXPIRY JOB] Error: {e}")
        
        logger.info("[PREDICTION EXPIRY JOB] Stopped")