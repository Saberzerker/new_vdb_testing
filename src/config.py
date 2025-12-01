# src/config.py
"""
Configuration for Hybrid VDB System

All tunable parameters in one place.
Defines the three-tier architecture parameters and learning thresholds.

Author: Saberzerker
Date: 2025-11-17
"""

import os
from pathlib import Path


class Config:
    """Central configuration for hybrid VDB system."""
    
    # ═══════════════════════════════════════════════════════════
    # PATHS
    # ═══════════════════════════════════════════════════════════
    
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    
    BASE_LAYER_PATH = DATA_DIR / "permanent"
    DYNAMIC_LAYER_PATH = DATA_DIR / "dynamic"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Create directories if they don't exist
    BASE_LAYER_PATH.mkdir(parents=True, exist_ok=True)
    DYNAMIC_LAYER_PATH.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # ═══════════════════════════════════════════════════════════
    # VECTOR DIMENSIONS
    # ═══════════════════════════════════════════════════════════
    
    VECTOR_DIMENSION = 384  # all-MiniLM-L6-v2 dimension
    
    # ═══════════════════════════════════════════════════════════
    # STORAGE CAPACITY (THREE-TIER ARCHITECTURE)
    # ═══════════════════════════════════════════════════════════
    
    # TIER 1: Permanent Layer (Kitchen - Privacy Layer)
    PERMANENT_LAYER_CAPACITY = 300  # Read-only, pre-seeded, never evicted
    
    # TIER 2: Dynamic Layer (Backpack - Learning Engine, FIXED SIZE!)
    DYNAMIC_LAYER_CAPACITY = 700    # Learning space, NEVER grows beyond this!
    
    # Hot partition (in-memory portion of dynamic)
    HOT_PARTITION_RAM_LIMIT = 350   # Half of dynamic in RAM for speed
    
    # TIER 3: Cloud VDB (Bakery - Canonical Truth)
    # No capacity limit (cloud has unlimited storage)
    
    # ═══════════════════════════════════════════════════════════
    # CLOUD VDB CONFIGURATION (TIER 3)
    # ═══════════════════════════════════════════════════════════
    
    CLOUD_PROVIDER = os.getenv("CLOUD_PROVIDER", "qdrant")
    CLOUD_URL = os.getenv("QDRANT_URL", "https://6e6e7451-fa6d-4dcb-b987-49dba2bb7373.europe-west3-0.gcp.cloud.qdrant.io:6333")
    CLOUD_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.duR7ZSdOyXclYfAm6AR8-ttSg1sAdBYmF4Vw4ykshaE")
    CLOUD_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "hybrid_vdb_test")
    
    # Cloud artificial latency for testing (when using stub)
    CLOUD_ARTIFICIAL_LATENCY_MS = 200
    CLOUD_TIMEOUT_SECONDS = 10.0
    
    # ═══════════════════════════════════════════════════════════
    # SEARCH PARAMETERS
    # ═══════════════════════════════════════════════════════════
    
    DEFAULT_SEARCH_K = 5
    LOCAL_CONFIDENCE_THRESHOLD = 0.75  # Min score to trust local result (75%)
    
    # Similarity thresholds
    MIN_SIMILARITY_THRESHOLD = 0.50  # Below this = not relevant
    HIGH_SIMILARITY_THRESHOLD = 0.90  # Above this = very confident
    
    # ═══════════════════════════════════════════════════════════
    # PREFETCHING PARAMETERS (THE SMART LEARNING!)
    # ═══════════════════════════════════════════════════════════
    
    PREFETCH_ENABLED = True
    PREFETCH_K = 5  # Number of predictions to generate per query
    
    # Phase thresholds (determines prefetch strategy)
    COLD_START_QUERIES = 3   # Queries 1-3: Fill aggressively
    WARMUP_QUERIES = 20      # Queries 4-20: Refine accuracy
    # After query 20: Steady state (high accuracy maintenance)
    
    # Prediction matching threshold
    PREDICTION_SIMILARITY_THRESHOLD = 0.85  # 85% similar = prediction match
    
    # Neighborhood checking thresholds (SMART CHECK!)
    # These determine when to SKIP fetching (because we already have it)
    NEIGHBORHOOD_THRESHOLD_COLD = 0.85       # Cold start (less strict)
    NEIGHBORHOOD_THRESHOLD_WARMUP = 0.90     # Warmup (moderate)
    NEIGHBORHOOD_THRESHOLD_STEADY = 0.92     # Steady state (very strict)
    
    # Prediction generation noise
    NOISE_SCALE_COLD = 0.30     # High noise for exploration
    NOISE_SCALE_WARMUP = 0.15   # Medium noise
    NOISE_SCALE_STEADY = 0.05   # Low noise, exploit known paths
    
    # ═══════════════════════════════════════════════════════════
    # ANCHOR SYSTEM (STAR RATING SYSTEM!)
    # ═══════════════════════════════════════════════════════════
    
    # Anchor type thresholds (strength levels)
    ANCHOR_WEAK_THRESHOLD = 0.0        # ⭐ (0-25 strength)
    ANCHOR_MEDIUM_THRESHOLD = 25.0     # ⭐⭐⭐ (25-60 strength)
    ANCHOR_STRONG_THRESHOLD = 60.0     # ⭐⭐⭐⭐⭐ (60-90 strength)
    ANCHOR_PERMANENT_THRESHOLD = 90.0  # ⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐ (90+ strength)
    
    # Reinforcement parameters
    ANCHOR_HIT_REWARD = 1.0          # Strength increase on correct prediction
    ANCHOR_MISS_PENALTY = 0.5        # Strength decrease on wrong prediction
    ANCHOR_PARENT_CREDIT = 0.5       # Credit to parent anchor (50%)
    ANCHOR_CHILD_BOOST = 0.3         # Priority boost to children
    
    # Decay rates (per hour of idle time)
    ANCHOR_DECAY_WEAK = 0.5      # Weak anchors decay 50%/hour
    ANCHOR_DECAY_MEDIUM = 0.2    # Medium anchors decay 20%/hour
    ANCHOR_DECAY_STRONG = 0.1    # Strong anchors decay 10%/hour
    ANCHOR_DECAY_PERMANENT = 0.0 # Permanent anchors NEVER decay
    
    # Eviction timeouts (idle time before removal)
    ANCHOR_EVICT_WEAK_SECONDS = 3600      # 1 hour idle
    ANCHOR_EVICT_MEDIUM_SECONDS = 21600   # 6 hours idle
    ANCHOR_EVICT_STRONG_SECONDS = 86400   # 24 hours idle
    # Permanent anchors are NEVER evicted
    
    # Minimum strength before eviction
    ANCHOR_MIN_STRENGTH = 0.1
    
    # ═══════════════════════════════════════════════════════════
    # SEMANTIC CACHE (MOMENTUM-BASED CLUSTERING)
    # ═══════════════════════════════════════════════════════════
    
    SEMANTIC_DRIFT_THRESHOLD = 0.35  # Distance > 0.35 = new cluster created
    MOMENTUM_ALPHA = 0.9  # Centroid momentum (0.9 = 90% old, 10% new)
    MIN_CLUSTER_REINFORCEMENT_SCORE = 0.70  # Min score to reinforce cluster
    
    # Cluster lifecycle
    CLUSTER_TTL_SECONDS = 86400  # 24 hours before cluster can be evicted
    CLUSTER_MIN_ACCESS_COUNT = 3  # Min accesses to avoid eviction
    
    # ═══════════════════════════════════════════════════════════
    # BACKGROUND JOBS (MAINTENANCE)
    # ═══════════════════════════════════════════════════════════
    
    COMPACTION_INTERVAL_SECONDS = 3600  # Compact dynamic layer every 1 hour
    DECAY_INTERVAL_SECONDS = 1800       # Check for decay every 30 minutes
    EVICTION_INTERVAL_SECONDS = 3600    # Evict weak anchors every 1 hour
    
    # Auto-save interval
    AUTOSAVE_INTERVAL_SECONDS = 600  # Save state every 10 minutes
    
    # ═══════════════════════════════════════════════════════════
    # LOGGING & DEBUGGING
    # ═══════════════════════════════════════════════════════════
    
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"
    
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = LOGS_DIR / "hybrid_vdb.log"
    
    # ═══════════════════════════════════════════════════════════
    # DEMO APP SETTINGS
    # ═══════════════════════════════════════════════════════════
    
    DEMO_HOST = "0.0.0.0"
    DEMO_PORT = 5000
    DEMO_DEBUG = False
    
    # LLM settings (for RAG)
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
    OLLAMA_TIMEOUT = 45 
    
    # RAG settings
    RAG_CONTEXT_SIZE = 3  # Top-3 documents for context
    RAG_MAX_CONTEXT_LENGTH = 2000  # Max characters per document
    
    # ═══════════════════════════════════════════════════════════
    # METRICS & TELEMETRY
    # ═══════════════════════════════════════════════════════════
    
    METRICS_ENABLED = True
    METRICS_EXPORT_INTERVAL = 60  # Export metrics every minute
    METRICS_HISTORY_LIMIT = 1000  # Keep last 1000 events
    
    # ═══════════════════════════════════════════════════════════
    # DEVELOPMENT & TESTING
    # ═══════════════════════════════════════════════════════════
    
    USE_REAL_SYSTEM = os.getenv("USE_REAL_SYSTEM", "true").lower() == "true"
    ENABLE_PROFILING = False
    
    # Test mode settings
    TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"
    TEST_QUERY_LIMIT = 100  # Max queries in test mode



# ═══════════════════════════════════════════════════════════
# CREATE SINGLETON & EXPORT ALL CONSTANTS
# ═══════════════════════════════════════════════════════════

# Create singleton instance
config = Config()

# Vector dimensions
VECTOR_DIMENSION = config.VECTOR_DIMENSION

# Paths
BASE_LAYER_PATH = config.BASE_LAYER_PATH
DYNAMIC_LAYER_PATH = config.DYNAMIC_LAYER_PATH
LOGS_DIR = config.LOGS_DIR
DATA_DIR = config.DATA_DIR

# Storage capacity
PERMANENT_LAYER_CAPACITY = config.PERMANENT_LAYER_CAPACITY
DYNAMIC_LAYER_CAPACITY = config.DYNAMIC_LAYER_CAPACITY
HOT_PARTITION_RAM_LIMIT = config.HOT_PARTITION_RAM_LIMIT

# Search parameters
DEFAULT_SEARCH_K = config.DEFAULT_SEARCH_K
LOCAL_CONFIDENCE_THRESHOLD = config.LOCAL_CONFIDENCE_THRESHOLD
MIN_SIMILARITY_THRESHOLD = config.MIN_SIMILARITY_THRESHOLD
HIGH_SIMILARITY_THRESHOLD = config.HIGH_SIMILARITY_THRESHOLD

# Prefetch parameters
PREFETCH_ENABLED = config.PREFETCH_ENABLED
PREFETCH_K = config.PREFETCH_K
COLD_START_QUERIES = config.COLD_START_QUERIES
WARMUP_QUERIES = config.WARMUP_QUERIES
PREDICTION_SIMILARITY_THRESHOLD = config.PREDICTION_SIMILARITY_THRESHOLD
NEIGHBORHOOD_THRESHOLD_COLD = config.NEIGHBORHOOD_THRESHOLD_COLD
NEIGHBORHOOD_THRESHOLD_WARMUP = config.NEIGHBORHOOD_THRESHOLD_WARMUP
NEIGHBORHOOD_THRESHOLD_STEADY = config.NEIGHBORHOOD_THRESHOLD_STEADY
NOISE_SCALE_COLD = config.NOISE_SCALE_COLD
NOISE_SCALE_WARMUP = config.NOISE_SCALE_WARMUP
NOISE_SCALE_STEADY = config.NOISE_SCALE_STEADY

# Anchor system
ANCHOR_WEAK_THRESHOLD = config.ANCHOR_WEAK_THRESHOLD
ANCHOR_MEDIUM_THRESHOLD = config.ANCHOR_MEDIUM_THRESHOLD
ANCHOR_STRONG_THRESHOLD = config.ANCHOR_STRONG_THRESHOLD
ANCHOR_PERMANENT_THRESHOLD = config.ANCHOR_PERMANENT_THRESHOLD
ANCHOR_HIT_REWARD = config.ANCHOR_HIT_REWARD
ANCHOR_MISS_PENALTY = config.ANCHOR_MISS_PENALTY
ANCHOR_PARENT_CREDIT = config.ANCHOR_PARENT_CREDIT
ANCHOR_CHILD_BOOST = config.ANCHOR_CHILD_BOOST
ANCHOR_DECAY_WEAK = config.ANCHOR_DECAY_WEAK
ANCHOR_DECAY_MEDIUM = config.ANCHOR_DECAY_MEDIUM
ANCHOR_DECAY_STRONG = config.ANCHOR_DECAY_STRONG
ANCHOR_DECAY_PERMANENT = config.ANCHOR_DECAY_PERMANENT
ANCHOR_EVICT_WEAK_SECONDS = config.ANCHOR_EVICT_WEAK_SECONDS
ANCHOR_EVICT_MEDIUM_SECONDS = config.ANCHOR_EVICT_MEDIUM_SECONDS
ANCHOR_EVICT_STRONG_SECONDS = config.ANCHOR_EVICT_STRONG_SECONDS
ANCHOR_MIN_STRENGTH = config.ANCHOR_MIN_STRENGTH

# Semantic cache
SEMANTIC_DRIFT_THRESHOLD = config.SEMANTIC_DRIFT_THRESHOLD
MOMENTUM_ALPHA = config.MOMENTUM_ALPHA
MIN_CLUSTER_REINFORCEMENT_SCORE = config.MIN_CLUSTER_REINFORCEMENT_SCORE
CLUSTER_TTL_SECONDS = config.CLUSTER_TTL_SECONDS
CLUSTER_MIN_ACCESS_COUNT = config.CLUSTER_MIN_ACCESS_COUNT

# Background jobs
COMPACTION_INTERVAL_SECONDS = config.COMPACTION_INTERVAL_SECONDS
DECAY_INTERVAL_SECONDS = config.DECAY_INTERVAL_SECONDS
EVICTION_INTERVAL_SECONDS = config.EVICTION_INTERVAL_SECONDS
AUTOSAVE_INTERVAL_SECONDS = config.AUTOSAVE_INTERVAL_SECONDS

# Logging
LOG_LEVEL = config.LOG_LEVEL
VERBOSE = config.VERBOSE
LOG_FORMAT = config.LOG_FORMAT
LOG_FILE = config.LOG_FILE

# Demo app
DEMO_HOST = config.DEMO_HOST
DEMO_PORT = config.DEMO_PORT
DEMO_DEBUG = config.DEMO_DEBUG
OLLAMA_HOST = config.OLLAMA_HOST
OLLAMA_MODEL = config.OLLAMA_MODEL
OLLAMA_TIMEOUT = config.OLLAMA_TIMEOUT
RAG_CONTEXT_SIZE = config.RAG_CONTEXT_SIZE
RAG_MAX_CONTEXT_LENGTH = config.RAG_MAX_CONTEXT_LENGTH

# Metrics
METRICS_ENABLED = config.METRICS_ENABLED
METRICS_EXPORT_INTERVAL = config.METRICS_EXPORT_INTERVAL
METRICS_HISTORY_LIMIT = config.METRICS_HISTORY_LIMIT

# Development
USE_REAL_SYSTEM = config.USE_REAL_SYSTEM
ENABLE_PROFILING = config.ENABLE_PROFILING
TEST_MODE = config.TEST_MODE
TEST_QUERY_LIMIT = config.TEST_QUERY_LIMIT

# Cloud (add this if missing from Config class)
CLOUD_TIMEOUT_SECONDS = 30
CLOUD_PROVIDER = config.CLOUD_PROVIDER
CLOUD_URL = config.CLOUD_URL
CLOUD_API_KEY = config.CLOUD_API_KEY
CLOUD_COLLECTION_NAME = config.CLOUD_COLLECTION_NAME
CLOUD_ARTIFICIAL_LATENCY_MS = config.CLOUD_ARTIFICIAL_LATENCY_MS
ANCHOR_DECAY_CHECK_INTERVAL = config.DECAY_INTERVAL_SECONDS

# Convenience function
def get_config():
    """Get configuration singleton."""
    return config



USE_HNSW = True  # Enable HNSW for fast approximate search

# HNSW parameters
HNSW_M = 16  # Connections per layer (higher = more accurate, slower build)
             # 16 is optimal for most cases
             
HNSW_EF_CONSTRUCTION = 200  # Build quality (higher = better index, slower build)
                            # 200 is good default
                            
HNSW_EF_SEARCH = 50  # Search quality (higher = more accurate, slower search)
                     # 50 gives 99%+ recall with good speed
                     # Increase to 100 for 99.9%+ recall
                     
# ═══════════════════════════════════════════════════════════
# QUANTIZATION CONFIGURATION (OPTIMIZATION)
# ═══════════════════════════════════════════════════════════

USE_QUANTIZATION = True  # Enable INT8 quantization for 4× memory reduction

# Quantization trades memory for slight accuracy loss:
# - Memory: 1.5MB → 384KB (4× reduction)
# - Accuracy: >98% retained (cosine similarity)
# - Speed: Slightly slower (quantize/dequantize overhead ~5-10%)
#
# Set to False if you need 100% accuracy or have abundant memory
