# benchmark/benchmark_config.py

"""
Benchmark-specific configuration.

Separate from src/config.py to keep middleware clean.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load credentials
load_dotenv()


class BenchmarkConfig:
    """Configuration for benchmark runs."""
    
    # ═══════════════════════════════════════════════════════════
    # PATHS
    # ═══════════════════════════════════════════════════════════
    
    # Project root
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # Benchmark directories
    BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
    QUERIES_DIR = BENCHMARK_DIR / "queries"
    RESULTS_DIR = BENCHMARK_DIR / "results"
    
    # Data directories
    DATA_DIR = PROJECT_ROOT / "data"
    PERMANENT_DIR = DATA_DIR / "permanent"
    DYNAMIC_DIR = DATA_DIR / "dynamic"
    
    # Create directories
    for directory in [QUERIES_DIR, RESULTS_DIR, DATA_DIR, PERMANENT_DIR, DYNAMIC_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # ═══════════════════════════════════════════════════════════
    # CLOUD CREDENTIALS
    # ═══════════════════════════════════════════════════════════
    
    # Qdrant Cloud
    QDRANT_URL = os.getenv("QDRANT_URL", "https://your-cluster.gcp.cloud.qdrant.io:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "your-api-key-here")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "pubmed_qa_full")
    
    # ═══════════════════════════════════════════════════════════
    # DATASET
    # ═══════════════════════════════════════════════════════════
    
    DATASET_NAME = "pubmed_qa"
    DATASET_SUBSET = "pqa_labeled"
    
    # ═══════════════════════════════════════════════════════════
    # STORAGE CONFIGURATION
    # ═══════════════════════════════════════════════════════════
    
    # TIER 1: Permanent (read-only baseline)
    TIER1_SIZE = 10_000                    # 10k vectors (~32 MB)
    
    # TIER 2: Dynamic (learning space)
    TIER2_CAPACITY = 50_000                # 50k capacity (~188 MB with quantization)
    
    # TIER 3: Cloud (full dataset)
    # ~211,000 vectors (~850 MB on Qdrant)
    
    # ═══════════════════════════════════════════════════════════
    # BENCHMARK SETTINGS
    # ═══════════════════════════════════════════════════════════
    
    # Test modes
    QUICK_TEST_SIZE = 100                  # Fast validation
    FULL_TEST_SIZE = 1000                  # Realistic workload
    
    # Query distribution (for full test)
    # Simulates medical chatbot usage pattern
    IN_DATASET_RATIO = 0.65               # 65% - Normal questions
    EDGE_CASE_RATIO = 0.25                # 25% - Paraphrased/similar
    OOD_RATIO = 0.10                      # 10% - Novel questions
    
    # Timing
    QUERY_INTERVAL_MS = 500               # 500ms = 2 queries/sec (realistic)
    
    # ═══════════════════════════════════════════════════════════
    # MODEL
    # ═══════════════════════════════════════════════════════════
    
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    MODEL_CACHE_DIR = PROJECT_ROOT / "models"  # Cache downloaded models
    
    # ═══════════════════════════════════════════════════════════
    # VDB OPTIMIZATIONS
    # ═══════════════════════════════════════════════════════════
    
    USE_HNSW = True                       # Fast approximate search
    USE_QUANTIZATION = True               # INT8 compression (4× memory savings)
    PREFETCH_ENABLED = True               # Smart prefetching
    
    # HNSW parameters
    HNSW_M = 16                           # Connections per layer
    HNSW_EF_CONSTRUCTION = 200            # Build quality
    HNSW_EF_SEARCH = 50                   # Search quality
    
    # ═══════════════════════════════════════════════════════════
    # OUTPUT
    # ═══════════════════════════════════════════════════════════
    
    SAVE_PLOTS = True
    SAVE_JSON = True
    SAVE_CSV = True
    
    PLOT_DPI = 150                        # Plot resolution
    
    # ═══════════════════════════════════════════════════════════
    # DISPLAY
    # ═══════════════════════════════════════════════════════════
    
    VERBOSE = True                        # Print detailed progress
    SHOW_WARNINGS = True


# Create singleton instance
config = BenchmarkConfig()


# ═══════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════

def validate_config():
    """Validate configuration before running."""
    
    errors = []
    
    # Check credentials
    if "your-cluster" in config.QDRANT_URL:
        errors.append("❌ QDRANT_URL not set in .env")
    
    if "your-api-key" in config.QDRANT_API_KEY:
        errors.append("❌ QDRANT_API_KEY not set in .env")
    
    # Check storage
    total_capacity = config.TIER1_SIZE + config.TIER2_CAPACITY
    estimated_mb = total_capacity * 384 * 4 / (1024**2)
    
    print("="*70)
    print("STORAGE CONFIGURATION")
    print("="*70)
    print(f"TIER 1 (Permanent): {config.TIER1_SIZE:,} vectors (~{config.TIER1_SIZE * 1.5 / 1000:.0f} MB)")
    print(f"TIER 2 (Dynamic):   {config.TIER2_CAPACITY:,} capacity (~{config.TIER2_CAPACITY * 1.5 / 1000:.0f} MB)")
    print(f"Total Local:        ~{estimated_mb:.0f} MB")
    print(f"With Quantization:  ~{estimated_mb * 0.6:.0f} MB (40% savings)")
    print("="*70)
    
    if errors:
        print("\n⚠️  CONFIGURATION ERRORS:")
        for error in errors:
            print(f"  {error}")
        print("\n💡 Create a .env file with:")
        print("   QDRANT_URL=https://your-cluster.gcp.cloud.qdrant.io:6333")
        print("   QDRANT_API_KEY=your-api-key-here")
        print("   QDRANT_COLLECTION=pubmed_qa_full")
        return False
    
    print("✅ Configuration valid")
    return True


if __name__ == "__main__":
    validate_config()
