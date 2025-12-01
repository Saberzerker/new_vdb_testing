# 🚀 Hybrid VDB Benchmark

> A three-tier hybrid vector database system with momentum-based learning, semantic caching, and intelligent prefetching

## 📋 Overview

This project demonstrates an advanced hybrid vector database architecture designed for low-latency retrieval with privacy guarantees. It combines local storage (permanent + dynamic layers) with cloud-based canonical knowledge, implementing smart prefetching and momentum-based anchor trajectory prediction.

**Key Innovation:** Dynamic cache learns user query patterns through semantic clustering and predictive prefetching, achieving high cache hit rates without sacrificing accuracy.

## ✨ Features

- **Three-Tier Architecture**
  - 🏠 **TIER 1 (Permanent)**: Privacy-guaranteed, read-only baseline knowledge
  - 🎒 **TIER 2 (Dynamic)**: Learning layer with momentum-based prefetch cache
  - ☁️ **TIER 3 (Cloud)**: Canonical truth via Qdrant Cloud

- **Smart Retrieval**
  - Momentum-based anchor graphs for trajectory prediction
  - Semantic clustering with drift detection
  - Neighborhood-aware prefetch skipping (avoids redundant fetches)

- **Comprehensive Benchmarking**
  - Realistic query distributions (in-dataset, edge-case, out-of-distribution)
  - Performance telemetry: hit rates, latency percentiles (p50/p95/p99)
  - Learning curve visualization showing TIER 2 effectiveness over time

- **Production-Ready**
  - Thread-safe metrics tracking
  - Background prefetch operations (non-blocking)
  - HNSW indexing with optional INT8 quantization

## 📦 Requirements

- Python 3.8+
- Git
- Qdrant Cloud account (free tier available)
- ~2GB RAM for full benchmark

## 🛠️ Installation

1. **Clone the repository**
   ```
   git clone https://github.com/your-username/hybrid-vdb-benchmark.git
   cd hybrid-vdb-benchmark
   ```

2. **Create virtual environment**
   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```
   QDRANT_URL=https://your-cluster.gcp.cloud.qdrant.io:6333
   QDRANT_API_KEY=your-api-key-here
   QDRANT_COLLECTION=vdb_testing
   
   # Storage paths
   DATA_DIR=./data
   PERMANENT_DIR=./data/permanent
   DYNAMIC_DIR=./data/dynamic
   ```

   ⚠️ **Important:** Ensure no trailing commas in URLs!

## 🚀 Quick Start

### Run Quick Test (100 queries)
```
python benchmark/benchmark.py --mode quick
```

### Run Full Benchmark (1000 queries)
```
python benchmark/benchmark.py --mode full
```

### Run with Custom Queries
```
python benchmark/benchmark.py --mode quick --queries-file queries/custom_queries.txt
```

## 📊 What to Expect

When you run the benchmark, you'll see:

1. **Initialization Phase**
   - Loading embedding model (all-MiniLM-L6-v2, 384-dim)
   - Connecting to Qdrant Cloud
   - Loading PubMed QA dataset

2. **Benchmark Execution**
   - Real-time progress bar
   - Query distribution: 65% in-dataset, 25% edge-case, 10% OOD
   - Non-blocking background prefetch operations

3. **Performance Results**
   ```
   📊 OVERALL PERFORMANCE:
   Total queries: 1,000
   Avg latency: 45.2ms
   P50 latency: 12.3ms
   P95 latency: 156.7ms
   
   🎯 HIT RATES:
   TIER 1 (Permanent): 15.2%
   TIER 2 (Dynamic): 62.8%  ← Learning effectiveness!
   TIER 3 (Cloud): 22.0%
   Local Total (T1+T2): 78.0%
   
   🚀 SPEEDUP: 4.4× faster than cloud-only
   ```

4. **Generated Outputs**
   - `benchmark/results/<mode>/results.json` - Complete metrics
   - `benchmark/results/<mode>/results.csv` - Per-query data
   - `benchmark/results/<mode>/plots/` - Visualizations (learning curve, latency distribution)

## 📁 Project Structure

```
hybrid-vdb-benchmark/
├── src/                      # Core system modules
│   ├── hybrid_router.py      # Main routing logic
│   ├── local_vdb.py          # Three-tier local VDB interface
│   ├── cloud_client.py       # Qdrant Cloud integration
│   ├── anchor_system.py      # Momentum-based trajectory prediction
│   ├── semantic_cache.py     # Clustering with drift detection
│   ├── storage_engine.py     # HNSW + quantization storage
│   ├── metrics.py            # Thread-safe telemetry
│   └── config.py             # Central configuration
│
├── benchmark/                # Benchmarking framework
│   ├── benchmark.py          # Main benchmark script
│   ├── benchmark_config.py   # Benchmark-specific config
│   ├── data_loader.py        # PubMed QA dataset loader
│   ├── query_generator.py    # Realistic query distribution
│   └── visualizer.py         # Results visualization
│
├── data/                     # Vector storage (auto-created)
│   ├── permanent/            # TIER 1 storage
│   └── dynamic/              # TIER 2 storage
│
├── .env                      # Environment variables (YOU create this)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🔧 Configuration

Key parameters in `src/config.py`:

```
# Storage capacity
PERMANENT_LAYER_CAPACITY = 300      # TIER 1 size
DYNAMIC_LAYER_CAPACITY = 700        # TIER 2 size

# Search thresholds
LOCAL_CONFIDENCE_THRESHOLD = 0.75   # Min confidence for local results
PREDICTION_SIMILARITY_THRESHOLD = 0.85  # Prefetch prediction matching

# Anchor system (star ratings)
ANCHOR_STRONG_THRESHOLD = 60.0      # 5-star anchor threshold
ANCHOR_PERMANENT_THRESHOLD = 90.0   # 10-star permanent anchor
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` in virtual env |
| Qdrant connection error | Check `.env` credentials, ensure no trailing commas |
| `logger not defined` | Verify `import logging` at top of cloud_client.py |
| Low TIER 2 hit rate (<30%) | Normal for first 20 queries (cold start phase) |

## 📈 Understanding Results

- **TIER 1 hits**: Queries answered from baseline privacy layer
- **TIER 2 hits**: Queries predicted and prefetched (learning effectiveness!)
- **TIER 3 hits**: Queries requiring cloud fallback
- **Learning curve**: Shows TIER 2 hit rate improving over time (cold → warmup → steady)

Expected progression:
- Queries 1-3: ~10% TIER 2 rate (cold start)
- Queries 4-20: ~35% TIER 2 rate (warmup)
- Queries 20+: ~60-70% TIER 2 rate (steady state)

## 🤝 Contributing

This is an educational project. For improvements:
1. Fork the repo
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

## 📄 License

Educational/Research use. See source files for specific licensing.

## 👤 Author

**Saberzerker**
- Original implementation: 2025-11-17
- Benchmark framework: 2025-11-30

## 🙏 Acknowledgments

- PubMed QA dataset for medical question answering
- Qdrant for vector database infrastructure
- sentence-transformers for embedding models

---

**⭐ If this helps your research, consider starring the repo!**
```
