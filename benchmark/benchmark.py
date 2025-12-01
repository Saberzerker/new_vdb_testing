# benchmark/benchmark.py

"""
Main Benchmark Script for Hybrid VDB System

Two modes:
- Quick: 100 queries for fast validation
- Full: 1000 queries with realistic distribution (65/25/10)

Usage:
    python benchmark/benchmark.py --mode quick
    python benchmark/benchmark.py --mode full
    python benchmark/benchmark.py --mode quick --queries-file queries/custom_queries.txt

Author: Saberzerker
Date: 2025-11-30
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import middleware
from src.hybrid_router import HybridRouter
from src.local_vdb import LocalVDB
from src.cloud_client import QdrantCloudClient
from src.anchor_system import AnchorSystem
from src.semantic_cache import SemanticClusterCache
from src.metrics import MetricsTracker
from src.config import Config

# Import benchmark modules
from benchmark.benchmark_config import config as benchmark_config, validate_config
from benchmark.data_loader import DataLoader
from benchmark.query_generator import QueryGenerator
from benchmark.visualizer import BenchmarkVisualizer


class HybridVDBBenchmark:
    """
    Main benchmark orchestrator.
    """
    
    def __init__(
        self,
        test_mode: str = "full",
        custom_queries_file: Optional[str] = None
    ):
        """
        Initialize benchmark.
        
        Args:
            test_mode: "quick" (100 queries) or "full" (1000 queries)
            custom_queries_file: Optional path to custom queries
        """
        self.test_mode = test_mode
        self.custom_queries_file = custom_queries_file
        
        # Validate configuration
        if not validate_config():
            raise ValueError("Invalid configuration. Check your .env file.")
        
        # Determine query count
        if test_mode == "quick":
            self.num_queries = benchmark_config.QUICK_TEST_SIZE
        else:
            self.num_queries = benchmark_config.FULL_TEST_SIZE
        
        # Print configuration
        self._print_header()
        
        # Initialize components
        self.data_loader = DataLoader()
        self.query_generator = QueryGenerator(self.data_loader)
        self.visualizer = BenchmarkVisualizer(test_mode)
        
        # VDB system (initialized later)
        self.router = None
        self.metrics = None
        self.anchor_system = None
        self.local_vdb = None
        
        # Results storage
        self.results = {
            'config': {
                'test_mode': test_mode,
                'num_queries': self.num_queries,
                'tier1_size': benchmark_config.TIER1_SIZE,
                'tier2_capacity': benchmark_config.TIER2_CAPACITY,
                'timestamp': time.time(),
            },
            'per_query': [],
            'anchor_snapshots': [],
            'cache_snapshots': [],
        }
    
    def _print_header(self):
        """Print benchmark header."""
        print("="*70)
        print("HYBRID VDB BENCHMARK")
        print("="*70)
        print(f"Test Mode:        {self.test_mode.upper()}")
        print(f"Total Queries:    {self.num_queries:,}")
        print(f"\nStorage Configuration:")
        print(f"  TIER 1 (Perm):  {benchmark_config.TIER1_SIZE:,} docs")
        print(f"  TIER 2 (Dyn):   {benchmark_config.TIER2_CAPACITY:,} capacity")
        print(f"  TIER 3 (Cloud): ~211,000 docs")
        
        if self.test_mode == "full":
            in_count = int(self.num_queries * benchmark_config.IN_DATASET_RATIO)
            edge_count = int(self.num_queries * benchmark_config.EDGE_CASE_RATIO)
            ood_count = int(self.num_queries * benchmark_config.OOD_RATIO)
            
            print(f"\nQuery Distribution:")
            print(f"  In-dataset:     {in_count} ({benchmark_config.IN_DATASET_RATIO*100:.0f}%)")
            print(f"  Edge cases:     {edge_count} ({benchmark_config.EDGE_CASE_RATIO*100:.0f}%)")
            print(f"  Out-of-dist:    {ood_count} ({benchmark_config.OOD_RATIO*100:.0f}%)")
        
        est_time = self.num_queries * benchmark_config.QUERY_INTERVAL_MS / 1000 / 60
        print(f"\nEstimated Time:   ~{est_time:.1f} minutes")
        print("="*70)
    
    def initialize_vdb_system(self):
        """Initialize VDB components."""
        print("\n[VDB] Initializing Hybrid VDB system...")
        
        # Create middleware config
        config = Config()
        config.DYNAMIC_LAYER_CAPACITY = benchmark_config.TIER2_CAPACITY
        config.BASE_LAYER_PATH = str(benchmark_config.PERMANENT_DIR)
        config.DYNAMIC_LAYER_PATH = str(benchmark_config.DYNAMIC_DIR)
        config.PREFETCH_ENABLED = benchmark_config.PREFETCH_ENABLED
        config.USE_HNSW = benchmark_config.USE_HNSW
        config.USE_QUANTIZATION = benchmark_config.USE_QUANTIZATION
        
        # Initialize components
        self.local_vdb = LocalVDB(config)
        self.cloud_vdb = QdrantCloudClient(config)
        self.semantic_cache = SemanticClusterCache(config)
        self.anchor_system = AnchorSystem(config)
        self.metrics = MetricsTracker()
        
        # Create router
        self.router = HybridRouter(
            local_vdb=self.local_vdb,
            cloud_vdb=self.cloud_vdb,
            semantic_cache=self.semantic_cache,
            anchor_system=self.anchor_system,
            metrics=self.metrics
        )
        
        print("[VDB] ✅ System initialized")
    
    def get_queries(self) -> Tuple[List[str], List[str], np.ndarray, Optional[List[str]]]:
        """
        Get queries based on test mode.
        
        Returns:
            (query_ids, query_texts, query_embeddings, query_types)
        """
        if self.test_mode == "quick" and self.custom_queries_file:
            # Load from file
            print(f"\n[QUERIES] Loading from {self.custom_queries_file}...")
            query_ids, query_texts, query_embeddings = \
                self.query_generator.load_custom_queries(
                    self.custom_queries_file,
                    limit=self.num_queries
                )
            query_types = None
            
        elif self.test_mode == "quick":
            # Sample from dataset
            print(f"\n[QUERIES] Sampling {self.num_queries} quick test queries...")
            query_ids, query_texts, query_embeddings = \
                self.query_generator.generate_quick_queries(self.num_queries)
            query_types = None
            
        else:
            # Full test with realistic distribution
            print(f"\n[QUERIES] Generating {self.num_queries} queries with realistic mix...")
            query_ids, query_texts, query_embeddings, query_types = \
                self.query_generator.generate_full_test_queries(
                    total=self.num_queries,
                    in_dataset_ratio=benchmark_config.IN_DATASET_RATIO,
                    edge_case_ratio=benchmark_config.EDGE_CASE_RATIO,
                    ood_ratio=benchmark_config.OOD_RATIO
                )
        
        return query_ids, query_texts, query_embeddings, query_types
    
    def run_benchmark(
        self,
        query_ids: List[str],
        query_texts: List[str],
        query_embeddings: np.ndarray,
        query_types: Optional[List[str]] = None
    ):
        """Execute benchmark queries."""
        print("\n" + "="*70)
        print("STARTING BENCHMARK")
        print("="*70)
        
        start_time = time.time()
        snapshot_interval = 10 if self.test_mode == "quick" else 50
        
        for i in tqdm(range(len(query_ids)), desc="Running queries"):
            qid = query_ids[i]
            qtext = query_texts[i]
            qvec = query_embeddings[i]
            qtype = query_types[i] if query_types else "unknown"
            
            # Execute query
            result = self.router.search(
                query_vector=qvec,
                query_id=qid,
                query_text=qtext,
                k=5
            )
            
            # Record result
            self.results['per_query'].append({
                'query_num': i + 1,
                'query_id': qid,
                'query_type': qtype,
                'query_text': qtext[:80],
                'latency_ms': result['latency_ms'],
                'source': result['source'],
                'confidence': result.get('confidence', 0),
                'timestamp': time.time() - start_time,
            })
            
            # Take snapshots
            if (i + 1) % snapshot_interval == 0:
                # Anchor snapshot
                anchor_stats = self.anchor_system.get_anchor_stats()
                self.results['anchor_snapshots'].append({
                    'query_num': i + 1,
                    **anchor_stats
                })
                
                # Cache snapshot
                cache_stats = self.local_vdb.get_dynamic_stats()
                self.results['cache_snapshots'].append({
                    'query_num': i + 1,
                    **cache_stats
                })
            
            # Realistic delay
            time.sleep(benchmark_config.QUERY_INTERVAL_MS / 1000.0)
        
        total_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("BENCHMARK COMPLETE")
        print("="*70)
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print("="*70)
    
    def analyze_results(self):
        """Analyze and display results."""
        print("\n" + "="*70)
        print("RESULTS ANALYSIS")
        print("="*70)
        
        metrics = self.metrics.get_summary()
        
        # Overall performance
        print("\n📊 OVERALL PERFORMANCE:")
        print(f"Total queries:        {metrics['total_queries']:,}")
        print(f"Avg latency:          {metrics['avg_latency']:.1f}ms")
        print(f"P50 latency:          {metrics['p50_latency']:.1f}ms")
        print(f"P95 latency:          {metrics['p95_latency']:.1f}ms")
        print(f"Min latency:          {metrics['min_latency']:.1f}ms")
        print(f"Max latency:          {metrics['max_latency']:.1f}ms")
        
        # Hit rates
        print("\n🎯 HIT RATES:")
        print(f"TIER 1 (Permanent):   {metrics['tier1_hit_rate']:.1f}%")
        print(f"TIER 2 (Dynamic):     {metrics['tier2_hit_rate']:.1f}%")
        print(f"TIER 3 (Cloud):       {metrics['tier3_hit_rate']:.1f}%")
        print(f"Local Total (T1+T2):  {metrics['local_hit_rate']:.1f}%")
        
        # Learning progression
        if len(self.results['anchor_snapshots']) > 0:
            final_anchors = self.results['anchor_snapshots'][-1]
            
            print("\n⚓ ANCHOR SYSTEM:")
            print(f"Total anchors:        {final_anchors.get('total_anchors', 0)}")
            print(f"  Weak:               {final_anchors.get('weak_anchors', 0)}")
            print(f"  Medium:             {final_anchors.get('medium_anchors', 0)}")
            print(f"  Strong:             {final_anchors.get('strong_anchors', 0)}")
            print(f"  Permanent:          {final_anchors.get('permanent_anchors', 0)}")
        
        if len(self.results['cache_snapshots']) > 0:
            final_cache = self.results['cache_snapshots'][-1]
            
            print("\n💾 DYNAMIC CACHE:")
            print(f"Size:                 {final_cache.get('current_size', 0):,} / "
                  f"{final_cache.get('capacity', 0):,}")
            print(f"Fill rate:            {final_cache.get('fill_rate', 0):.1f}%")
            print(f"Avg weight:           {final_cache.get('avg_weight', 0):.2f}")
        
        # Speedup
        cloud_latency = 200
        speedup = cloud_latency / metrics['avg_latency']
        
        print("\n🚀 SPEEDUP:")
        print(f"Cloud-only baseline:  {cloud_latency}ms")
        print(f"Hybrid system:        {metrics['avg_latency']:.1f}ms")
        print(f"Speedup:              {speedup:.1f}× faster")
        
        print("\n" + "="*70)
    
    def save_results(self):
        """Save results to files."""
        output_dir = benchmark_config.RESULTS_DIR / self.test_mode
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[SAVE] Saving results to {output_dir}...")
        
        # Save JSON
        import json
        json_file = output_dir / "results.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"[SAVE] ✅ JSON: {json_file.name}")
        
        # Save CSV
        import pandas as pd
        df = pd.DataFrame(self.results['per_query'])
        csv_file = output_dir / "results.csv"
        df.to_csv(csv_file, index=False)
        print(f"[SAVE] ✅ CSV: {csv_file.name}")
    
    def run(self):
        """Run complete benchmark pipeline."""
        try:
            # Initialize VDB
            self.initialize_vdb_system()
            
            # Get queries
            query_ids, query_texts, query_embeddings, query_types = self.get_queries()
            
            # Run benchmark
            self.run_benchmark(query_ids, query_texts, query_embeddings, query_types)
            
            # Analyze
            self.analyze_results()
            
            # Visualize
            self.visualizer.generate_plots(self.results)
            
            # Save
            self.save_results()
            
            print("\n" + "="*70)
            print("✅ BENCHMARK COMPLETE!")
            print("="*70)
            print(f"Results: {benchmark_config.RESULTS_DIR / self.test_mode}")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Benchmark interrupted by user")
            print("Partial results saved.")
            self.save_results()
            
        except Exception as e:
            print(f"\n❌ BENCHMARK FAILED: {e}")
            import traceback
            traceback.print_exc()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Hybrid VDB Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (100 queries)
  python benchmark.py --mode quick
  
  # Quick test with custom queries
  python benchmark.py --mode quick --queries-file queries/custom_queries.txt
  
  # Full test (1000 queries, 65/25/10 distribution)
  python benchmark.py --mode full
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['quick', 'full'],
        default='full',
        help='Test mode: quick (100) or full (1000 queries)'
    )
    
    parser.add_argument(
        '--queries-file',
        type=str,
        help='Path to custom queries file (for quick mode)'
    )
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = HybridVDBBenchmark(
        test_mode=args.mode,
        custom_queries_file=args.queries_file
    )
    benchmark.run()


if __name__ == "__main__":
    main()
