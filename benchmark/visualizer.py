# benchmark/visualizer.py

"""
Visualization utilities for benchmark results.

Generates comprehensive plots:
1. Latency over time
2. Source distribution
3. Anchor evolution
4. Cache utilization
5. Hit rate comparison
6. Learning curves
"""

from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from benchmark.benchmark_config import config as benchmark_config

sns.set_style("whitegrid")


class BenchmarkVisualizer:
    """
    Creates visualizations for benchmark results.
    """
    
    def __init__(self, test_mode: str):
        """
        Initialize visualizer.
        
        Args:
            test_mode: "quick" or "full"
        """
        self.test_mode = test_mode
        self.output_dir = benchmark_config.RESULTS_DIR / test_mode
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_plots(self, results: Dict):
        """
        Generate all visualization plots.
        
        Args:
            results: Benchmark results dictionary
        """
        if not benchmark_config.SAVE_PLOTS:
            return
        
        print("\n[PLOTS] Generating visualizations...")
        
        df = pd.DataFrame(results['per_query'])
        
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Latency over time
        self._plot_latency_over_time(fig.add_subplot(gs[0, :2]), df)
        
        # Plot 2: Source distribution (pie)
        self._plot_source_distribution(fig.add_subplot(gs[0, 2]), df)
        
        # Plot 3: Anchor evolution
        if len(results['anchor_snapshots']) > 0:
            anchor_df = pd.DataFrame(results['anchor_snapshots'])
            self._plot_anchor_evolution(fig.add_subplot(gs[1, :]), anchor_df)
        
        # Plot 4: Cache fill rate
        if len(results['cache_snapshots']) > 0:
            cache_df = pd.DataFrame(results['cache_snapshots'])
            self._plot_cache_fill_rate(fig.add_subplot(gs[2, 0]), cache_df)
            
            # Plot 5: Cache weight
            self._plot_cache_weight(fig.add_subplot(gs[2, 1]), cache_df)
        
        # Plot 6: Latency distribution
        self._plot_latency_distribution(fig.add_subplot(gs[2, 2]), df)
        
        # Save
        output_file = self.output_dir / "benchmark_plots.png"
        plt.savefig(output_file, dpi=benchmark_config.PLOT_DPI, bbox_inches='tight')
        print(f"[PLOTS] ✅ Saved to {output_file.name}")
        plt.close()
        
        # Generate per-query-type plots if applicable
        if 'query_type' in df.columns and df['query_type'].iloc[0] != 'unknown':
            self._generate_per_type_plots(df)
    
    def _plot_latency_over_time(self, ax, df):
        """Plot latency over time with moving average."""
        ax.plot(df['query_num'], df['latency_ms'], 
                alpha=0.3, linewidth=0.5, color='gray', label='Raw')
        
        window = 20 if self.test_mode == "quick" else 50
        if len(df) >= window:
            rolling = df['latency_ms'].rolling(window=window).mean()
            ax.plot(df['query_num'], rolling, 
                   linewidth=2, color='#e74c3c', label=f'Moving avg ({window})')
        
        ax.set_xlabel('Query Number', fontsize=11)
        ax.set_ylabel('Latency (ms)', fontsize=11)
        ax.set_title('Latency Over Time', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_source_distribution(self, ax, df):
        """Plot source distribution as pie chart."""
        source_counts = df['source'].value_counts()
        colors = {
            'tier1_permanent': '#e74c3c',
            'tier2_dynamic': '#2ecc71',
            'tier3_cloud': '#3498db',
            'offline_fallback': '#95a5a6'
        }
        
        pie_colors = [colors.get(s, '#999') for s in source_counts.index]
        
        ax.pie(source_counts.values, labels=source_counts.index,
              autopct='%1.1f%%', colors=pie_colors, startangle=90)
        ax.set_title('Query Source Distribution', fontsize=12, fontweight='bold')
    
    def _plot_anchor_evolution(self, ax, anchor_df):
        """Plot anchor evolution as stacked area chart."""
        ax.fill_between(anchor_df['query_num'], 0, 
                        anchor_df.get('weak_anchors', 0),
                        label='Weak', alpha=0.7, color='#e74c3c')
        
        ax.fill_between(anchor_df['query_num'], 
                        anchor_df.get('weak_anchors', 0),
                        anchor_df.get('weak_anchors', 0) + anchor_df.get('medium_anchors', 0),
                        label='Medium', alpha=0.7, color='#f39c12')
        
        ax.fill_between(anchor_df['query_num'],
                        anchor_df.get('weak_anchors', 0) + anchor_df.get('medium_anchors', 0),
                        anchor_df.get('weak_anchors', 0) + anchor_df.get('medium_anchors', 0) + 
                        anchor_df.get('strong_anchors', 0),
                        label='Strong', alpha=0.7, color='#2ecc71')
        
        ax.fill_between(anchor_df['query_num'],
                        anchor_df.get('weak_anchors', 0) + anchor_df.get('medium_anchors', 0) + 
                        anchor_df.get('strong_anchors', 0),
                        anchor_df.get('total_anchors', 0),
                        label='Permanent', alpha=0.7, color='#3498db')
        
        ax.set_xlabel('Query Number', fontsize=11)
        ax.set_ylabel('Anchor Count', fontsize=11)
        ax.set_title('Anchor System Evolution', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_cache_fill_rate(self, ax, cache_df):
        """Plot cache fill rate over time."""
        ax.plot(cache_df['query_num'], cache_df.get('fill_rate', 0),
               linewidth=2, color='#3498db')
        ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Capacity')
        ax.set_xlabel('Query Number', fontsize=11)
        ax.set_ylabel('Fill Rate (%)', fontsize=11)
        ax.set_title('Cache Fill Rate', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_cache_weight(self, ax, cache_df):
        """Plot average cache weight over time."""
        ax.plot(cache_df['query_num'], cache_df.get('avg_weight', 0),
               linewidth=2, color='#2ecc71')
        ax.set_xlabel('Query Number', fontsize=11)
        ax.set_ylabel('Average Weight', fontsize=11)
        ax.set_title('Cache Vector Quality', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_latency_distribution(self, ax, df):
        """Plot latency distribution histogram."""
        ax.hist(df['latency_ms'], bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        mean_lat = df['latency_ms'].mean()
        ax.axvline(mean_lat, color='red', linestyle='--',
                  label=f'Mean: {mean_lat:.1f}ms', linewidth=2)
        ax.set_xlabel('Latency (ms)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Latency Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _generate_per_type_plots(self, df):
        """Generate separate plots for each query type."""
        print("[PLOTS] Generating per-query-type plots...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, qtype in enumerate(['in_dataset', 'edge_case', 'out_of_distribution']):
            type_df = df[df['query_type'] == qtype]
            
            if len(type_df) == 0:
                continue
            
            ax = axes[idx]
            
            # Calculate rolling hit rate
            is_local = type_df['source'].isin(['tier1_permanent', 'tier2_dynamic']).astype(int)
            window = 20
            if len(type_df) >= window:
                rolling_rate = is_local.rolling(window=window).mean() * 100
                ax.plot(type_df['query_num'], rolling_rate, linewidth=2)
            
            ax.set_xlabel('Query Number', fontsize=11)
            ax.set_ylabel('Local Hit Rate (%)', fontsize=11)
            ax.set_title(qtype.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / "per_type_learning_curves.png"
        plt.savefig(output_file, dpi=benchmark_config.PLOT_DPI)
        print(f"[PLOTS] ✅ Saved to {output_file.name}")
        plt.close()
