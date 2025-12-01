# scripts/download_dataset.py

"""
Pre-download PubMedQA dataset for faster benchmarking.

Downloads and caches the dataset locally.
"""

import sys
from pathlib import Path
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.benchmark_config import config


def main():
    print("="*70)
    print("DOWNLOADING PUBMEDQA DATASET")
    print("="*70)
    print(f"Dataset: {config.DATASET_NAME}")
    print(f"Subset: {config.DATASET_SUBSET}")
    print(f"Cache: {config.DATA_DIR / 'cache'}")
    print("="*70)
    
    print("\n[1/2] Downloading dataset...")
    dataset = load_dataset(
        config.DATASET_NAME,
        config.DATASET_SUBSET,
        split="train",
        cache_dir=str(config.DATA_DIR / "cache")
    )
    
    print(f"✅ Downloaded {len(dataset):,} documents")
    
    print("\n[2/2] Downloading embedding model...")
    from sentence_transformers import SentenceTransformer
    
    embedder = SentenceTransformer(
        config.EMBEDDING_MODEL,
        cache_folder=str(config.MODEL_CACHE_DIR)
    )
    
    print(f"✅ Downloaded {config.EMBEDDING_MODEL}")
    
    print("\n" + "="*70)
    print("✅ DOWNLOAD COMPLETE!")
    print("="*70)
    print("\nDataset and model are cached locally.")
    print("Benchmarks will run faster now.")


if __name__ == "__main__":
    main()
