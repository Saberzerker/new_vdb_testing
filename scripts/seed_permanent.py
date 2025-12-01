# scripts/seed_permanent.py

"""
Seed local TIER 1 (permanent layer) with 10k vectors.

Fetches first 10k vectors from Qdrant Cloud and saves locally.
"""

import os
import sys
from pathlib import Path
import numpy as np
import faiss
import json
from tqdm import tqdm
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "pubmed_qa_full")
PERMANENT_DIR = Path(os.getenv("PERMANENT_DIR", "./data/permanent"))
TIER1_SIZE = 10_000


def main():
    print("="*70)
    print("SEEDING LOCAL TIER 1 (PERMANENT LAYER)")
    print("="*70)
    print(f"Target: {PERMANENT_DIR}")
    print(f"Vectors: {TIER1_SIZE:,}")
    print("="*70)
    
    PERMANENT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Connect
    print("\n[1/4] Connecting to Qdrant Cloud...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    info = client.get_collection(QDRANT_COLLECTION)
    print(f"✅ Cloud has {info.vectors_count:,} vectors")
    
    if info.vectors_count == 0:
        print("❌ Cloud is empty! Run seed_cloud.py first.")
        return
    
    # Fetch vectors
    print(f"\n[2/4] Fetching {TIER1_SIZE:,} vectors...")
    
    vectors = []
    metadata = {}
    
    offset = None
    batch_size = 100
    
    with tqdm(total=TIER1_SIZE, desc="Fetching") as pbar:
        while len(vectors) < TIER1_SIZE:
            results, offset = client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=batch_size,
                offset=offset,
                with_vectors=True
            )
            
            if not results:
                break
            
            for point in results:
                if len(vectors) >= TIER1_SIZE:
                    break
                
                vectors.append(np.array(point.vector, dtype='float32'))
                metadata[f"pubmed_{len(vectors)-1}"] = {
                    'cloud_id': point.id,
                    'question': point.payload.get('question', '')[:100],
                    'local_idx': len(vectors) - 1,
                    'partition_file': str(PERMANENT_DIR / "partition_0.index")
                }
                
                pbar.update(1)
            
            if offset is None:
                break
    
    print(f"✅ Fetched {len(vectors):,} vectors")
    
    # Create index
    print(f"\n[3/4] Creating FAISS HNSW index...")
    vectors_array = np.array(vectors, dtype='float32')
    
    index = faiss.IndexHNSWFlat(384, 16)
    index.hnsw.efConstruction = 200
    index.add(vectors_array)
    
    print(f"✅ Created index with {index.ntotal:,} vectors")
    
    # Save
    print(f"\n[4/4] Saving to {PERMANENT_DIR}...")
    
    index_file = PERMANENT_DIR / "partition_0.index"
    faiss.write_index(index, str(index_file))
    print(f"✅ Index: {index_file}")
    
    metadata_file = PERMANENT_DIR / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata: {metadata_file}")
    
    size_mb = (len(vectors) * 384 * 4 * 2) / (1024**2)
    print(f"\n📊 TIER 1 size: ~{size_mb:.0f} MB")
    
    print("\n" + "="*70)
    print("✅ TIER 1 SEEDED!")
    print("="*70)


if __name__ == "__main__":
    main()
