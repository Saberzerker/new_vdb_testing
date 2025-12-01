# scripts/seed_cloud.py

"""
Seed Qdrant Cloud with full PubMedQA dataset.

Uploads ~211,000 vectors in batches.
Estimated time: 30-60 minutes
Estimated storage: ~850 MB
"""

import os
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "pubmed_qa_full")
BATCH_SIZE = 100
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    print("="*70)
    print("SEEDING QDRANT CLOUD WITH PUBMEDQA")
    print("="*70)
    print(f"Target: {QDRANT_URL}")
    print(f"Collection: {QDRANT_COLLECTION}")
    print(f"Batch size: {BATCH_SIZE}")
    print("="*70)
    print("\n⚠️  This will take 30-60 minutes and upload ~850 MB")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Connect
    print("\n[1/6] Connecting to Qdrant Cloud...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    print("✅ Connected")
    
    # Check/create collection
    print(f"\n[2/6] Checking collection '{QDRANT_COLLECTION}'...")
    try:
        client.get_collection(QDRANT_COLLECTION)
        print("✅ Collection exists")
        
        response = input("⚠️  Delete and recreate? (y/n): ")
        if response.lower() == 'y':
            client.delete_collection(QDRANT_COLLECTION)
            print("🗑️  Deleted")
            
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            print("✅ Created new collection")
    except:
        print("⚠️  Creating collection...")
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print("✅ Created")
    
    # Load dataset
    print("\n[3/6] Loading PubMedQA dataset...")
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    total_docs = len(dataset)
    print(f"✅ Loaded {total_docs:,} documents")
    
    # Load model
    print("\n[4/6] Loading embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    print(f"✅ Loaded {EMBEDDING_MODEL}")
    
    # Process and upload
    print(f"\n[5/6] Processing and uploading {total_docs:,} documents...")
    
    points = []
    uploaded = 0
    
    for i in tqdm(range(total_docs), desc="Uploading"):
        item = dataset[i]
        
        # Extract text
        question = item.get('question', '')
        context = ' '.join(item.get('context', {}).get('contexts', []))
        full_text = f"{question} {context}"
        
        # Generate embedding
        embedding = embedder.encode(full_text, convert_to_numpy=True)
        
        # Create point
        point = PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload={
                'question': question,
                'context_snippet': context[:200],
                'source': 'pubmed_qa',
                'index': i
            }
        )
        points.append(point)
        
        # Upload batch
        if len(points) >= BATCH_SIZE:
            client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=points
            )
            uploaded += len(points)
            points = []
    
    # Upload remaining
    if points:
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points
        )
        uploaded += len(points)
    
    print(f"\n✅ Uploaded {uploaded:,} vectors")
    
    # Verify
    print("\n[6/6] Verifying...")
    info = client.get_collection(QDRANT_COLLECTION)
    print(f"✅ Collection has {info.vectors_count:,} vectors")
    
    storage_mb = info.vectors_count * 384 * 4 / (1024**2)
    print(f"📊 Storage: ~{storage_mb:.0f} MB")
    
    print("\n" + "="*70)
    print("✅ SEEDING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
