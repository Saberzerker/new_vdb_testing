# scripts/seed_cloud_all.py
"""
Seed Qdrant Cloud with MAXIMUM medical conversations.
Uses all 256,916 conversations OR configurable limit.

Author: Saberzerker
Date: 2025-11-16
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def install_dependencies():
    import subprocess
    required = ['datasets', 'sentence-transformers', 'qdrant-client']
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"📦 Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
    print()


def main():
    print("\n" + "="*70)
    print("SEEDING QDRANT CLOUD - MAXIMUM MEDICAL DATASET")
    print("="*70 + "\n")
    
    install_dependencies()
    
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    
    from src.config import (
        QDRANT_URL,
        QDRANT_API_KEY,
        QDRANT_COLLECTION_NAME,
        VECTOR_DIMENSION
    )
    
    # ================================================================
    # CONFIGURATION: Change this to control dataset size
    # ================================================================
    
    # Option 1: ALL conversations (256,916) - takes 30-40 minutes
    USE_ALL = False
    
    # Option 2: Limited for faster demo (recommended)
    MAX_DOCUMENTS = 5000  # Good balance: large corpus, reasonable time
    
    # ================================================================
    
    print("="*70)
    print("STEP 1: LOADING DATASET")
    print("="*70 + "\n")
    
    print("📥 Loading ruslanmv/ai-medical-chatbot...")
    dataset = load_dataset("ruslanmv/ai-medical-chatbot", split="train")
    total_available = len(dataset)
    print(f"✅ Dataset loaded: {total_available} conversations available\n")
    
    # Determine how many to use
    if USE_ALL:
        max_docs = total_available
        print(f"🚀 Using ALL {max_docs} conversations")
        print(f"⏳ Estimated time: 30-40 minutes\n")
    else:
        max_docs = min(MAX_DOCUMENTS, total_available)
        print(f"📊 Using {max_docs} conversations (configurable)")
        print(f"⏳ Estimated time: 8-12 minutes\n")
    
    response = input(f"Proceed with {max_docs} documents? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled.")
        return
    
    # Extract documents
    print("\n📝 Extracting medical conversations...")
    documents = []
    
    for idx, item in enumerate(dataset):
        if idx >= max_docs:
            break
        
        # Extract fields: Description, Patient, Doctor
        description = item.get('Description', '')
        patient = item.get('Patient', '')
        doctor = item.get('Doctor', '')
        
        if patient and doctor:
            # Format 1: Full conversation
            doc = f"Medical Question: {description}\n\nPatient: {patient}\n\nDoctor: {doctor}"
            documents.append(doc)
            
            # Format 2: Doctor's answer alone (for retrieval diversity)
            if len(doctor) > 150:
                documents.append(f"Medical Answer: {doctor}")
        
        if (idx + 1) % 1000 == 0:
            print(f"   Processed {idx + 1}/{max_docs}... ({len(documents)} documents)")
    
    print(f"\n✅ Extracted {len(documents)} medical documents\n")
    
    # Upload to Qdrant
    print("="*70)
    print("STEP 2: UPLOADING TO QDRANT CLOUD")
    print("="*70 + "\n")
    
    print(f"📚 Documents: {len(documents)}")
    print(f"☁️  Target: {QDRANT_COLLECTION_NAME}\n")
    
    # Connect
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Recreate collection
    try:
        client.delete_collection(QDRANT_COLLECTION_NAME)
        print("✅ Cleared existing collection\n")
    except:
        pass
    
    client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIMENSION, distance=Distance.COSINE)
    )
    print("✅ Created collection\n")
    
    # Load model
    print("Loading sentence transformer...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded\n")
    
    # Upload in batches
    print(f"🚀 Uploading {len(documents)} documents...")
    print(f"⏳ Estimated: {len(documents) // 200} minutes\n")
    
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        embeddings = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        
        points = [
            PointStruct(
                id=i + j,
                vector=embeddings[j].tolist(),
                payload={"text": doc[:1000]}  # Limit payload size
            )
            for j, doc in enumerate(batch)
        ]
        
        client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)
        
        progress = min(i + batch_size, len(documents))
        if progress % 500 == 0 or progress == len(documents):
            percentage = (progress / len(documents)) * 100
            print(f"   ✅ {progress}/{len(documents)} ({percentage:.1f}%)")
    
    # Verify
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70 + "\n")
    
    info = client.get_collection(QDRANT_COLLECTION_NAME)
    
    print("="*70)
    print("✅ CLOUD SEEDING COMPLETE")
    print("="*70)
    print(f"\n☁️  Vectors in Cloud: {info.points_count}")
    print(f"📖 Source: ruslanmv/ai-medical-chatbot")
    print(f"🏥 Coverage: Comprehensive medical Q&A")
    print(f"🤖 Status: Ready for RAG!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted.")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()