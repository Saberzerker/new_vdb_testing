# src/persistence.py
"""
Persistence layer for Hybrid VDB system.
Saves/loads FAISS index, anchor state, and metrics to disk.

This enables:
- Resuming from previous sessions
- State inspection and debugging
- Long-term learning across restarts
"""
import faiss
import pickle
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
from datetime import datetime

from src.config import *


class PersistenceManager:
    """
    Manages saving and loading of all system state.
    """
    
    def __init__(self, storage_dir: str = "storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Define paths
        self.faiss_index_path = self.storage_dir / "local_faiss.index"
        self.local_ids_path = self.storage_dir / "local_ids.pkl"
        self.local_metadata_path = self.storage_dir / "local_metadata.pkl"
        self.anchor_state_path = self.storage_dir / "anchor_state.pkl"
        self.semantic_cache_path = self.storage_dir / "semantic_cache.pkl"
        self.metrics_path = self.storage_dir / "metrics.json"
        
        print(f"[PERSISTENCE] Storage directory: {self.storage_dir.absolute()}")
    
    def save_faiss_index(self, index: faiss.Index, local_ids: list, local_metadata: list):
        """
        Save FAISS index and associated metadata.
        """
        try:
            # Save FAISS index
            faiss.write_index(index, str(self.faiss_index_path))
            
            # Save IDs and metadata
            with open(self.local_ids_path, 'wb') as f:
                pickle.dump(local_ids, f)
            
            with open(self.local_metadata_path, 'wb') as f:
                pickle.dump(local_metadata, f)
            
            print(f"[PERSISTENCE] ✅ Saved FAISS index ({index.ntotal} vectors)")
            return True
            
        except Exception as e:
            print(f"[PERSISTENCE] ❌ Failed to save FAISS index: {e}")
            return False
    
    def load_faiss_index(self, dimension: int):
        """
        Load FAISS index from disk.
        Returns: (index, local_ids, local_metadata) or (None, [], [])
        """
        try:
            if not self.faiss_index_path.exists():
                print("[PERSISTENCE] No saved FAISS index found")
                return None, [], []
            
            # Load FAISS index
            index = faiss.read_index(str(self.faiss_index_path))
            
            # Load IDs
            with open(self.local_ids_path, 'rb') as f:
                local_ids = pickle.load(f)
            
            # Load metadata
            with open(self.local_metadata_path, 'rb') as f:
                local_metadata = pickle.load(f)
            
            print(f"[PERSISTENCE] ✅ Loaded FAISS index ({index.ntotal} vectors)")
            return index, local_ids, local_metadata
            
        except Exception as e:
            print(f"[PERSISTENCE] ❌ Failed to load FAISS index: {e}")
            return None, [], []
    
    def save_anchor_system(self, anchor_system):
        """
        Save complete anchor system state.
        Note: We only save serializable data, not numpy arrays.
        """
        try:
            state = {
                "anchor_counter": anchor_system.anchor_counter,
                "prediction_counter": anchor_system.prediction_counter,
                "metrics": anchor_system.metrics,
                "anchors": {
                    aid: {
                        "id": a.id,
                        "query_id": a.query_id,
                        "query_text": a.query_text,
                        "strength": a.strength,
                        "hits": a.hits,
                        "misses": a.misses,
                        "type": a.type.value,
                        "created_at": a.created_at,
                        "last_hit": a.last_hit,
                        "last_decay": a.last_decay,
                        "predictions_generated": a.predictions_generated,
                        "successful_predictions": a.successful_predictions,
                        "parent_anchor": a.parent_anchor,
                        "child_anchors": a.child_anchors,
                        "vector": a.vector.tolist()  # Convert to list for JSON
                    }
                    for aid, a in anchor_system.anchors.items()
                },
                "predictions": {
                    pid: {
                        "id": p.id,
                        "source_anchor_id": p.source_anchor_id,
                        "created_at": p.created_at,
                        "status": p.status,
                        "matched_query_id": p.matched_query_id,
                        "similarity_score": p.similarity_score,
                        "vector": p.vector.tolist()
                    }
                    for pid, p in anchor_system.predictions.items()
                }
            }
            
            with open(self.anchor_state_path, 'wb') as f:
                pickle.dump(state, f)
            
            print(f"[PERSISTENCE] ✅ Saved anchor system ({len(state['anchors'])} anchors)")
            return True
            
        except Exception as e:
            print(f"[PERSISTENCE] ❌ Failed to save anchor system: {e}")
            return False
    
    def load_anchor_system(self):
        """
        Load anchor system state.
        Returns: state dict or None
        """
        try:
            if not self.anchor_state_path.exists():
                print("[PERSISTENCE] No saved anchor state found")
                return None
            
            with open(self.anchor_state_path, 'rb') as f:
                state = pickle.load(f)
            
            print(f"[PERSISTENCE] ✅ Loaded anchor system ({len(state['anchors'])} anchors)")
            return state
            
        except Exception as e:
            print(f"[PERSISTENCE] ❌ Failed to load anchor system: {e}")
            return None
    
    def save_semantic_cache(self, semantic_cache):
        """Save semantic cache state."""
        try:
            state = {
                "cluster_counter": semantic_cache.cluster_counter,
                "query_to_cluster": semantic_cache.query_to_cluster,
                "metrics": semantic_cache.metrics,
                "clusters": {
                    cid: {
                        "centroid": c["centroid"].tolist(),
                        "queries": c["queries"],
                        "created_at": c["created_at"],
                        "last_access": c["last_access"],
                        "access_count": c["access_count"],
                        "total_weight": c["total_weight"]
                    }
                    for cid, c in semantic_cache.clusters.items()
                }
            }
            
            with open(self.semantic_cache_path, 'wb') as f:
                pickle.dump(state, f)
            
            print(f"[PERSISTENCE] ✅ Saved semantic cache ({len(state['clusters'])} clusters)")
            return True
            
        except Exception as e:
            print(f"[PERSISTENCE] ❌ Failed to save semantic cache: {e}")
            return False
    
    def load_semantic_cache(self):
        """Load semantic cache state."""
        try:
            if not self.semantic_cache_path.exists():
                print("[PERSISTENCE] No saved semantic cache found")
                return None
            
            with open(self.semantic_cache_path, 'rb') as f:
                state = pickle.load(f)
            
            print(f"[PERSISTENCE] ✅ Loaded semantic cache ({len(state['clusters'])} clusters)")
            return state
            
        except Exception as e:
            print(f"[PERSISTENCE] ❌ Failed to load semantic cache: {e}")
            return None
    
    def save_metrics(self, metrics: Dict[str, Any]):
        """Save metrics to JSON."""
        try:
            # Add timestamp
            metrics["saved_at"] = datetime.utcnow().isoformat()
            
            with open(self.metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            print(f"[PERSISTENCE] ✅ Saved metrics")
            return True
            
        except Exception as e:
            print(f"[PERSISTENCE] ❌ Failed to save metrics: {e}")
            return False
    
    def save_all(self, vdb):
        """
        Save complete system state.
        Call this after important operations or before shutdown.
        """
        print(f"\n[PERSISTENCE] Saving complete system state...")
        
        success = True
        success &= self.save_faiss_index(vdb.local_index, vdb.local_ids, vdb.local_metadata)
        success &= self.save_anchor_system(vdb.anchor_system)
        success &= self.save_semantic_cache(vdb.semantic_cache)
        success &= self.save_metrics(vdb.metrics)
        
        if success:
            print(f"[PERSISTENCE] ✅ Complete system saved\n")
        else:
            print(f"[PERSISTENCE] ⚠️ Some components failed to save\n")
        
        return success
    
    def clear_all(self):
        """Clear all saved state (fresh start)."""
        try:
            for path in [self.faiss_index_path, self.local_ids_path, 
                        self.local_metadata_path, self.anchor_state_path,
                        self.semantic_cache_path, self.metrics_path]:
                if path.exists():
                    path.unlink()
            
            print("[PERSISTENCE] ✅ Cleared all saved state")
            return True
            
        except Exception as e:
            print(f"[PERSISTENCE] ❌ Failed to clear state: {e}")
            return False