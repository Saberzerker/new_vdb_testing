# src/anchor_system.py
"""
Anchor System - Momentum-Based Trajectory Prediction

Think of anchors as "pins on a map" of your journey through cookie land.

How it works:
1. Each query creates an anchor (pin)
2. Anchors remember: "what did user ask after this?"
3. Strong anchors (many hits) = reliable paths
4. Weak anchors (no hits) = dead ends, decay over time

NO machine learning. Just:
- Weighted vectors (momentum toward proven paths)
- Graph structure (parent-child relationships)
- Reinforcement learning (hits strengthen, misses weaken)

Author: Saberzerker
Date: 2025-11-17 00:35 UTC
"""

import numpy as np
import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.config import (
    ANCHOR_WEAK_THRESHOLD,
    ANCHOR_MEDIUM_THRESHOLD,
    ANCHOR_STRONG_THRESHOLD,
    ANCHOR_PERMANENT_THRESHOLD,
    PREDICTION_SIMILARITY_THRESHOLD,
    PREFETCH_K,
    VECTOR_DIMENSION,
)

logger = logging.getLogger(__name__)


class AnchorType(Enum):
    """Anchor strength types (like star ratings ⭐)."""

    WEAK = "weak"  # 0-25 strength (⭐)
    MEDIUM = "medium"  # 25-60 strength (⭐⭐⭐)
    STRONG = "strong"  # 60-90 strength (⭐⭐⭐⭐⭐)
    PERMANENT = "permanent"  # 90+ strength (⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐)


@dataclass
class Anchor:
    """
    An anchor is a "pin" in semantic space.

    Represents a query and tracks:
    - Where user went AFTER this query
    - How often predictions were correct
    - Relationship to other anchors (parent/children)
    """

    id: int
    vector: np.ndarray
    query_id: str
    query_text: str
    strength: float
    anchor_type: AnchorType
    created_at: float
    last_accessed: float
    hits: int
    misses: int
    parent_id: Optional[int]
    children_ids: List[int]
    predictions: List[np.ndarray]
    metadata: Dict


class AnchorSystem:
    """
    Manages anchor graph and generates trajectory predictions.

    Key operations:
    1. create_anchor() - Add new pin to map
    2. generate_predictions() - Predict where user goes next
    3. check_prediction_match() - Did we predict correctly?
    4. strengthen_anchor() - Reward correct predictions
    5. decay_weak_anchors() - Remove dead ends
    """

    def __init__(self):
        """Initialize anchor system."""
        self.anchors = {}  # {anchor_id: Anchor}
        self.anchor_counter = 0

        # Active predictions for matching
        self.active_predictions = {}  # {prediction_id: (vector, anchor_id)}
        self.prediction_counter = 0

        logger.info("[ANCHORS] Initialized anchor system")

    # ═══════════════════════════════════════════════════════════
    # ANCHOR CREATION
    # ═══════════════════════════════════════════════════════════

    def create_anchor(
        self,
        query_vector: np.ndarray,
        query_id: str,
        query_text: str = "",
        parent_anchor_id: Optional[int] = None,
    ) -> int:
        """
        Create new anchor (pin on map).

        Args:
            query_vector: Query embedding
            query_id: Unique query identifier
            query_text: Original query text
            parent_anchor_id: If query was predicted, ID of anchor that predicted it

        Returns:
            anchor_id: New anchor's ID
        """
        anchor_id = self.anchor_counter
        self.anchor_counter += 1

        anchor = Anchor(
            id=anchor_id,
            vector=query_vector.copy(),
            query_id=query_id,
            query_text=query_text,
            strength=(
                0.0 if parent_anchor_id is None else 1.0
            ),  # Child anchors start with 1.0
            anchor_type=AnchorType.WEAK,
            created_at=time.time(),
            last_accessed=time.time(),
            hits=0,
            misses=0,
            parent_id=parent_anchor_id,
            children_ids=[],
            predictions=[],
            metadata={},
        )

        self.anchors[anchor_id] = anchor

        # Link to parent
        if parent_anchor_id is not None and parent_anchor_id in self.anchors:
            self.anchors[parent_anchor_id].children_ids.append(anchor_id)
            logger.debug(
                f"[ANCHORS] Created Anchor #{anchor_id} (child of #{parent_anchor_id})"
            )
        else:
            logger.debug(f"[ANCHORS] Created Anchor #{anchor_id} (root)")

        return anchor_id

    # ═══════════════════════════════════════════════════════════
    # PREDICTION GENERATION (THE MAGIC!)
    # ═══════════════════════════════════════════════════════════

    def generate_predictions(
        self,
        anchor_id: int,
        centroid: Optional[np.ndarray] = None,
        count: int = PREFETCH_K,
        noise_scale: float = 0.15,
    ) -> List[np.ndarray]:
        """
        Generate trajectory predictions from anchor.

        Strategy:
        - If centroid exists: Bias predictions TOWARD centroid (momentum)
        - If no centroid: Random walk from anchor
        - Noise: Controlled by noise_scale (exploration vs exploitation)

        Args:
            anchor_id: Anchor to generate from
            centroid: Cluster centroid to guide trajectory
            count: Number of predictions
            noise_scale: Amount of random exploration (0.0-0.5)

        Returns:
            List of prediction vectors
        """
        if anchor_id not in self.anchors:
            logger.warning(f"[ANCHORS] Anchor #{anchor_id} not found")
            return []

        anchor = self.anchors[anchor_id]
        anchor_vec = anchor.vector

        predictions = []

        if centroid is not None:
            # TRAJECTORY-GUIDED (toward cluster centroid)
            # This implements MOMENTUM!

            # Calculate success rate for this anchor
            total_predictions = anchor.hits + anchor.misses
            success_rate = anchor.hits / max(total_predictions, 1)

            # High success → High momentum (stay close to path)
            # Low success → Low momentum (explore more)
            momentum = 0.5 + (0.4 * success_rate)  # Range: 0.5 to 0.9

            for i in range(count):
                # Interpolate between anchor and centroid
                alpha = (i + 1) / (count + 1)  # 0.16, 0.33, 0.5, 0.66, 0.83

                # Momentum-based prediction
                pred = momentum * centroid + (1 - momentum) * anchor_vec

                # Add adaptive noise
                noise = np.random.normal(0, noise_scale, pred.shape)
                pred = pred + noise

                # Normalize
                pred = pred / np.linalg.norm(pred)

                predictions.append(pred)

            logger.debug(
                f"[PREDICTIONS] Generated {count} trajectory predictions "
                f"(momentum={momentum:.2f})"
            )

        else:
            # RANDOM WALK (no guidance yet)
            for i in range(count):
                # Random perturbation around anchor
                noise = np.random.normal(0, noise_scale * 2, anchor_vec.shape)
                pred = anchor_vec + noise
                pred = pred / np.linalg.norm(pred)

                predictions.append(pred)

            logger.debug(f"[PREDICTIONS] Generated {count} random walk predictions")

        # Store predictions in anchor
        anchor.predictions = predictions

        return predictions

    def generate_trajectory_prediction(
        self, query_vector: np.ndarray, target_vector: np.ndarray, momentum: float = 0.9
    ) -> np.ndarray:
        """
        Generate single prediction along trajectory.

        Simple momentum formula:
          prediction = momentum * target + (1 - momentum) * query

        Args:
            query_vector: Current query
            target_vector: Proven path target (strong anchor)
            momentum: How much to trust proven path (0.0-1.0)

        Returns:
            Prediction vector
        """
        pred = momentum * target_vector + (1 - momentum) * query_vector
        pred = pred / np.linalg.norm(pred)
        return pred

    def register_prediction(self, prediction_vector: np.ndarray, anchor_id: int):
        """
        Register prediction for future matching.

        When user asks next query, we check if it matches any active predictions.
        """
        pred_id = self.prediction_counter
        self.prediction_counter += 1

        self.active_predictions[pred_id] = (prediction_vector.copy(), anchor_id)

        logger.debug(
            f"[PREDICTIONS] Registered prediction #{pred_id} from Anchor #{anchor_id}"
        )

    # ═══════════════════════════════════════════════════════════
    # PREDICTION MATCHING (THE LEARNING!)
    # ═══════════════════════════════════════════════════════════

    def check_prediction_match(
        self, query_vector: np.ndarray, query_id: str
    ) -> Optional[Tuple[int, int, float]]:
        """
        Check if query matches any active predictions.

        This is where LEARNING happens!

        Args:
            query_vector: New query
            query_id: Query identifier

        Returns:
            (prediction_id, source_anchor_id, similarity) if match found
            None if no match
        """
        if not self.active_predictions:
            return None

        best_match = None
        best_similarity = 0.0
        best_pred_id = None
        best_anchor_id = None

        for pred_id, (pred_vec, anchor_id) in self.active_predictions.items():
            # Calculate similarity
            similarity = self._cosine_similarity(query_vector, pred_vec)

            if similarity > best_similarity:
                best_similarity = similarity
                best_pred_id = pred_id
                best_anchor_id = anchor_id
                best_match = pred_id

        # Check threshold
        if best_similarity >= PREDICTION_SIMILARITY_THRESHOLD:
            # MATCH!
            logger.info(
                f"[MATCH] 🎯 Query matched prediction #{best_pred_id} "
                f"from Anchor #{best_anchor_id} (sim={best_similarity:.3f})"
            )

            # Remove matched prediction (it's been used)
            del self.active_predictions[best_pred_id]

            return best_pred_id, best_anchor_id, best_similarity

        return None

    # ═══════════════════════════════════════════════════════════
    # REINFORCEMENT (STRENGTHEN/WEAKEN)
    # ═══════════════════════════════════════════════════════════

    def strengthen_anchor(self, anchor_id: int, reward: float = 1.0):
        """
        Strengthen anchor after correct prediction.

        Like adding stars ⭐ to a restaurant.

        Args:
            anchor_id: Anchor that made correct prediction
            reward: Strength increase (default 1.0)
        """
        if anchor_id not in self.anchors:
            return

        anchor = self.anchors[anchor_id]

        # Increase strength
        old_strength = anchor.strength
        anchor.strength += reward
        anchor.hits += 1
        anchor.last_accessed = time.time()

        # Update type based on strength
        old_type = anchor.anchor_type
        anchor.anchor_type = self._determine_anchor_type(anchor.strength)

        # Propagate to parent (they deserve credit too!)
        if anchor.parent_id is not None and anchor.parent_id in self.anchors:
            parent = self.anchors[anchor.parent_id]
            parent.strength += reward * 0.5  # Half credit
            parent.anchor_type = self._determine_anchor_type(parent.strength)

        logger.info(
            f"[REINFORCE] Anchor #{anchor_id}: "
            f"{old_strength:.1f} → {anchor.strength:.1f} "
            f"({old_type.value} → {anchor.anchor_type.value})"
        )

    def weaken_anchor(self, anchor_id: int, penalty: float = 0.5):
        """
        Weaken anchor after incorrect prediction.

        Like removing stars ⭐ from a restaurant.

        Args:
            anchor_id: Anchor that made wrong prediction
            penalty: Strength decrease (default 0.5)
        """
        if anchor_id not in self.anchors:
            return

        anchor = self.anchors[anchor_id]

        # Decrease strength
        old_strength = anchor.strength
        anchor.strength = max(0.0, anchor.strength - penalty)
        anchor.misses += 1

        # Update type
        old_type = anchor.anchor_type
        anchor.anchor_type = self._determine_anchor_type(anchor.strength)

        logger.debug(
            f"[WEAKEN] Anchor #{anchor_id}: "
            f"{old_strength:.1f} → {anchor.strength:.1f} "
            f"({old_type.value} → {anchor.anchor_type.value})"
        )

    def _determine_anchor_type(self, strength: float) -> AnchorType:
        """Determine anchor type based on strength."""
        if strength >= ANCHOR_PERMANENT_THRESHOLD:
            return AnchorType.PERMANENT  # ⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐
        elif strength >= ANCHOR_STRONG_THRESHOLD:
            return AnchorType.STRONG  # ⭐⭐⭐⭐⭐
        elif strength >= ANCHOR_MEDIUM_THRESHOLD:
            return AnchorType.MEDIUM  # ⭐⭐⭐
        else:
            return AnchorType.WEAK  # ⭐

    # ═══════════════════════════════════════════════════════════
    # DECAY & EVICTION
    # ═══════════════════════════════════════════════════════════

    def decay_weak_anchors(self, decay_rate: float = 0.1):
        """
        Decay WEAK anchors over time (like stale cookies).

        Strong anchors decay slowly.
        Weak anchors decay quickly.
        Permanent anchors don't decay.
        """
        current_time = time.time()
        to_remove = []

        for anchor_id, anchor in self.anchors.items():
            idle_time = current_time - anchor.last_accessed

            # Decay based on type
            if anchor.anchor_type == AnchorType.WEAK:
                if idle_time > 3600:  # 1 hour idle
                    anchor.strength *= 1 - decay_rate

                    if anchor.strength < 0.1:
                        to_remove.append(anchor_id)

            elif anchor.anchor_type == AnchorType.MEDIUM:
                if idle_time > 21600:  # 6 hours idle
                    anchor.strength *= 1 - decay_rate * 0.5

            elif anchor.anchor_type == AnchorType.STRONG:
                if idle_time > 86400:  # 24 hours idle
                    anchor.strength *= 1 - decay_rate * 0.2

            # PERMANENT never decays

        # Remove dead anchors
        for anchor_id in to_remove:
            del self.anchors[anchor_id]
            logger.debug(f"[DECAY] Removed Anchor #{anchor_id} (too weak)")

        if to_remove:
            logger.info(f"[DECAY] Removed {len(to_remove)} weak anchors")

    # ═══════════════════════════════════════════════════════════
    # QUERIES
    # ═══════════════════════════════════════════════════════════

    def get_strong_anchors(self) -> List[Anchor]:
        """Get all STRONG or PERMANENT anchors."""
        return [
            anchor
            for anchor in self.anchors.values()
            if anchor.anchor_type in [AnchorType.STRONG, AnchorType.PERMANENT]
        ]

    def get_anchor_by_id(self, anchor_id: int) -> Optional[Anchor]:
        """Get specific anchor by ID."""
        return self.anchors.get(anchor_id)

    def get_anchor_stats(self) -> Dict:
        """Get comprehensive anchor statistics."""
        if not self.anchors:
            return {
                "total_anchors": 0,
                "active_predictions": 0,
                "anchor_types": {},
                "avg_strength": 0.0,
                "prediction_accuracy": 0.0,
            }

        # Count by type
        type_counts = {"weak": 0, "medium": 0, "strong": 0, "permanent": 0}

        total_hits = 0
        total_predictions = 0
        total_strength = 0.0

        for anchor in self.anchors.values():
            type_counts[anchor.anchor_type.value] += 1
            total_hits += anchor.hits
            total_predictions += anchor.hits + anchor.misses
            total_strength += anchor.strength

        prediction_accuracy = (
            total_hits / total_predictions * 100 if total_predictions > 0 else 0.0
        )

        return {
            "total_anchors": len(self.anchors),
            "active_predictions": len(self.active_predictions),
            "anchor_types": type_counts,
            "avg_strength": total_strength / len(self.anchors),
            "max_strength": max(a.strength for a in self.anchors.values()),
            "prediction_accuracy": prediction_accuracy,
            "total_hits": total_hits,
            "total_misses": total_predictions - total_hits,
        }

    def get_anchor_graph(self) -> Dict:
        """
        Get anchor graph structure for visualization.

        Returns:
            Dict with nodes and edges for graph rendering
        """
        nodes = []
        edges = []

        for anchor in self.anchors.values():
            nodes.append(
                {
                    "id": anchor.id,
                    "label": anchor.query_text[:30],
                    "type": anchor.anchor_type.value,
                    "strength": anchor.strength,
                    "hits": anchor.hits,
                    "misses": anchor.misses,
                }
            )

            # Add edges to children
            for child_id in anchor.children_ids:
                edges.append(
                    {"from": anchor.id, "to": child_id, "weight": anchor.strength}
                )

        return {"nodes": nodes, "edges": edges}

    # ═══════════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════════

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return dot / norm if norm > 0 else 0.0
