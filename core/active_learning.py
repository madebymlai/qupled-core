"""
Active learning classifier for knowledge item grouping.

Learns from past LLM decisions to reduce future LLM calls by 70-90%.
Uses Query by Committee (QBC) for uncertainty estimation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from core.features import PairFeatures, extract_features, should_add_to_training
from core.transitive import TransitiveInference

logger = logging.getLogger(__name__)


@dataclass
class TrainingRecord:
    """A single training record from past LLM decision."""

    features: list[float]  # Feature vector (anonymous)
    label: int  # 1 = match, 0 = no match


class ActiveLearner:
    """
    Query-by-Committee active learner for pair classification.

    Uses ensemble of RandomForest classifiers.
    Uncertainty = disagreement between committee members.
    """

    def __init__(self, n_estimators: int = 3):
        self.committee = [
            RandomForestClassifier(n_estimators=50, random_state=i, n_jobs=1)
            for i in range(n_estimators)
        ]
        self.X_train: list[np.ndarray] = []
        self.y_train: list[int] = []
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit all committee members."""
        if len(X) < 2:
            return

        # Need at least one of each class
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            return

        for clf in self.committee:
            clf.fit(X, y)
        self.X_train = list(X)
        self.y_train = list(y)
        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get average probability from committee."""
        if not self.is_fitted:
            return np.full((len(X), 2), 0.5)

        probas = np.array([clf.predict_proba(X) for clf in self.committee])
        return probas.mean(axis=0)

    def uncertainty(self, X: np.ndarray) -> np.ndarray:
        """Compute uncertainty via committee disagreement."""
        if not self.is_fitted:
            return np.ones(len(X))

        # Get predictions from each committee member
        predictions = np.array([clf.predict(X) for clf in self.committee])

        # Disagreement = entropy of committee votes
        vote_counts = predictions.sum(axis=0)  # Count positive votes
        p_positive = vote_counts / len(self.committee)

        # Entropy (avoiding log(0))
        epsilon = 1e-10
        p_pos = np.clip(p_positive, epsilon, 1 - epsilon)
        entropy = -p_pos * np.log2(p_pos) - (1 - p_pos) * np.log2(1 - p_pos)

        return entropy

    def teach(self, X: np.ndarray, y: np.ndarray) -> None:
        """Add new labeled data and retrain."""
        self.X_train.extend(X)
        self.y_train.extend(y)

        X_all = np.array(self.X_train)
        y_all = np.array(self.y_train)

        self.fit(X_all, y_all)


@dataclass
class ClassificationResult:
    """Result of classifying an item."""

    group_id: str | None  # ID of matched group, or None if new
    is_new: bool  # Whether item is new (no match found)
    confidence: float  # Confidence in decision (0-1)
    method: str  # "prediction", "llm", or "transitive"


@dataclass
class ActiveClassifierStats:
    """Statistics for active classifier."""

    llm_calls: int = 0
    predictions: int = 0
    transitive_inferences: int = 0
    training_samples: int = 0

    @property
    def total(self) -> int:
        return self.llm_calls + self.predictions + self.transitive_inferences

    @property
    def llm_call_rate(self) -> float:
        return self.llm_calls / max(self.total, 1)

    @property
    def prediction_rate(self) -> float:
        return self.predictions / max(self.total, 1)

    def to_dict(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "predictions": self.predictions,
            "transitive_inferences": self.transitive_inferences,
            "total": self.total,
            "llm_call_rate": round(self.llm_call_rate, 3),
            "prediction_rate": round(self.prediction_rate, 3),
            "training_samples": self.training_samples,
        }


@dataclass
class ActiveClassifier:
    """
    Active learning-powered item classifier.

    Learns from past LLM decisions to reduce future LLM calls.
    Training data is examina-wide (features are anonymous).
    Transitive graph is per-session (item relationships).
    """

    high_confidence: float = 0.85  # Above this -> use prediction
    low_confidence: float = 0.15  # Below this -> use prediction (negative)
    min_training_samples: int = 20  # Minimum samples before using predictions

    learner: ActiveLearner = field(default_factory=lambda: ActiveLearner(n_estimators=3))
    transitive: TransitiveInference = field(default_factory=TransitiveInference)
    stats: ActiveClassifierStats = field(default_factory=ActiveClassifierStats)
    _training_records: list[TrainingRecord] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock)  # Thread safety for parallel processing

    def load_training_data(self, records: list[dict]) -> None:
        """
        Load training data from database.

        Args:
            records: List of dicts with 'features' and 'label' keys
        """
        self._training_records = [
            TrainingRecord(features=r["features"], label=r["label"]) for r in records
        ]

        if len(self._training_records) >= self.min_training_samples:
            X = np.array([r.features for r in self._training_records])
            y = np.array([r.label for r in self._training_records])
            self.learner.fit(X, y)
            logger.info(f"ActiveClassifier loaded {len(X)} training samples")

        self.stats.training_samples = len(self._training_records)

    def export_training_data(self) -> dict:
        """
        Export training data to JSON-serializable dict.

        Returns:
            Dict with version, timestamp, sample count, and training data
        """
        return {
            "version": 1,
            "exported_at": datetime.utcnow().isoformat(),
            "samples": len(self._training_records),
            "data": [
                {"features": r.features, "label": r.label}
                for r in self._training_records
            ],
        }

    def import_training_data(self, data: dict) -> int:
        """
        Import training data from dict (e.g., loaded from JSON file).

        Args:
            data: Dict with 'version', 'data' keys

        Returns:
            Number of samples loaded

        Raises:
            ValueError: If version is unsupported
        """
        version = data.get("version", 0)
        if version != 1:
            raise ValueError(f"Unsupported training data version: {version}")

        records = data.get("data", [])
        self.load_training_data(records)

        logger.info(
            f"Imported {len(records)} training samples "
            f"(exported at {data.get('exported_at', 'unknown')})"
        )
        return len(records)

    def get_new_training_records(self) -> list[TrainingRecord]:
        """Get training records added this session (for saving to DB)."""
        # Return records added after initial load
        initial_count = self.stats.training_samples - len(self._training_records)
        return self._training_records[initial_count:] if initial_count < 0 else []

    def decide(
        self,
        item_a: dict,
        item_b: dict,
        features: PairFeatures,
    ) -> tuple[str, float, bool]:
        """
        Decide how to classify this pair.

        Returns:
            Tuple of (decision, confidence, needs_llm)
            decision: "match", "no_match", or "uncertain"
            confidence: 0.0-1.0
            needs_llm: whether LLM call is needed
        """
        # Step 1: Check transitive inference
        item_a_id = str(item_a.get("id", id(item_a)))
        item_b_id = str(item_b.get("id", id(item_b)))
        transitive_result = self.transitive.infer(item_a_id, item_b_id)

        if transitive_result is not None:
            is_match, conf = transitive_result
            return ("match" if is_match else "no_match", conf, False)

        # Step 2: Check learner prediction
        if not self.learner.is_fitted:
            return ("uncertain", 0.5, True)

        X = features.to_vector().reshape(1, -1)
        proba = self.learner.predict_proba(X)[0][1]  # P(match)

        if proba >= self.high_confidence:
            return ("match", float(proba), False)
        elif proba <= self.low_confidence:
            return ("no_match", float(1 - proba), False)
        else:
            return ("uncertain", float(proba), True)

    def record_decision(
        self,
        item_a: dict,
        item_b: dict,
        features: PairFeatures,
        is_match: bool,
        llm_confidence: float,
    ) -> bool:
        """
        Record LLM decision for future training.

        Thread-safe: uses lock for shared state modification.

        Returns:
            True if added to training data, False if filtered by quality gate
        """
        # Quality gate (read-only, no lock needed)
        if not should_add_to_training(features, llm_confidence):
            return False

        # Lock for all mutable operations
        with self._lock:
            # Add to training records
            record = TrainingRecord(
                features=features.to_list(),
                label=1 if is_match else 0,
            )
            self._training_records.append(record)

            # Add to transitive graph
            item_a_id = str(item_a.get("id", id(item_a)))
            item_b_id = str(item_b.get("id", id(item_b)))
            self.transitive.add_edge(item_a_id, item_b_id, is_match, llm_confidence)

            # Retrain if we have enough data
            if len(self._training_records) >= self.min_training_samples:
                X = np.array([r.features for r in self._training_records])
                y = np.array([r.label for r in self._training_records])
                self.learner.fit(X, y)

            self.stats.training_samples = len(self._training_records)
        return True

    def classify(
        self,
        new_item: dict,
        existing_groups: list[dict],
        llm_classify_fn,  # Function to call LLM: (item, groups) -> result
    ) -> ClassificationResult:
        """
        Classify a new item into an existing group or mark as NEW.

        Uses active learning to minimize LLM calls.

        Args:
            new_item: Item to classify
            existing_groups: List of existing groups to compare against
            llm_classify_fn: Function to call LLM when uncertain

        Returns:
            ClassificationResult with group_id, is_new, confidence, method
        """
        if not existing_groups:
            return ClassificationResult(
                group_id=None,
                is_new=True,
                confidence=1.0,
                method="prediction",
            )

        # Pre-compute embedding for new item
        from core.features import compute_embedding

        new_embedding = compute_embedding(new_item.get("description", ""))

        # Compare against each existing group
        best_match = None
        best_confidence = 0.0
        best_method = "prediction"
        uncertain_pairs: list[tuple[dict, PairFeatures, float]] = []

        for group in existing_groups:
            group_embedding = compute_embedding(group.get("description", ""))
            features = extract_features(new_item, group, new_embedding, group_embedding)

            decision, confidence, needs_llm = self.decide(new_item, group, features)

            if decision == "match" and confidence > best_confidence:
                best_match = group
                best_confidence = confidence
                best_method = "transitive" if not needs_llm else "prediction"

            if needs_llm and confidence > 0.3:
                uncertain_pairs.append((group, features, confidence))

        # If we have a high-confidence match, use it
        if best_match and best_confidence >= self.high_confidence:
            if best_method == "prediction":
                self.stats.predictions += 1
            else:
                self.stats.transitive_inferences += 1

            return ClassificationResult(
                group_id=str(best_match.get("id")),
                is_new=False,
                confidence=best_confidence,
                method=best_method,
            )

        # Need to query LLM for uncertain cases
        if uncertain_pairs:
            # Sort by uncertainty (middle values most uncertain)
            uncertain_pairs.sort(key=lambda x: abs(x[2] - 0.5))

            # Query LLM for top uncertain cases (limit queries)
            for group, features, _ in uncertain_pairs[:3]:
                llm_result = llm_classify_fn(new_item, [group])
                self.stats.llm_calls += 1

                is_match = not llm_result.get("is_new", True)
                llm_confidence = llm_result.get("confidence", 0.5)

                # Record for training
                self.record_decision(new_item, group, features, is_match, llm_confidence)

                if is_match:
                    return ClassificationResult(
                        group_id=str(group.get("id")),
                        is_new=False,
                        confidence=llm_confidence,
                        method="llm",
                    )

        # No match found
        return ClassificationResult(
            group_id=None,
            is_new=True,
            confidence=1.0 - best_confidence if best_confidence > 0 else 1.0,
            method="prediction",
        )

    def get_stats(self) -> dict:
        """Get classification statistics."""
        return self.stats.to_dict()
