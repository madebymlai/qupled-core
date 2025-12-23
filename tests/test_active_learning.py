"""
Tests for active learning classification system.
"""

import numpy as np


class TestPairFeatures:
    """Test feature extraction."""

    def test_extract_features_similar_items(self):
        from core.features import extract_features

        item_a = {
            "name": "velocity_calculation",
            "description": "Calculate velocity using kinematic equations",
            "category": "Kinematics",
        }
        item_b = {
            "name": "speed_calculation",
            "description": "Compute speed using kinematic formulas",
            "category": "Kinematics",
        }

        features = extract_features(item_a, item_b)

        # Embedding similarity should be high (synonyms: velocity≈speed, calculate≈compute)
        assert features.embedding_similarity > 0.7
        # Same category
        assert features.same_category is True
        # Verb match (Calculate vs Compute - different)
        assert features.verb_match is False

    def test_extract_features_different_items(self):
        from core.features import extract_features

        item_a = {
            "name": "velocity_calculation",
            "description": "Calculate velocity using kinematic equations",
            "category": "Kinematics",
        }
        item_b = {
            "name": "thermodynamics_laws",
            "description": "Explain the laws of thermodynamics",
            "category": "Thermodynamics",
        }

        features = extract_features(item_a, item_b)

        # Embedding similarity should be low (completely different topics)
        assert features.embedding_similarity < 0.5
        # Different category
        assert features.same_category is False

    def test_to_vector(self):
        from core.features import PairFeatures

        features = PairFeatures(
            embedding_similarity=0.8,
            token_jaccard=0.5,
            trigram_jaccard=0.6,
            desc_length_ratio=0.9,
            same_category=True,
            verb_match=True,
            name_similarity=0.7,
        )

        vector = features.to_vector()

        assert len(vector) == 7
        assert vector[0] == 0.8  # embedding_similarity
        assert vector[4] == 1.0  # same_category (True -> 1.0)
        assert vector[5] == 1.0  # verb_match (True -> 1.0)


class TestTransitiveInference:
    """Test transitive inference engine."""

    def test_direct_edge(self):
        from core.transitive import TransitiveInference

        ti = TransitiveInference()
        ti.add_edge("a", "b", True, 0.95)

        result = ti.infer("a", "b")

        assert result is not None
        assert result[0] is True  # is_match
        assert result[1] == 0.95  # confidence

    def test_transitive_inference(self):
        from core.transitive import TransitiveInference

        ti = TransitiveInference()
        ti.add_edge("a", "b", True, 0.95)
        ti.add_edge("b", "c", True, 0.90)

        result = ti.infer("a", "c")

        assert result is not None
        assert result[0] is True  # is_match
        assert result[1] >= 0.85  # 0.95 * 0.90 = 0.855

    def test_no_inference_low_confidence(self):
        from core.transitive import TransitiveInference

        ti = TransitiveInference(min_confidence=0.75)
        ti.add_edge("a", "b", True, 0.5)  # Low confidence
        ti.add_edge("b", "c", True, 0.5)  # Low confidence

        result = ti.infer("a", "c")

        # Should not infer because 0.5 * 0.5 = 0.25 < 0.75
        assert result is None

    def test_get_component(self):
        from core.transitive import TransitiveInference

        ti = TransitiveInference()
        ti.add_edge("a", "b", True, 0.95)
        ti.add_edge("b", "c", True, 0.95)
        ti.add_edge("d", "e", True, 0.95)  # Separate component

        component_a = ti.get_component("a")

        assert "a" in component_a
        assert "b" in component_a
        assert "c" in component_a
        assert "d" not in component_a  # Different component


class TestActiveLearner:
    """Test the active learner (QBC)."""

    def test_fit_and_predict(self):
        from core.active_learning import ActiveLearner

        learner = ActiveLearner(n_estimators=3)

        # Create simple training data
        X = np.array([
            [0.9, 0.8, 0.7, 0.9, 1.0, 1.0, 0.8],  # Match
            [0.85, 0.75, 0.65, 0.85, 1.0, 1.0, 0.75],  # Match
            [0.2, 0.1, 0.1, 0.5, 0.0, 0.0, 0.2],  # No match
            [0.15, 0.05, 0.05, 0.4, 0.0, 0.0, 0.15],  # No match
        ])
        y = np.array([1, 1, 0, 0])

        learner.fit(X, y)

        assert learner.is_fitted is True

        # Test prediction on similar item
        test_match = np.array([[0.88, 0.78, 0.68, 0.88, 1.0, 1.0, 0.78]])
        proba = learner.predict_proba(test_match)

        assert proba[0][1] > 0.5  # Should predict match

    def test_uncertainty(self):
        from core.active_learning import ActiveLearner

        learner = ActiveLearner(n_estimators=3)

        # Without fitting, everything should be uncertain
        X = np.array([[0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5]])
        uncertainty = learner.uncertainty(X)

        assert uncertainty[0] == 1.0  # Max uncertainty when not fitted


class TestQualityGates:
    """Test training data quality gates."""

    def test_high_confidence_passes(self):
        from core.features import PairFeatures, should_add_to_training

        features = PairFeatures(
            embedding_similarity=0.8,
            token_jaccard=0.5,
            trigram_jaccard=0.6,
            desc_length_ratio=0.9,
            same_category=True,
            verb_match=True,
            name_similarity=0.7,
        )

        # High confidence LLM decision should pass
        assert should_add_to_training(features, 0.95) is True

    def test_uncertain_fails(self):
        from core.features import PairFeatures, should_add_to_training

        features = PairFeatures(
            embedding_similarity=0.5,
            token_jaccard=0.3,
            trigram_jaccard=0.4,
            desc_length_ratio=0.8,
            same_category=True,
            verb_match=False,
            name_similarity=0.5,
        )

        # Uncertain LLM decision should be filtered
        assert should_add_to_training(features, 0.5) is False

    def test_suspicious_match_fails(self):
        from core.features import PairFeatures, should_add_to_training

        features = PairFeatures(
            embedding_similarity=0.1,  # Very low similarity
            token_jaccard=0.0,
            trigram_jaccard=0.1,
            desc_length_ratio=0.9,
            same_category=False,
            verb_match=False,
            name_similarity=0.1,
        )

        # LLM says "match" but features say "no way" - suspicious
        assert should_add_to_training(features, 0.95) is False
