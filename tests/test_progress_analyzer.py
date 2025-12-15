"""
Unit tests for ProgressAnalyzer.

Tests cover all static methods for mastery calculation, gap detection,
and learning path generation without requiring database access.
"""

import sys
import os
from datetime import datetime, timezone, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.progress_analyzer import ProgressAnalyzer
from core.dto.mastery import (
    ExerciseReviewData,
    GapSeverity,
    MasteryLevel,
    MasteryTrend,
    TopicMasteryInput,
    TopicMasteryResult,
)
from core.dto.progress import KnowledgeGap


# ============================================================================
# Test calculate_mastery_score()
# ============================================================================


def test_calculate_mastery_score_empty():
    """Test mastery score with no reviews returns 0."""
    score = ProgressAnalyzer.calculate_mastery_score([])
    assert score == 0.0, f"Expected 0.0 for empty reviews, got {score}"
    print("✓ test_calculate_mastery_score_empty passed")


def test_calculate_mastery_score_perfect():
    """Test mastery score with perfect reviews."""
    now = datetime.now(timezone.utc)
    reviews = [
        ExerciseReviewData(
            exercise_id="1",
            mastery_level=MasteryLevel.MASTERED,
            interval_days=90,
            total_reviews=10,
            correct_reviews=10,
            last_reviewed_at=now - timedelta(days=1),
        ),
        ExerciseReviewData(
            exercise_id="2",
            mastery_level=MasteryLevel.MASTERED,
            interval_days=90,
            total_reviews=10,
            correct_reviews=10,
            last_reviewed_at=now - timedelta(days=2),
        ),
        ExerciseReviewData(
            exercise_id="3",
            mastery_level=MasteryLevel.MASTERED,
            interval_days=90,
            total_reviews=10,
            correct_reviews=10,
            last_reviewed_at=now - timedelta(days=3),
        ),
    ]
    score = ProgressAnalyzer.calculate_mastery_score(reviews, now)

    # 40% accuracy (1.0) + 30% mastery (1.0) + 20% interval (1.0) + 5% recency = 0.95
    assert score >= 0.9, f"Expected >= 0.9 for perfect reviews, got {score}"
    print("✓ test_calculate_mastery_score_perfect passed")


def test_calculate_mastery_score_learning():
    """Test mastery score for exercises in learning state."""
    reviews = [
        ExerciseReviewData(
            exercise_id="1",
            mastery_level=MasteryLevel.LEARNING,
            interval_days=1,
            total_reviews=5,
            correct_reviews=2,
            last_reviewed_at=None,
        ),
    ]
    score = ProgressAnalyzer.calculate_mastery_score(reviews)

    # 40% * 0.4 + 30% * 0.3 + 20% * 0.011 + 0 recency = ~0.252
    assert 0.2 <= score <= 0.35, f"Expected 0.2-0.35 for learning state, got {score}"
    print("✓ test_calculate_mastery_score_learning passed")


def test_calculate_mastery_score_mixed():
    """Test mastery score with mixed mastery levels."""
    now = datetime.now(timezone.utc)
    reviews = [
        ExerciseReviewData(
            exercise_id="1",
            mastery_level=MasteryLevel.MASTERED,
            interval_days=60,
            total_reviews=10,
            correct_reviews=9,
            last_reviewed_at=now - timedelta(days=5),
        ),
        ExerciseReviewData(
            exercise_id="2",
            mastery_level=MasteryLevel.REVIEWING,
            interval_days=14,
            total_reviews=5,
            correct_reviews=4,
            last_reviewed_at=now - timedelta(days=3),
        ),
        ExerciseReviewData(
            exercise_id="3",
            mastery_level=MasteryLevel.LEARNING,
            interval_days=1,
            total_reviews=3,
            correct_reviews=1,
            last_reviewed_at=now - timedelta(days=6),
        ),
    ]
    score = ProgressAnalyzer.calculate_mastery_score(reviews, now)

    # Should be moderate score
    assert 0.4 <= score <= 0.7, f"Expected 0.4-0.7 for mixed reviews, got {score}"
    print("✓ test_calculate_mastery_score_mixed passed")


# ============================================================================
# Test determine_trend()
# ============================================================================


def test_determine_trend_empty():
    """Test trend with no reviews returns NEW."""
    trend = ProgressAnalyzer.determine_trend([])
    assert trend == MasteryTrend.NEW, f"Expected NEW for empty reviews, got {trend}"
    print("✓ test_determine_trend_empty passed")


def test_determine_trend_improving():
    """Test trend when >50% mastered."""
    reviews = [
        ExerciseReviewData(
            exercise_id="1",
            mastery_level=MasteryLevel.MASTERED,
            interval_days=30,
            total_reviews=10,
            correct_reviews=10,
        ),
        ExerciseReviewData(
            exercise_id="2",
            mastery_level=MasteryLevel.MASTERED,
            interval_days=30,
            total_reviews=10,
            correct_reviews=10,
        ),
        ExerciseReviewData(
            exercise_id="3",
            mastery_level=MasteryLevel.REVIEWING,
            interval_days=7,
            total_reviews=5,
            correct_reviews=4,
        ),
    ]
    trend = ProgressAnalyzer.determine_trend(reviews)
    assert trend == MasteryTrend.IMPROVING, f"Expected IMPROVING, got {trend}"
    print("✓ test_determine_trend_improving passed")


def test_determine_trend_new():
    """Test trend when >50% learning."""
    reviews = [
        ExerciseReviewData(
            exercise_id="1",
            mastery_level=MasteryLevel.LEARNING,
            interval_days=1,
            total_reviews=2,
            correct_reviews=1,
        ),
        ExerciseReviewData(
            exercise_id="2",
            mastery_level=MasteryLevel.LEARNING,
            interval_days=1,
            total_reviews=2,
            correct_reviews=1,
        ),
        ExerciseReviewData(
            exercise_id="3",
            mastery_level=MasteryLevel.REVIEWING,
            interval_days=7,
            total_reviews=5,
            correct_reviews=4,
        ),
    ]
    trend = ProgressAnalyzer.determine_trend(reviews)
    assert trend == MasteryTrend.NEW, f"Expected NEW, got {trend}"
    print("✓ test_determine_trend_new passed")


def test_determine_trend_stable():
    """Test trend when evenly distributed."""
    reviews = [
        ExerciseReviewData(
            exercise_id="1",
            mastery_level=MasteryLevel.MASTERED,
            interval_days=30,
            total_reviews=10,
            correct_reviews=10,
        ),
        ExerciseReviewData(
            exercise_id="2",
            mastery_level=MasteryLevel.REVIEWING,
            interval_days=14,
            total_reviews=5,
            correct_reviews=4,
        ),
        ExerciseReviewData(
            exercise_id="3",
            mastery_level=MasteryLevel.REVIEWING,
            interval_days=7,
            total_reviews=5,
            correct_reviews=4,
        ),
        ExerciseReviewData(
            exercise_id="4",
            mastery_level=MasteryLevel.LEARNING,
            interval_days=1,
            total_reviews=2,
            correct_reviews=1,
        ),
    ]
    trend = ProgressAnalyzer.determine_trend(reviews)
    assert trend == MasteryTrend.STABLE, f"Expected STABLE, got {trend}"
    print("✓ test_determine_trend_stable passed")


# ============================================================================
# Test classify_gap_severity()
# ============================================================================


def test_classify_gap_severity_critical():
    """Test critical gap severity for low mastery."""
    severity = ProgressAnalyzer.classify_gap_severity(0.2)
    assert severity == GapSeverity.CRITICAL, f"Expected CRITICAL for 0.2, got {severity}"
    print("✓ test_classify_gap_severity_critical passed")


def test_classify_gap_severity_significant():
    """Test significant gap severity for medium-low mastery."""
    severity = ProgressAnalyzer.classify_gap_severity(0.4)
    assert severity == GapSeverity.SIGNIFICANT, f"Expected SIGNIFICANT for 0.4, got {severity}"
    print("✓ test_classify_gap_severity_significant passed")


def test_classify_gap_severity_minor():
    """Test minor gap severity for medium mastery."""
    severity = ProgressAnalyzer.classify_gap_severity(0.55)
    assert severity == GapSeverity.MINOR, f"Expected MINOR for 0.55, got {severity}"
    print("✓ test_classify_gap_severity_minor passed")


def test_classify_gap_severity_boundary():
    """Test boundary values."""
    assert ProgressAnalyzer.classify_gap_severity(0.29) == GapSeverity.CRITICAL
    assert ProgressAnalyzer.classify_gap_severity(0.30) == GapSeverity.SIGNIFICANT
    assert ProgressAnalyzer.classify_gap_severity(0.49) == GapSeverity.SIGNIFICANT
    assert ProgressAnalyzer.classify_gap_severity(0.50) == GapSeverity.MINOR
    print("✓ test_classify_gap_severity_boundary passed")


# ============================================================================
# Test calculate_recommendations()
# ============================================================================


def test_calculate_recommendations_critical():
    """Test recommendations for critical mastery."""
    exercises, time = ProgressAnalyzer.calculate_recommendations(0.2)
    assert exercises == 20, f"Expected 20 exercises for critical, got {exercises}"
    assert time == 60, f"Expected 60 minutes for critical, got {time}"
    print("✓ test_calculate_recommendations_critical passed")


def test_calculate_recommendations_significant():
    """Test recommendations for significant mastery."""
    exercises, time = ProgressAnalyzer.calculate_recommendations(0.4)
    assert exercises == 15, f"Expected 15 exercises for significant, got {exercises}"
    assert time == 45, f"Expected 45 minutes for significant, got {time}"
    print("✓ test_calculate_recommendations_significant passed")


def test_calculate_recommendations_minor():
    """Test recommendations for minor mastery."""
    exercises, time = ProgressAnalyzer.calculate_recommendations(0.55)
    assert exercises == 10, f"Expected 10 exercises for minor, got {exercises}"
    assert time == 30, f"Expected 30 minutes for minor, got {time}"
    print("✓ test_calculate_recommendations_minor passed")


# ============================================================================
# Test detect_common_mistakes()
# ============================================================================


def test_detect_common_mistakes_no_reviews():
    """Test common mistakes with no reviews."""
    mistakes = ProgressAnalyzer.detect_common_mistakes([], "Test Topic")
    assert "No practice attempts yet" in mistakes, f"Expected no attempts message, got {mistakes}"
    print("✓ test_detect_common_mistakes_no_reviews passed")


def test_detect_common_mistakes_low_accuracy():
    """Test common mistakes with low accuracy."""
    reviews = [
        ExerciseReviewData(
            exercise_id="1",
            mastery_level=MasteryLevel.LEARNING,
            interval_days=1,
            total_reviews=10,
            correct_reviews=3,  # 30% accuracy
        ),
    ]
    mistakes = ProgressAnalyzer.detect_common_mistakes(reviews, "Math")
    assert any("Low accuracy" in m for m in mistakes), (
        f"Expected low accuracy message, got {mistakes}"
    )
    print("✓ test_detect_common_mistakes_low_accuracy passed")


def test_detect_common_mistakes_stuck_learning():
    """Test common mistakes when stuck in learning mode."""
    reviews = [
        ExerciseReviewData(
            exercise_id=str(i),
            mastery_level=MasteryLevel.LEARNING,
            interval_days=1,
            total_reviews=5,
            correct_reviews=4,
        )
        for i in range(10)  # 100% in learning
    ]
    mistakes = ProgressAnalyzer.detect_common_mistakes(reviews, "Physics")
    assert any("Difficulty progressing" in m for m in mistakes), (
        f"Expected stuck message, got {mistakes}"
    )
    print("✓ test_detect_common_mistakes_stuck_learning passed")


# ============================================================================
# Test calculate_topic_mastery()
# ============================================================================


def test_calculate_topic_mastery():
    """Test complete topic mastery calculation."""
    now = datetime.now(timezone.utc)
    topic_input = TopicMasteryInput(
        topic_id="topic-1",
        topic_name="Calculus",
        exercise_reviews=[
            ExerciseReviewData(
                exercise_id="1",
                mastery_level=MasteryLevel.MASTERED,
                interval_days=30,
                total_reviews=10,
                correct_reviews=9,
                last_reviewed_at=now - timedelta(days=2),
            ),
            ExerciseReviewData(
                exercise_id="2",
                mastery_level=MasteryLevel.REVIEWING,
                interval_days=7,
                total_reviews=5,
                correct_reviews=4,
                last_reviewed_at=now - timedelta(days=5),
            ),
        ],
        total_exercises=5,
    )

    result = ProgressAnalyzer.calculate_topic_mastery(topic_input)

    assert result.topic_id == "topic-1"
    assert result.topic_name == "Calculus"
    assert result.exercises_total == 5
    assert result.exercises_reviewed == 2
    assert result.correct_count == 13
    assert result.mastery_score > 0
    assert result.last_practiced_at is not None
    print("✓ test_calculate_topic_mastery passed")


# ============================================================================
# Test build_knowledge_gap()
# ============================================================================


def test_build_knowledge_gap():
    """Test building a complete knowledge gap."""
    reviews = [
        ExerciseReviewData(
            exercise_id="1",
            mastery_level=MasteryLevel.LEARNING,
            interval_days=1,
            total_reviews=5,
            correct_reviews=2,
        ),
    ]

    gap = ProgressAnalyzer.build_knowledge_gap(
        topic_id="topic-1",
        topic_name="Algebra",
        mastery=0.25,
        reviews=reviews,
        knowledge_item_names=["Linear Equations", "Quadratics"],
    )

    assert gap.topic_id == "topic-1"
    assert gap.topic_name == "Algebra"
    assert gap.gap_severity == GapSeverity.CRITICAL
    assert gap.current_mastery == 25.0  # Converted to percentage
    assert gap.target_mastery == 80.0
    assert gap.exercises_to_review == 20  # Critical = 20
    assert gap.estimated_time_minutes == 60  # Critical = 60
    assert len(gap.recommended_actions) > 0
    assert any("Linear Equations" in action for action in gap.recommended_actions)
    print("✓ test_build_knowledge_gap passed")


# ============================================================================
# Test prioritize_learning_items()
# ============================================================================


def test_prioritize_learning_items():
    """Test learning path prioritization."""
    topic_results = [
        TopicMasteryResult(
            topic_id="t1",
            topic_name="Topic 1",
            mastery_score=0.25,  # Critical
            mastery_trend=MasteryTrend.NEW,
            exercises_total=10,
            exercises_reviewed=2,
            correct_count=1,
            accuracy_percentage=50.0,
        ),
        TopicMasteryResult(
            topic_id="t2",
            topic_name="Topic 2",
            mastery_score=0.7,  # In progress
            mastery_trend=MasteryTrend.STABLE,
            exercises_total=10,
            exercises_reviewed=7,
            correct_count=6,
            accuracy_percentage=85.7,
        ),
        TopicMasteryResult(
            topic_id="t3",
            topic_name="Topic 3",
            mastery_score=0.85,  # Mastered
            mastery_trend=MasteryTrend.IMPROVING,
            exercises_total=10,
            exercises_reviewed=10,
            correct_count=9,
            accuracy_percentage=90.0,
        ),
    ]

    gaps = [
        KnowledgeGap(
            topic_id="t1",
            topic_name="Topic 1",
            gap_severity=GapSeverity.CRITICAL,
            current_mastery=25.0,
            target_mastery=80.0,
            common_mistakes=[],
            recommended_actions=[],
            exercises_to_review=20,
            estimated_time_minutes=60,
        ),
    ]

    items = ProgressAnalyzer.prioritize_learning_items(topic_results, gaps, max_items=5)

    # Should have critical gap first
    assert len(items) >= 1
    assert items[0].item_id == "t1"
    assert items[0].priority == "high"

    # Order should be correct
    for i, item in enumerate(items):
        assert item.order == i + 1

    print("✓ test_prioritize_learning_items passed")


def test_prioritize_learning_items_empty():
    """Test learning path with no data."""
    items = ProgressAnalyzer.prioritize_learning_items([], [], max_items=5)
    assert items == [], f"Expected empty list, got {items}"
    print("✓ test_prioritize_learning_items_empty passed")


# ============================================================================
# Test generate_recommended_actions()
# ============================================================================


def test_generate_recommended_actions():
    """Test generating recommended actions."""
    actions = ProgressAnalyzer.generate_recommended_actions(
        topic_name="Physics",
        mastery=0.2,
        knowledge_item_names=["Kinematics", "Dynamics"],
    )

    assert len(actions) > 0
    assert any("foundations" in a.lower() for a in actions)
    assert any("Physics" in a for a in actions)
    assert any("20" in a for a in actions)  # Critical = 20 exercises
    assert any("Kinematics" in a for a in actions)
    print("✓ test_generate_recommended_actions passed")


# ============================================================================
# Main runner
# ============================================================================


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running ProgressAnalyzer tests")
    print("=" * 60 + "\n")

    # calculate_mastery_score tests
    test_calculate_mastery_score_empty()
    test_calculate_mastery_score_perfect()
    test_calculate_mastery_score_learning()
    test_calculate_mastery_score_mixed()

    # determine_trend tests
    test_determine_trend_empty()
    test_determine_trend_improving()
    test_determine_trend_new()
    test_determine_trend_stable()

    # classify_gap_severity tests
    test_classify_gap_severity_critical()
    test_classify_gap_severity_significant()
    test_classify_gap_severity_minor()
    test_classify_gap_severity_boundary()

    # calculate_recommendations tests
    test_calculate_recommendations_critical()
    test_calculate_recommendations_significant()
    test_calculate_recommendations_minor()

    # detect_common_mistakes tests
    test_detect_common_mistakes_no_reviews()
    test_detect_common_mistakes_low_accuracy()
    test_detect_common_mistakes_stuck_learning()

    # calculate_topic_mastery tests
    test_calculate_topic_mastery()

    # build_knowledge_gap tests
    test_build_knowledge_gap()

    # prioritize_learning_items tests
    test_prioritize_learning_items()
    test_prioritize_learning_items_empty()

    # generate_recommended_actions tests
    test_generate_recommended_actions()

    print("\n" + "=" * 60)
    print("All ProgressAnalyzer tests passed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all_tests()
