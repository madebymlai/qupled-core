"""
Unit tests for SM-2 spaced repetition algorithm.

Tests cover:
- Core SM-2 calculation logic
- Quality score conversion
- Mastery level determination
- Edge cases and boundary conditions
"""

import unittest
from datetime import datetime, timedelta
from core.sm2 import SM2Algorithm


class TestSM2Algorithm(unittest.TestCase):
    """Test cases for SM-2 spaced repetition algorithm."""

    def setUp(self):
        """Set up test fixtures."""
        self.sm2 = SM2Algorithm()

    def test_first_review_quality_5(self):
        """Test first review with perfect recall (quality 5)."""
        result = self.sm2.calculate_next_review(
            quality=5, current_ef=2.5, current_interval=0, repetition=0
        )

        self.assertEqual(result["new_ef"], 2.5)
        self.assertEqual(result["new_interval"], 1)
        self.assertEqual(result["new_repetition"], 1)
        self.assertIsInstance(result["next_review_date"], datetime)

    def test_second_review_quality_5(self):
        """Test second review with perfect recall."""
        result = self.sm2.calculate_next_review(
            quality=5, current_ef=2.5, current_interval=1, repetition=1
        )

        self.assertEqual(result["new_ef"], 2.5)
        self.assertEqual(result["new_interval"], 6)
        self.assertEqual(result["new_repetition"], 2)

    def test_third_review_quality_5(self):
        """Test third review with perfect recall (exponential growth starts)."""
        result = self.sm2.calculate_next_review(
            quality=5, current_ef=2.5, current_interval=6, repetition=2
        )

        self.assertEqual(result["new_ef"], 2.5)
        self.assertEqual(result["new_interval"], 15)  # 6 * 2.5 = 15
        self.assertEqual(result["new_repetition"], 3)

    def test_fourth_review_quality_5(self):
        """Test fourth review with perfect recall."""
        result = self.sm2.calculate_next_review(
            quality=5, current_ef=2.5, current_interval=15, repetition=3
        )

        self.assertEqual(result["new_ef"], 2.5)
        self.assertEqual(result["new_interval"], 38)  # 15 * 2.5 = 37.5 -> 38
        self.assertEqual(result["new_repetition"], 4)

    def test_quality_4_reduces_ef_slightly(self):
        """Test that quality 4 keeps EF stable."""
        result = self.sm2.calculate_next_review(
            quality=4, current_ef=2.5, current_interval=0, repetition=0
        )

        # Quality 4: EF' = EF + (0.1 - 1 * (0.08 + 1 * 0.02)) = EF + (0.1 - 0.1) = EF
        self.assertEqual(result["new_ef"], 2.5)
        self.assertEqual(result["new_interval"], 1)
        self.assertEqual(result["new_repetition"], 1)

    def test_quality_3_reduces_ef(self):
        """Test that quality 3 reduces EF."""
        result = self.sm2.calculate_next_review(
            quality=3, current_ef=2.5, current_interval=0, repetition=0
        )

        # Quality 3: EF' = EF + (0.1 - 2 * (0.08 + 2 * 0.02))
        # = 2.5 + (0.1 - 2 * 0.12) = 2.5 + (0.1 - 0.24) = 2.5 - 0.14 = 2.36
        self.assertEqual(result["new_ef"], 2.36)
        self.assertEqual(result["new_interval"], 1)
        self.assertEqual(result["new_repetition"], 1)

    def test_quality_2_resets_repetition(self):
        """Test that quality < 3 resets repetition counter."""
        result = self.sm2.calculate_next_review(
            quality=2, current_ef=2.5, current_interval=15, repetition=3
        )

        # Poor recall resets progress
        self.assertEqual(result["new_interval"], 1)
        self.assertEqual(result["new_repetition"], 0)
        # EF still decreases
        self.assertLess(result["new_ef"], 2.5)

    def test_quality_0_resets_repetition(self):
        """Test that quality 0 (complete blackout) resets repetition."""
        result = self.sm2.calculate_next_review(
            quality=0, current_ef=2.5, current_interval=30, repetition=5
        )

        self.assertEqual(result["new_interval"], 1)
        self.assertEqual(result["new_repetition"], 0)
        # EF significantly decreases
        # Quality 0: EF' = EF + (0.1 - 5 * (0.08 + 5 * 0.02))
        # = 2.5 + (0.1 - 5 * 0.18) = 2.5 + (0.1 - 0.9) = 2.5 - 0.8 = 1.7
        self.assertEqual(result["new_ef"], 1.7)

    def test_ef_clamped_at_minimum(self):
        """Test that EF is clamped at 1.3 minimum."""
        result = self.sm2.calculate_next_review(
            quality=0, current_ef=1.3, current_interval=1, repetition=0
        )

        # Even with quality 0, EF shouldn't go below 1.3
        self.assertGreaterEqual(result["new_ef"], 1.3)

    def test_ef_clamped_at_maximum(self):
        """Test that EF is clamped at 2.5 maximum."""
        result = self.sm2.calculate_next_review(
            quality=5, current_ef=2.5, current_interval=1, repetition=1
        )

        # EF shouldn't exceed 2.5
        self.assertLessEqual(result["new_ef"], 2.5)

    def test_quality_progression_mixed(self):
        """Test a realistic progression with mixed quality scores."""
        # Simulate a learning sequence: 5, 4, 3, 5, 4, 5
        ef = 2.5
        interval = 0
        rep = 0

        # Review 1: quality 5
        result = self.sm2.calculate_next_review(5, ef, interval, rep)
        self.assertEqual(result["new_interval"], 1)
        self.assertEqual(result["new_repetition"], 1)
        ef, interval, rep = result["new_ef"], result["new_interval"], result["new_repetition"]

        # Review 2: quality 4
        result = self.sm2.calculate_next_review(4, ef, interval, rep)
        self.assertEqual(result["new_interval"], 6)
        self.assertEqual(result["new_repetition"], 2)
        ef, interval, rep = result["new_ef"], result["new_interval"], result["new_repetition"]

        # Review 3: quality 3 (struggled)
        result = self.sm2.calculate_next_review(3, ef, interval, rep)
        self.assertGreater(result["new_interval"], 6)  # Still progresses
        self.assertEqual(result["new_repetition"], 3)
        self.assertLess(result["new_ef"], 2.5)  # EF decreases
        ef, interval, rep = result["new_ef"], result["new_interval"], result["new_repetition"]

        # Review 4: quality 5 (recovered)
        result = self.sm2.calculate_next_review(5, ef, interval, rep)
        self.assertGreater(result["new_interval"], interval)
        self.assertEqual(result["new_repetition"], 4)

    def test_get_review_quality_perfect_performance(self):
        """Test quality conversion for perfect performance."""
        quality = self.sm2.get_review_quality_from_score(
            correct=True, time_taken=60, hint_used=False, expected_time=180
        )
        self.assertEqual(quality, 5)

    def test_get_review_quality_good_performance(self):
        """Test quality conversion for good performance."""
        quality = self.sm2.get_review_quality_from_score(
            correct=True, time_taken=150, hint_used=False, expected_time=180
        )
        self.assertEqual(quality, 4)

    def test_get_review_quality_with_hint(self):
        """Test quality conversion when hint was used."""
        quality = self.sm2.get_review_quality_from_score(
            correct=True, time_taken=150, hint_used=True, expected_time=180
        )
        self.assertEqual(quality, 4)

    def test_get_review_quality_slow_but_correct(self):
        """Test quality conversion for slow but correct answer."""
        quality = self.sm2.get_review_quality_from_score(
            correct=True, time_taken=300, hint_used=False, expected_time=180
        )
        self.assertEqual(quality, 3)

    def test_get_review_quality_incorrect(self):
        """Test quality conversion for incorrect answer."""
        quality = self.sm2.get_review_quality_from_score(
            correct=False, time_taken=120, hint_used=False, expected_time=180
        )
        self.assertEqual(quality, 2)

    def test_mastery_level_new(self):
        """Test mastery level for new item."""
        level = self.sm2.get_mastery_level(
            repetition=0, interval_days=0, correct_count=0, total_count=0
        )
        self.assertEqual(level, "new")

    def test_mastery_level_learning(self):
        """Test mastery level for learning stage."""
        level = self.sm2.get_mastery_level(
            repetition=1, interval_days=1, correct_count=1, total_count=1
        )
        self.assertEqual(level, "learning")

        level = self.sm2.get_mastery_level(
            repetition=2, interval_days=6, correct_count=2, total_count=2
        )
        self.assertEqual(level, "learning")

    def test_mastery_level_reviewing(self):
        """Test mastery level for reviewing stage."""
        level = self.sm2.get_mastery_level(
            repetition=5, interval_days=15, correct_count=5, total_count=6
        )
        self.assertEqual(level, "reviewing")

    def test_mastery_level_mastered(self):
        """Test mastery level for mastered stage."""
        level = self.sm2.get_mastery_level(
            repetition=12, interval_days=45, correct_count=12, total_count=13
        )
        self.assertEqual(level, "mastered")

    def test_mastery_level_requires_high_accuracy_for_mastered(self):
        """Test that 'mastered' requires high accuracy."""
        # Many reviews but low accuracy shouldn't be 'mastered'
        level = self.sm2.get_mastery_level(
            repetition=12,
            interval_days=45,
            correct_count=6,  # Only 50% correct
            total_count=12,
        )
        self.assertEqual(level, "reviewing")

    def test_backward_compatibility_calculate_method(self):
        """Test that old calculate() method still works."""
        result = self.sm2.calculate(
            quality=5, easiness_factor=2.5, repetition_number=0, previous_interval=0
        )

        self.assertEqual(result.easiness_factor, 2.5)
        self.assertEqual(result.interval_days, 1)
        self.assertEqual(result.repetition_number, 1)
        self.assertEqual(result.quality, 5)

    def test_backward_compatibility_convert_score_to_quality(self):
        """Test that old convert_score_to_quality() method still works."""
        quality = self.sm2.convert_score_to_quality(score=0.95, hint_used=False)
        self.assertEqual(quality, 5)

    def test_is_due_for_review_yes(self):
        """Test due for review check when item is due."""
        last_review = datetime.now() - timedelta(days=10)
        interval_days = 7

        is_due = self.sm2.is_due_for_review(last_review, interval_days)
        self.assertTrue(is_due)

    def test_is_due_for_review_no(self):
        """Test due for review check when item is not due."""
        last_review = datetime.now() - timedelta(days=3)
        interval_days = 7

        is_due = self.sm2.is_due_for_review(last_review, interval_days)
        self.assertFalse(is_due)

    def test_get_next_review_date(self):
        """Test next review date calculation."""
        last_review = datetime(2025, 1, 1, 12, 0, 0)
        interval_days = 7

        next_review = self.sm2.get_next_review_date(last_review, interval_days)
        expected = datetime(2025, 1, 8, 12, 0, 0)

        self.assertEqual(next_review, expected)

    def test_edge_case_quality_out_of_range_high(self):
        """Test that quality > 5 is clamped to 5."""
        result = self.sm2.calculate_next_review(
            quality=10,  # Invalid, should be clamped to 5
            current_ef=2.5,
            current_interval=0,
            repetition=0,
        )

        # Should behave same as quality 5
        self.assertEqual(result["new_interval"], 1)
        self.assertEqual(result["new_repetition"], 1)

    def test_edge_case_quality_out_of_range_low(self):
        """Test that quality < 0 is clamped to 0."""
        result = self.sm2.calculate_next_review(
            quality=-5,  # Invalid, should be clamped to 0
            current_ef=2.5,
            current_interval=15,
            repetition=3,
        )

        # Should behave same as quality 0
        self.assertEqual(result["new_interval"], 1)
        self.assertEqual(result["new_repetition"], 0)

    def test_edge_case_zero_expected_time(self):
        """Test quality conversion with zero expected time."""
        # Should handle division by zero gracefully
        with self.assertRaises(ZeroDivisionError):
            self.sm2.get_review_quality_from_score(
                correct=True, time_taken=100, hint_used=False, expected_time=0
            )

    def test_realistic_study_session(self):
        """Test a realistic study session over multiple days."""
        # Simulate studying the same item over time
        ef = 2.5
        interval = 0
        rep = 0

        # Day 1: First encounter, perfect recall
        result = self.sm2.calculate_next_review(5, ef, interval, rep)
        self.assertEqual(result["new_interval"], 1)
        ef, interval, rep = result["new_ef"], result["new_interval"], result["new_repetition"]

        # Day 2: Review, good recall
        result = self.sm2.calculate_next_review(4, ef, interval, rep)
        self.assertEqual(result["new_interval"], 6)
        ef, interval, rep = result["new_ef"], result["new_interval"], result["new_repetition"]

        # Day 8: Review, struggled a bit
        result = self.sm2.calculate_next_review(3, ef, interval, rep)
        self.assertGreater(result["new_interval"], 6)
        self.assertLess(result["new_ef"], 2.5)
        ef, interval, rep = result["new_ef"], result["new_interval"], result["new_repetition"]

        # Continue with good performance
        for _ in range(3):
            result = self.sm2.calculate_next_review(5, ef, interval, rep)
            ef, interval, rep = result["new_ef"], result["new_interval"], result["new_repetition"]

        # After multiple reviews, interval should be substantial
        self.assertGreater(interval, 30)
        self.assertGreater(rep, 5)


class TestSM2EdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.sm2 = SM2Algorithm()

    def test_all_quality_levels(self):
        """Test all quality levels 0-5."""
        for quality in range(6):
            result = self.sm2.calculate_next_review(
                quality=quality, current_ef=2.5, current_interval=6, repetition=2
            )

            # Verify result structure
            self.assertIn("new_ef", result)
            self.assertIn("new_interval", result)
            self.assertIn("new_repetition", result)
            self.assertIn("next_review_date", result)

            # Verify EF bounds
            self.assertGreaterEqual(result["new_ef"], 1.3)
            self.assertLessEqual(result["new_ef"], 2.5)

            # Verify interval is positive
            self.assertGreater(result["new_interval"], 0)

    def test_extreme_intervals(self):
        """Test very large intervals."""
        result = self.sm2.calculate_next_review(
            quality=5,
            current_ef=2.5,
            current_interval=365,  # 1 year
            repetition=10,
        )

        # Should continue growing
        self.assertGreater(result["new_interval"], 365)

    def test_mastery_all_levels(self):
        """Test all mastery levels can be reached."""
        levels_seen = set()

        # New
        level = self.sm2.get_mastery_level(0, 0, 0, 0)
        levels_seen.add(level)

        # Learning
        level = self.sm2.get_mastery_level(1, 1, 1, 1)
        levels_seen.add(level)

        # Reviewing
        level = self.sm2.get_mastery_level(5, 15, 5, 5)
        levels_seen.add(level)

        # Mastered
        level = self.sm2.get_mastery_level(12, 45, 12, 12)
        levels_seen.add(level)

        self.assertEqual(levels_seen, {"new", "learning", "reviewing", "mastered"})


if __name__ == "__main__":
    unittest.main()
