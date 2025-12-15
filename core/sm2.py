"""
SM-2 (SuperMemo 2) Spaced Repetition Algorithm

This module implements the SuperMemo 2 algorithm, a proven spaced repetition
system that optimally schedules review times based on recall performance.

The SM-2 algorithm was developed by Piotr Wozniak in 1987 and is the foundation
of many spaced repetition systems including Anki, SuperMemo, and Mnemosyne.

Algorithm Overview:
------------------
The SM-2 algorithm schedules reviews based on three key metrics:

1. **Quality (q)**: How well you recalled the information (0-5)
   - 5: Perfect recall, immediate and effortless
   - 4: Correct response after hesitation
   - 3: Correct response with difficulty
   - 2: Incorrect, but remembered after seeing answer
   - 1: Incorrect, vaguely familiar
   - 0: Complete blackout, no recollection

2. **Easiness Factor (EF)**: How easy the item is to remember (1.3-2.5)
   - Higher EF = easier to remember = longer intervals
   - Lower EF = harder to remember = shorter intervals
   - Adjusted based on quality of recall

3. **Interval**: Days until next review
   - Increases exponentially for correct recalls
   - Resets to 1 day for poor recalls (quality < 3)

Interval Calculation:
--------------------
- If quality < 3: repetition = 0, interval = 1 day (restart learning)
- If repetition = 0: interval = 1 day (first review)
- If repetition = 1: interval = 6 days (second review)
- If repetition > 1: interval = previous_interval × EF (exponential growth)

Easiness Factor Adjustment:
--------------------------
EF' = EF + (0.1 - (5 - q) × (0.08 + (5 - q) × 0.02))

This formula:
- Increases EF for good recalls (q = 4 or 5)
- Decreases EF for poor recalls (q < 4)
- Is clamped between 1.3 and 2.5

Example Progression:
-------------------
Perfect recall sequence (all quality 5):
- Review 1: 1 day interval
- Review 2: 6 days interval
- Review 3: 15 days interval (6 × 2.5)
- Review 4: 38 days interval (15 × 2.5)
- Review 5: 95 days interval (38 × 2.5)

Mixed quality sequence (5, 4, 3, 5):
- Review 1 (q=5): 1 day, EF=2.5
- Review 2 (q=4): 6 days, EF=2.5
- Review 3 (q=3): 15 days, EF=2.36
- Review 4 (q=5): 35 days, EF=2.46

Poor recall (quality 2 or less) resets to day 1.

References:
----------
- Original paper: https://www.supermemo.com/english/ol/sm2.htm
- Algorithm details: https://en.wikipedia.org/wiki/SuperMemo#Description_of_SM-2_algorithm
"""

from datetime import datetime, timedelta
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class SM2Result:
    """Result of SM-2 calculation."""

    easiness_factor: float  # EF: 1.3 to 2.5
    repetition_number: int  # n: number of consecutive correct answers
    interval_days: int  # I: days until next review
    next_review_date: datetime
    quality: int  # Quality of answer (0-5)


class SM2Algorithm:
    """SuperMemo 2 spaced repetition algorithm implementation."""

    # SM-2 quality rating scale:
    # 5 - Perfect response (immediate, effortless)
    # 4 - Correct response with hesitation
    # 3 - Correct response with serious difficulty
    # 2 - Incorrect response; correct one seemed easy to recall
    # 1 - Incorrect response; correct one seemed familiar
    # 0 - Complete blackout, no recollection

    MIN_EF = 1.3  # Minimum easiness factor
    MAX_EF = 2.5  # Maximum easiness factor (default starting value)
    DEFAULT_EF = 2.5  # Starting easiness factor for new items

    # Quality thresholds
    MIN_PASSING_QUALITY = 3  # Quality < 3 resets the learning process
    PERFECT_QUALITY = 5
    MIN_QUALITY = 0

    # Mastery level thresholds
    MASTERY_LEARNING_REVIEWS = 3  # Reviews needed to reach 'learning'
    MASTERY_REVIEWING_REVIEWS = 6  # Reviews needed to reach 'reviewing'
    MASTERY_MASTERED_REVIEWS = 10  # Reviews needed to reach 'mastered'
    MASTERY_LEARNING_INTERVAL = 7  # Days interval for 'learning' level
    MASTERY_REVIEWING_INTERVAL = 30  # Days interval for 'reviewing' level
    MASTERY_RATE_THRESHOLD = 0.9  # 90% correct rate for 'mastered'

    def __init__(self):
        """Initialize SM-2 algorithm."""
        pass

    def calculate_next_review(
        self,
        quality: int,
        current_ef: float = DEFAULT_EF,
        current_interval: int = 0,
        repetition: int = 0,
    ) -> Dict:
        """
        Calculate next review schedule based on recall quality.

        This is the core SM-2 algorithm implementation.

        Args:
            quality: Recall quality from 0-5
                5 = Perfect recall (immediate, effortless)
                4 = Correct after hesitation
                3 = Correct with difficulty
                2 = Incorrect, remembered after seeing answer
                1 = Incorrect, vaguely familiar
                0 = Complete blackout
            current_ef: Current easiness factor (1.3-2.5), defaults to 2.5
            current_interval: Current interval in days, defaults to 0 (new item)
            repetition: Number of consecutive correct reviews, defaults to 0

        Returns:
            Dictionary containing:
                - new_ef: Updated easiness factor (1.3-2.5)
                - new_interval: Days until next review
                - new_repetition: Updated repetition count
                - next_review_date: Calculated next review date (datetime)

        Example:
            >>> sm2 = SM2Algorithm()
            >>> # First review with perfect recall
            >>> result = sm2.calculate_next_review(quality=5)
            >>> print(result)
            {
                'new_ef': 2.5,
                'new_interval': 1,
                'new_repetition': 1,
                'next_review_date': datetime(2025, 11, 24, ...)
            }

            >>> # Second review with perfect recall
            >>> result = sm2.calculate_next_review(
            ...     quality=5,
            ...     current_ef=2.5,
            ...     current_interval=1,
            ...     repetition=1
            ... )
            >>> print(result['new_interval'])
            6

            >>> # Third review with perfect recall
            >>> result = sm2.calculate_next_review(
            ...     quality=5,
            ...     current_ef=2.5,
            ...     current_interval=6,
            ...     repetition=2
            ... )
            >>> print(result['new_interval'])
            15  # 6 × 2.5

            >>> # Poor recall resets the process
            >>> result = sm2.calculate_next_review(
            ...     quality=2,
            ...     current_ef=2.5,
            ...     current_interval=15,
            ...     repetition=3
            ... )
            >>> print(result['new_interval'], result['new_repetition'])
            1 0  # Reset to beginning
        """
        # Validate inputs
        quality = max(self.MIN_QUALITY, min(self.PERFECT_QUALITY, quality))
        current_ef = max(self.MIN_EF, min(self.MAX_EF, current_ef))

        # Calculate new easiness factor
        # Formula: EF' = EF + (0.1 - (5 - q) × (0.08 + (5 - q) × 0.02))
        ef_delta = 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
        new_ef = current_ef + ef_delta

        # Clamp EF to valid range
        new_ef = max(self.MIN_EF, min(self.MAX_EF, new_ef))

        # Calculate new repetition count and interval
        if quality < self.MIN_PASSING_QUALITY:
            # Poor recall (quality 0-2): Reset to beginning
            new_repetition = 0
            new_interval = 1
        else:
            # Good recall (quality 3-5): Progress forward
            new_repetition = repetition + 1

            if new_repetition == 1:
                # First successful review
                new_interval = 1
            elif new_repetition == 2:
                # Second successful review
                new_interval = 6
            else:
                # Third and subsequent reviews: exponential growth
                # Interval = previous_interval × EF
                new_interval = round(current_interval * new_ef)

        # Calculate next review date
        next_review_date = datetime.now() + timedelta(days=new_interval)

        return {
            "new_ef": round(new_ef, 2),
            "new_interval": new_interval,
            "new_repetition": new_repetition,
            "next_review_date": next_review_date,
        }

    def calculate(
        self,
        quality: int,
        easiness_factor: float = DEFAULT_EF,
        repetition_number: int = 0,
        previous_interval: int = 0,
        base_date: Optional[datetime] = None,
    ) -> SM2Result:
        """Calculate next review using SM-2 algorithm.

        This method is kept for backward compatibility. Use calculate_next_review() for new code.

        Args:
            quality: Quality of answer (0-5)
            easiness_factor: Current easiness factor (1.3-2.5)
            repetition_number: Number of consecutive correct answers
            previous_interval: Previous interval in days
            base_date: Base date for next review (defaults to now)

        Returns:
            SM2Result with updated parameters
        """
        if base_date is None:
            base_date = datetime.now()

        # Calculate new easiness factor
        new_ef = self._calculate_easiness_factor(easiness_factor, quality)

        # Determine if answer was correct (quality >= 3)
        if quality < 3:
            # Incorrect answer - reset repetition counter
            new_repetition = 0
            new_interval = 1  # Review again tomorrow
        else:
            # Correct answer - increment repetition
            new_repetition = repetition_number + 1

            # Calculate interval based on repetition number
            if new_repetition == 1:
                new_interval = 1
            elif new_repetition == 2:
                new_interval = 6
            else:
                # I(n) = I(n-1) * EF
                new_interval = int(round(previous_interval * new_ef))

        # Calculate next review date
        next_review = base_date + timedelta(days=new_interval)

        return SM2Result(
            easiness_factor=new_ef,
            repetition_number=new_repetition,
            interval_days=new_interval,
            next_review_date=next_review,
            quality=quality,
        )

    def _calculate_easiness_factor(self, current_ef: float, quality: int) -> float:
        """Calculate new easiness factor based on answer quality.

        Args:
            current_ef: Current easiness factor
            quality: Quality of answer (0-5)

        Returns:
            New easiness factor (1.3-2.5)
        """
        # EF' = EF + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
        new_ef = current_ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))

        # Clamp to valid range
        if new_ef < self.MIN_EF:
            new_ef = self.MIN_EF
        elif new_ef > self.MAX_EF:
            new_ef = self.MAX_EF

        return round(new_ef, 2)

    def get_review_quality_from_score(
        self, correct: bool, time_taken: int, hint_used: bool = False, expected_time: int = 180
    ) -> int:
        """
        Convert quiz performance metrics to SM-2 quality score (0-5).

        This helper method translates real-world quiz performance into the
        SM-2 quality scale. It considers correctness, speed, and hint usage.

        Quality Mapping:
        ---------------
        5 (Perfect): Correct, fast (<50% expected time), no hints
        4 (Good): Correct, normal speed or one hint used
        3 (Fair): Correct but slow (>150% expected time) or multiple hints
        2 (Poor): Incorrect but close (recognized after seeing answer)
        1 (Bad): Incorrect, vaguely familiar
        0 (Failure): Incorrect, complete confusion

        Args:
            correct: Whether the answer was correct
            time_taken: Time taken in seconds
            hint_used: Whether hints were used
            expected_time: Expected time in seconds (default: 180 = 3 minutes)

        Returns:
            SM-2 quality score (0-5)

        Example:
            >>> sm2 = SM2Algorithm()
            >>> # Perfect performance: correct in 60s, no hints (expected 180s)
            >>> sm2.get_review_quality_from_score(
            ...     correct=True,
            ...     time_taken=60,
            ...     hint_used=False,
            ...     expected_time=180
            ... )
            5

            >>> # Good performance: correct in 150s with hint
            >>> sm2.get_review_quality_from_score(
            ...     correct=True,
            ...     time_taken=150,
            ...     hint_used=True,
            ...     expected_time=180
            ... )
            4

            >>> # Fair performance: correct but took 300s
            >>> sm2.get_review_quality_from_score(
            ...     correct=True,
            ...     time_taken=300,
            ...     hint_used=False,
            ...     expected_time=180
            ... )
            3

            >>> # Incorrect answer
            >>> sm2.get_review_quality_from_score(
            ...     correct=False,
            ...     time_taken=120,
            ...     hint_used=False
            ... )
            2
        """
        if not correct:
            # Incorrect answers: quality 0-2
            # For now, we default to 2 (recognized after seeing answer)
            # In a full implementation, this could be determined by:
            # - How close the answer was (similarity check)
            # - Whether the user gave up vs attempted
            # - Number of attempts made
            return 2

        # Correct answers: quality 3-5 based on performance
        time_ratio = time_taken / expected_time

        if time_ratio < 0.5 and not hint_used:
            # Perfect: Fast and no hints needed
            return 5
        elif time_ratio <= 1.0 and not hint_used:
            # Good: Normal time, no hints
            return 4
        elif hint_used or time_ratio <= 1.5:
            # Fair: Either used hints or took extra time
            return 3 if not hint_used else 4
        else:
            # Correct but struggled: Very slow
            return 3

    def convert_score_to_quality(
        self, score: float, hint_used: bool = False, time_ratio: Optional[float] = None
    ) -> int:
        """Convert quiz score to SM-2 quality rating.

        This method is kept for backward compatibility. Use get_review_quality_from_score() for new code.

        Args:
            score: Score from 0.0 to 1.0
            hint_used: Whether hints were used
            time_ratio: Time taken vs expected (optional)

        Returns:
            Quality rating (0-5)
        """
        # Base quality on score
        if score >= 0.95:
            quality = 5  # Perfect
        elif score >= 0.85:
            quality = 4  # Good
        elif score >= 0.70:
            quality = 3  # Passable
        elif score >= 0.50:
            quality = 2  # Difficult but recognized
        elif score >= 0.20:
            quality = 1  # Familiar but wrong
        else:
            quality = 0  # Complete failure

        # Adjust for hints (reduce by 1 if hints were used and quality > 0)
        if hint_used and quality > 0:
            quality = max(0, quality - 1)

        # Optional: adjust for time taken
        if time_ratio is not None:
            # If took significantly longer than expected, reduce quality
            if time_ratio > 2.0 and quality > 0:
                quality = max(0, quality - 1)

        return quality

    def is_due_for_review(
        self, last_review: datetime, interval_days: int, current_date: Optional[datetime] = None
    ) -> bool:
        """Check if an item is due for review.

        Args:
            last_review: Date of last review
            interval_days: Interval in days until next review
            current_date: Current date (defaults to now)

        Returns:
            True if due for review
        """
        if current_date is None:
            current_date = datetime.now()

        next_review = last_review + timedelta(days=interval_days)
        return current_date >= next_review

    def get_next_review_date(self, last_review: datetime, interval_days: int) -> datetime:
        """Get the next review date.

        Args:
            last_review: Date of last review
            interval_days: Interval in days until next review

        Returns:
            Next review datetime
        """
        return last_review + timedelta(days=interval_days)

    def get_mastery_level(
        self, repetition: int, interval_days: int, correct_count: int, total_count: int
    ) -> str:
        """
        Determine mastery level based on SM-2 state and performance history.

        Mastery levels represent the learning stage:
        - 'new': Never reviewed (repetition = 0)
        - 'learning': Early reviews, short intervals
        - 'reviewing': Regular reviews, medium intervals
        - 'mastered': Long-term retention, long intervals, high accuracy

        Mastery Level Criteria:
        ----------------------
        new:
            - repetition = 0
            - Never successfully reviewed

        learning:
            - 1-2 successful reviews (repetition 1-2)
            - OR interval < 7 days
            - Still building initial memory

        reviewing:
            - 3-9 successful reviews (repetition 3-9)
            - interval 7-30 days
            - Established memory, regular reinforcement

        mastered:
            - 10+ successful reviews (repetition >= 10)
            - interval > 30 days
            - 90%+ correct rate (if enough attempts)
            - Strong long-term retention

        Args:
            repetition: Number of consecutive correct reviews
            interval_days: Current interval in days
            correct_count: Total number of correct reviews
            total_count: Total number of review attempts

        Returns:
            Mastery level: 'new', 'learning', 'reviewing', or 'mastered'

        Example:
            >>> sm2 = SM2Algorithm()
            >>> # Brand new item
            >>> sm2.get_mastery_level(
            ...     repetition=0,
            ...     interval_days=0,
            ...     correct_count=0,
            ...     total_count=0
            ... )
            'new'

            >>> # After first successful review
            >>> sm2.get_mastery_level(
            ...     repetition=1,
            ...     interval_days=1,
            ...     correct_count=1,
            ...     total_count=1
            ... )
            'learning'

            >>> # After several successful reviews
            >>> sm2.get_mastery_level(
            ...     repetition=5,
            ...     interval_days=15,
            ...     correct_count=5,
            ...     total_count=6
            ... )
            'reviewing'

            >>> # Long-term mastery
            >>> sm2.get_mastery_level(
            ...     repetition=12,
            ...     interval_days=45,
            ...     correct_count=12,
            ...     total_count=13
            ... )
            'mastered'
        """
        # Calculate correct rate (avoid division by zero)
        correct_rate = correct_count / total_count if total_count > 0 else 0.0

        # New: Never reviewed
        if repetition == 0:
            return "new"

        # Mastered: Long-term retention with high accuracy
        if (
            repetition >= self.MASTERY_MASTERED_REVIEWS
            and interval_days > self.MASTERY_REVIEWING_INTERVAL
            and (total_count < 5 or correct_rate >= self.MASTERY_RATE_THRESHOLD)
        ):
            return "mastered"

        # Reviewing: Established memory
        if (
            repetition >= self.MASTERY_LEARNING_REVIEWS
            and interval_days >= self.MASTERY_LEARNING_INTERVAL
        ):
            return "reviewing"

        # Learning: Building initial memory
        return "learning"
