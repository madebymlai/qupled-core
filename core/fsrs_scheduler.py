"""
FSRS (Free Spaced Repetition Scheduler) Implementation

FSRS is a modern spaced repetition algorithm that improves upon SM-2 by:
1. Using a mathematical model based on the forgetting curve
2. Separating difficulty from stability (memory strength)
3. Providing more accurate retention predictions

Key Concepts:
------------
- **Stability (S)**: Days until memory decays to 90% retention probability
- **Difficulty (D)**: Inherent difficulty of the item (1-10 scale)
- **State**: Learning phase (New, Learning, Review, Relearning)
- **Rating**: User feedback (1=Again, 2=Hard, 3=Good, 4=Easy)

The algorithm calculates optimal review intervals based on these parameters,
typically resulting in more efficient learning compared to SM-2.

References:
----------
- FSRS algorithm: https://github.com/open-spaced-repetition/fsrs4anki
- Paper: https://arxiv.org/abs/2204.10746
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from fsrs import Card, Rating, Scheduler, State


@dataclass
class FSRSResult:
    """Result of FSRS scheduling calculation."""

    stability: float  # Days until 90% retention
    difficulty: float  # Item difficulty (1-10)
    state: int  # 0=New, 1=Learning, 2=Review, 3=Relearning
    reps: int  # Number of reviews
    next_review_date: datetime
    interval_days: int  # Days until next review


class FSRSScheduler:
    """
    FSRS spaced repetition scheduler.

    Wraps the fsrs library to provide a clean interface for Qupled.
    Uses default FSRS parameters (will be optimized from ReviewLog data later).
    """

    def __init__(self, desired_retention: float = 0.9):
        """
        Initialize FSRS scheduler.

        Args:
            desired_retention: Target retention probability (default: 0.9 = 90%)
        """
        self.fsrs = Scheduler()
        self.desired_retention = desired_retention

    def schedule_review(
        self,
        rating: int,
        stability: Optional[float] = None,
        difficulty: Optional[float] = None,
        state: int = 0,
        last_review: Optional[datetime] = None,
        reps: int = 0,
    ) -> FSRSResult:
        """
        Calculate next review schedule based on rating.

        Args:
            rating: FSRS rating (1=Again, 2=Hard, 3=Good, 4=Easy)
            stability: Current stability in days (None for new items)
            difficulty: Current difficulty 1-10 (None for new items)
            state: Current state (0=New, 1=Learning, 2=Review, 3=Relearning)
            last_review: When the item was last reviewed (None for new items)
            reps: Number of previous reviews

        Returns:
            FSRSResult with updated scheduling parameters
        """
        # Create or reconstruct card
        card = Card()
        if stability is not None and last_review is not None:
            # Existing card - set state from stored values
            card.stability = stability
            card.difficulty = difficulty if difficulty is not None else 5.0
            card.state = State(state)
            card.step = reps  # fsrs uses 'step' for repetition tracking
            card.last_review = last_review

        # Map rating to FSRS Rating enum
        fsrs_rating = self._map_rating(rating)

        # Get scheduling info - review_card returns (updated_card, review_log)
        now = datetime.now(timezone.utc)
        scheduled_card, _ = self.fsrs.review_card(card, fsrs_rating, now)

        # Adjust difficulty: make "Good" ratings decrease difficulty meaningfully
        # FSRS only decreases by -0.01 for Good, we want -0.5 total (halfway to Easy's -1.7)
        if rating == 3:  # Good
            scheduled_card.difficulty = max(1.0, scheduled_card.difficulty - 0.5)

        # Calculate interval
        if scheduled_card.due:
            interval_days = max(1, (scheduled_card.due - now).days)
        else:
            interval_days = 1

        return FSRSResult(
            stability=scheduled_card.stability,
            difficulty=scheduled_card.difficulty,
            state=scheduled_card.state.value,
            reps=scheduled_card.step,  # fsrs uses 'step' for repetition count
            next_review_date=scheduled_card.due or (now + timedelta(days=1)),
            interval_days=interval_days,
        )

    def convert_score_to_rating(self, score: float) -> int:
        """
        Convert 0-1 score to FSRS rating (1-4).

        Score ranges (from plan):
        - score < 0.5 -> 1 (Again)
        - score < 0.7 -> 2 (Hard)
        - score < 0.9 -> 3 (Good)
        - score >= 0.9 -> 4 (Easy)

        Args:
            score: Score from 0.0 to 1.0

        Returns:
            FSRS rating 1-4
        """
        if score < 0.5:
            return 1  # Again
        elif score < 0.7:
            return 2  # Hard
        elif score < 0.9:
            return 3  # Good
        else:
            return 4  # Easy

    def _map_rating(self, rating: int) -> Rating:
        """Map integer rating to FSRS Rating enum."""
        rating_map = {
            1: Rating.Again,
            2: Rating.Hard,
            3: Rating.Good,
            4: Rating.Easy,
        }
        return rating_map.get(rating, Rating.Good)

    def estimate_stability_from_sm2(
        self,
        interval: int,
        repetitions: int,
    ) -> tuple[float, float, int]:
        """
        Estimate FSRS stability from SM2 parameters for backfill.

        Args:
            interval: SM2 interval in days
            repetitions: SM2 repetition count

        Returns:
            Tuple of (estimated_stability, default_difficulty, fsrs_state)
        """
        # Heuristic: stability ≈ interval * 0.9
        # This assumes the SM2 interval was calibrated for ~90% retention
        estimated_stability = max(1.0, interval * 0.9)

        # Default difficulty (medium)
        default_difficulty = 5.0

        # Map SM2 repetitions to FSRS state
        if repetitions == 0:
            fsrs_state = 0  # New
        elif repetitions <= 2:
            fsrs_state = 1  # Learning
        else:
            fsrs_state = 2  # Review

        return estimated_stability, default_difficulty, fsrs_state
