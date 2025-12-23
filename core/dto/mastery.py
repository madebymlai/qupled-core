"""Mastery-related Data Transfer Objects.

These DTOs enable database-agnostic business logic by providing
a common data format that can be used with any data source
(SQLite, PostgreSQL, in-memory, etc.).
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional


class MasteryLevel(Enum):
    """Exercise mastery level based on SM-2 algorithm state."""

    NOT_STARTED = "not_started"
    LEARNING = "learning"
    REVIEWING = "reviewing"
    MASTERED = "mastered"


class MasteryTrend(Enum):
    """Trend of mastery over time."""

    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    NEW = "new"


class GapSeverity(Enum):
    """Severity of a knowledge gap."""

    CRITICAL = "critical"  # mastery < 0.3
    SIGNIFICANT = "significant"  # mastery < 0.5
    MINOR = "minor"  # mastery < threshold but >= 0.5


@dataclass(frozen=True)
class ExerciseReviewData:
    """Exercise review data from any data source.

    This is the fundamental unit for mastery calculations.
    Can be populated from SQLite, PostgreSQL, or any other source.

    Attributes:
        exercise_id: Unique identifier for the exercise
        mastery_level: Current SM-2 mastery level
        interval_days: Days until next scheduled review
        total_reviews: Total number of review attempts
        correct_reviews: Number of correct review attempts
        last_reviewed_at: When the exercise was last reviewed
    """

    exercise_id: str
    mastery_level: MasteryLevel
    interval_days: int
    total_reviews: int
    correct_reviews: int
    last_reviewed_at: Optional[datetime] = None


@dataclass(frozen=True)
class TopicMasteryInput:
    """Input for topic mastery calculation.

    Groups exercise reviews by topic for aggregate calculations.

    Attributes:
        topic_id: Unique identifier for the topic
        topic_name: Human-readable topic name
        exercise_reviews: All exercise reviews for this topic
        total_exercises: Total exercises in topic (including unreviewed)
    """

    topic_id: str
    topic_name: str
    exercise_reviews: List[ExerciseReviewData]
    total_exercises: int


@dataclass
class TopicMasteryResult:
    """Result of topic mastery calculation.

    Contains all computed mastery metrics for a topic.

    Attributes:
        topic_id: Unique identifier for the topic
        topic_name: Human-readable topic name
        mastery_score: Overall mastery (0.0-1.0)
        mastery_trend: Direction of mastery change
        exercises_total: Total exercises in topic
        exercises_reviewed: Number of exercises with at least one review
        correct_count: Total correct reviews across all exercises
        accuracy_percentage: Overall accuracy (0-100)
        last_practiced_at: Most recent review timestamp
    """

    topic_id: str
    topic_name: str
    mastery_score: float
    mastery_trend: MasteryTrend
    exercises_total: int
    exercises_reviewed: int
    correct_count: int
    accuracy_percentage: float
    last_practiced_at: Optional[datetime] = None
