"""
Task Type Definitions for Provider Routing.

This module defines the task categories used by the Provider Routing Architecture
to determine which LLM provider should handle different types of operations.

Task categories are designed to align with:
- Cost optimization (bulk operations use cheaper providers)
- Quality requirements (interactive learning needs better models)
- Privacy considerations (minimize exposure of sensitive data)
"""

from enum import Enum


class TaskType(Enum):
    """Task categories for provider routing.

    Each task type represents a different usage pattern with specific
    requirements for model quality, cost, and latency.

    Task Categories:
    - BULK_ANALYSIS: High-volume operations that can use faster/cheaper models
    - INTERACTIVE: Real-time interactions requiring good quality but fast response
    - PREMIUM: High-quality deep explanations and adaptive teaching

    Usage:
        from core.task_types import TaskType

        # During PDF ingestion
        provider = router.route(TaskType.BULK_ANALYSIS, profile="free")

        # During interactive tutoring
        provider = router.route(TaskType.INTERACTIVE, profile="pro")

        # For premium explanations
        provider = router.route(TaskType.PREMIUM, profile="pro")
    """

    # Bulk processing operations (PDF ingestion, exercise splitting, batch analysis)
    # Characteristics: High volume, can tolerate some latency, cost-sensitive
    # Privacy: Low (exercise text)
    BULK_ANALYSIS = "bulk_analysis"

    # Interactive learning operations (tutoring, practice, quiz)
    # Characteristics: Real-time, moderate quality needs, user-facing
    # Privacy: Medium (user answers)
    INTERACTIVE = "interactive"

    # Premium deep-dive operations (adaptive teaching, proof guidance, concept graphs)
    # Characteristics: High quality needs, low volume, willing to pay for quality
    # Privacy: High (learning patterns, misconceptions)
    PREMIUM = "premium"

    def __str__(self) -> str:
        """String representation of task type."""
        return self.value

    @property
    def description(self) -> str:
        """Human-readable description of the task type."""
        descriptions = {
            TaskType.BULK_ANALYSIS: "High-volume analysis operations (PDF ingestion, batch processing)",
            TaskType.INTERACTIVE: "Real-time interactive learning (tutoring, quizzes, practice)",
            TaskType.PREMIUM: "Premium deep explanations (adaptive teaching, proof guidance)"
        }
        return descriptions.get(self, "Unknown task type")

    @property
    def privacy_level(self) -> str:
        """Privacy sensitivity of this task type."""
        levels = {
            TaskType.BULK_ANALYSIS: "low",
            TaskType.INTERACTIVE: "medium",
            TaskType.PREMIUM: "high"
        }
        return levels.get(self, "unknown")

    @classmethod
    def from_string(cls, value: str) -> 'TaskType':
        """Convert string to TaskType enum.

        Args:
            value: String value (e.g., "bulk_analysis", "interactive")

        Returns:
            TaskType enum

        Raises:
            ValueError: If value is not a valid task type
        """
        try:
            return cls(value)
        except ValueError:
            valid_types = [t.value for t in cls]
            raise ValueError(
                f"Invalid task type '{value}'. "
                f"Valid types: {', '.join(valid_types)}"
            )
