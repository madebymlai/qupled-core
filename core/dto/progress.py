"""Progress-related Data Transfer Objects.

DTOs for knowledge gaps, learning paths, and progress analytics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from .mastery import GapSeverity


@dataclass
class KnowledgeGap:
    """Identified knowledge gap for a topic.

    Represents a topic where the user's mastery is below threshold
    and provides remediation recommendations.

    Attributes:
        topic_id: Unique identifier for the topic
        topic_name: Human-readable topic name
        gap_severity: How severe the knowledge gap is
        current_mastery: Current mastery percentage (0-100)
        target_mastery: Target mastery percentage (default 80)
        common_mistakes: List of identified common mistakes
        recommended_actions: Ordered list of recommended actions
        exercises_to_review: Number of exercises recommended
        estimated_time_minutes: Estimated time to close the gap
    """

    topic_id: str
    topic_name: str
    gap_severity: GapSeverity
    current_mastery: float
    target_mastery: float = 80.0
    common_mistakes: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    exercises_to_review: int = 10
    estimated_time_minutes: int = 30


@dataclass
class LearningPathItem:
    """Single item in a recommended learning path.

    Represents one step in the personalized learning journey.

    Attributes:
        item_type: Type of item ("topic", "review", "quiz", "knowledge_item")
        item_id: Unique identifier for the item
        title: Display title
        description: Optional description
        difficulty: Difficulty level ("easy", "medium", "hard")
        estimated_time_minutes: Estimated time to complete
        priority: Priority level ("high", "medium", "low")
        reason: Explanation of why this item is recommended
        order: Position in the learning path
    """

    item_type: str
    item_id: str
    title: str
    difficulty: str
    estimated_time_minutes: int
    priority: str
    reason: str
    order: int
    description: Optional[str] = None


@dataclass
class LearningPathResult:
    """Complete learning path for a course.

    Contains all recommended items and summary statistics.

    Attributes:
        course_id: Course code
        course_name: Human-readable course name
        overall_mastery: Overall course mastery (0-100)
        recommended_items: Ordered list of learning items
        knowledge_gaps: Identified knowledge gaps
        total_estimated_time_minutes: Total time for all items
        generated_at: When the path was generated
    """

    course_id: str
    course_name: str
    overall_mastery: float
    recommended_items: List[LearningPathItem]
    knowledge_gaps: List[KnowledgeGap]
    total_estimated_time_minutes: int
    generated_at: datetime
