"""Database-agnostic progress analysis.

This module contains all business logic for mastery calculation, gap detection,
and learning path generation. All methods that don't need database access are
static, making them easily testable and reusable.

Extracted from examina-cloud/backend/app/api/v1/progress.py to enable:
- Reuse across CLI and web
- Unit testing without database
- Clean separation of business logic from data access
"""

from datetime import datetime, timezone
from typing import List, Optional, Tuple

from core.dto.mastery import (
    ExerciseReviewData,
    GapSeverity,
    MasteryLevel,
    MasteryTrend,
    TopicMasteryInput,
    TopicMasteryResult,
)
from core.dto.progress import KnowledgeGap, LearningPathItem
from core.ports.mastery_repository import MasteryRepository


class ProgressAnalyzer:
    """Database-agnostic progress analysis.

    Static methods handle pure calculations (no database).
    Instance methods use the injected repository for data access.

    Usage:
        # Pure calculation (no database needed)
        score = ProgressAnalyzer.calculate_mastery_score(reviews)

        # With repository
        analyzer = ProgressAnalyzer(repo)
        mastery = analyzer.get_topic_mastery(user_id, topic_id, topic_name)
    """

    def __init__(self, mastery_repo: Optional[MasteryRepository] = None):
        """Initialize with optional repository.

        Args:
            mastery_repo: Repository for data access. Required for instance methods.
        """
        self._repo = mastery_repo

    # ==================== STATIC METHODS (Pure Calculations) ====================

    @staticmethod
    def calculate_mastery_score(
        reviews: List[ExerciseReviewData],
        now: Optional[datetime] = None,
    ) -> float:
        """Calculate mastery score from exercise reviews.

        Implements weighted formula from progress.py:71-150:
        - 40% accuracy (correct/total ratio)
        - 30% mastery level average (enum to numeric)
        - 20% interval score (normalized to 90 days)
        - 10% recency bonus (if 3+ reviews in last 7 days)

        Args:
            reviews: List of exercise review data
            now: Current datetime for recency calculation (defaults to UTC now)

        Returns:
            Mastery score between 0.0 and 1.0
        """
        if not reviews:
            return 0.0

        if now is None:
            now = datetime.now(timezone.utc)

        # Accuracy score (40%)
        total_reviews = sum(r.total_reviews for r in reviews)
        correct_reviews = sum(r.correct_reviews for r in reviews)
        accuracy = correct_reviews / total_reviews if total_reviews > 0 else 0.0

        # Interval score (20%) - normalized to 90 days max
        avg_interval = sum(r.interval_days for r in reviews) / len(reviews)
        interval_score = min(avg_interval / 90.0, 1.0)

        # Mastery level score (30%)
        level_scores = {
            MasteryLevel.NOT_STARTED: 0.0,
            MasteryLevel.LEARNING: 0.3,
            MasteryLevel.REVIEWING: 0.6,
            MasteryLevel.MASTERED: 1.0,
        }
        avg_mastery_level = sum(level_scores.get(r.mastery_level, 0.0) for r in reviews) / len(
            reviews
        )

        # Recency bonus (10%) - if 3+ reviews in last 7 days
        recent_reviews = [
            r for r in reviews if r.last_reviewed_at and (now - r.last_reviewed_at).days <= 7
        ]
        recency_bonus = 0.05 if len(recent_reviews) >= 3 else 0.0

        # Combine with weights
        mastery = 0.4 * accuracy + 0.3 * avg_mastery_level + 0.2 * interval_score + recency_bonus

        return min(mastery, 1.0)

    @staticmethod
    def determine_trend(reviews: List[ExerciseReviewData]) -> MasteryTrend:
        """Determine mastery trend from review distribution.

        From progress.py:287-310:
        - >50% mastered → improving
        - >50% learning → new
        - else → stable

        Args:
            reviews: List of exercise review data

        Returns:
            MasteryTrend enum value
        """
        if not reviews:
            return MasteryTrend.NEW

        total = len(reviews)
        mastered_count = sum(1 for r in reviews if r.mastery_level == MasteryLevel.MASTERED)
        learning_count = sum(1 for r in reviews if r.mastery_level == MasteryLevel.LEARNING)

        if mastered_count / total > 0.5:
            return MasteryTrend.IMPROVING
        elif learning_count / total > 0.5:
            return MasteryTrend.NEW
        else:
            return MasteryTrend.STABLE

    @staticmethod
    def classify_gap_severity(mastery_score: float) -> GapSeverity:
        """Classify knowledge gap severity.

        From progress.py:432-438:
        - mastery < 0.3 → critical
        - mastery < 0.5 → significant
        - else → minor

        Args:
            mastery_score: Mastery score between 0.0 and 1.0

        Returns:
            GapSeverity enum value
        """
        if mastery_score < 0.3:
            return GapSeverity.CRITICAL
        elif mastery_score < 0.5:
            return GapSeverity.SIGNIFICANT
        return GapSeverity.MINOR

    @staticmethod
    def calculate_recommendations(mastery: float) -> Tuple[int, int]:
        """Calculate recommended exercises and time based on mastery.

        From progress.py:440-449.

        Args:
            mastery: Mastery score between 0.0 and 1.0

        Returns:
            Tuple of (recommended_exercises, estimated_time_minutes)
        """
        if mastery < 0.3:
            return (20, 60)
        elif mastery < 0.5:
            return (15, 45)
        return (10, 30)

    @staticmethod
    def generate_recommended_actions(
        topic_name: str,
        mastery: float,
        knowledge_item_names: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate recommended actions for addressing a knowledge gap.

        From progress.py:477-494.

        Args:
            topic_name: Name of the topic
            mastery: Current mastery score
            knowledge_item_names: Optional list of core loop names to focus on

        Returns:
            Ordered list of recommended action strings
        """
        recommended_exercises, _ = ProgressAnalyzer.calculate_recommendations(mastery)
        actions = []

        if mastery < 0.3:
            actions.append("Start with easier exercises to build foundations")

        actions.append(f"Review the {topic_name} concept materials")
        actions.append(f"Complete {recommended_exercises} practice exercises")

        if knowledge_item_names:
            focus_loops = knowledge_item_names[:3]
            actions.append(f"Focus on: {', '.join(focus_loops)}")

        actions.append(f"Take a quiz on {topic_name}")

        return actions

    @staticmethod
    def detect_common_mistakes(
        reviews: List[ExerciseReviewData],
        topic_name: str,
    ) -> List[str]:
        """Detect common mistakes from review patterns.

        From progress.py:451-472.

        Args:
            reviews: Exercise review data
            topic_name: Topic name for messages

        Returns:
            List of identified common mistake descriptions
        """
        mistakes = []

        if not reviews:
            mistakes.append("No practice attempts yet")
            return mistakes

        # Low accuracy check
        total = sum(r.total_reviews for r in reviews)
        correct = sum(r.correct_reviews for r in reviews)
        if total > 0 and (correct / total) < 0.5:
            mistakes.append(f"Low accuracy on {topic_name} exercises")

        # Stuck in learning mode check
        learning_count = sum(1 for r in reviews if r.mastery_level == MasteryLevel.LEARNING)
        if len(reviews) > 0 and learning_count / len(reviews) > 0.6:
            mistakes.append("Difficulty progressing past learning stage")

        return mistakes

    @staticmethod
    def calculate_topic_mastery(topic_input: TopicMasteryInput) -> TopicMasteryResult:
        """Calculate complete mastery result for a topic.

        Combines all static methods to produce full topic analysis.

        Args:
            topic_input: Topic with exercise reviews

        Returns:
            TopicMasteryResult with all computed metrics
        """
        reviews = topic_input.exercise_reviews
        mastery_score = ProgressAnalyzer.calculate_mastery_score(reviews)
        trend = ProgressAnalyzer.determine_trend(reviews)

        # Aggregate statistics
        correct_count = sum(r.correct_reviews for r in reviews)
        total_reviews = sum(r.total_reviews for r in reviews)
        accuracy = (correct_count / total_reviews * 100) if total_reviews > 0 else 0.0

        # Find last practiced timestamp
        last_practiced = None
        for r in reviews:
            if r.last_reviewed_at:
                if last_practiced is None or r.last_reviewed_at > last_practiced:
                    last_practiced = r.last_reviewed_at

        return TopicMasteryResult(
            topic_id=topic_input.topic_id,
            topic_name=topic_input.topic_name,
            mastery_score=mastery_score,
            mastery_trend=trend,
            exercises_total=topic_input.total_exercises,
            exercises_reviewed=len(reviews),
            correct_count=correct_count,
            accuracy_percentage=accuracy,
            last_practiced_at=last_practiced,
        )

    @staticmethod
    def build_knowledge_gap(
        topic_id: str,
        topic_name: str,
        mastery: float,
        reviews: List[ExerciseReviewData],
        knowledge_item_names: Optional[List[str]] = None,
    ) -> KnowledgeGap:
        """Build a KnowledgeGap object with all recommendations.

        Args:
            topic_id: Topic identifier
            topic_name: Topic name
            mastery: Current mastery score (0.0-1.0)
            reviews: Exercise reviews for common mistake detection
            knowledge_item_names: Optional core loop names for recommendations

        Returns:
            Complete KnowledgeGap with recommendations
        """
        severity = ProgressAnalyzer.classify_gap_severity(mastery)
        recommended_exercises, estimated_time = ProgressAnalyzer.calculate_recommendations(mastery)
        common_mistakes = ProgressAnalyzer.detect_common_mistakes(reviews, topic_name)
        recommended_actions = ProgressAnalyzer.generate_recommended_actions(
            topic_name, mastery, knowledge_item_names
        )

        return KnowledgeGap(
            topic_id=topic_id,
            topic_name=topic_name,
            gap_severity=severity,
            current_mastery=mastery * 100,  # Convert to percentage
            target_mastery=80.0,
            common_mistakes=common_mistakes,
            recommended_actions=recommended_actions,
            exercises_to_review=recommended_exercises,
            estimated_time_minutes=estimated_time,
        )

    @staticmethod
    def prioritize_learning_items(
        topic_results: List[TopicMasteryResult],
        knowledge_gaps: List[KnowledgeGap],
        max_items: int = 10,
    ) -> List[LearningPathItem]:
        """Build prioritized learning path items.

        From progress.py:580-705. Creates items in priority order:
        1. Critical gaps (top 3)
        2. Significant gaps (top 2)
        3. In-progress topics (mastery 0.6-0.8)
        4. Review items for mastered topics

        Args:
            topic_results: All topic mastery results
            knowledge_gaps: Identified knowledge gaps
            max_items: Maximum items to return

        Returns:
            Ordered list of learning path items
        """
        items = []
        order = 0

        # Tier 1: Critical gaps (top 3)
        critical_gaps = [g for g in knowledge_gaps if g.gap_severity == GapSeverity.CRITICAL]
        for gap in critical_gaps[:3]:
            if len(items) >= max_items:
                break
            order += 1
            items.append(
                LearningPathItem(
                    item_type="topic",
                    item_id=gap.topic_id,
                    title=gap.topic_name,
                    description=f"Critical knowledge gap requiring immediate attention",
                    difficulty="hard",
                    estimated_time_minutes=gap.estimated_time_minutes,
                    priority="high",
                    reason=f"Critical knowledge gap. Current mastery: {gap.current_mastery:.1f}%",
                    order=order,
                )
            )

        # Tier 2: Significant gaps (top 2)
        significant_gaps = [g for g in knowledge_gaps if g.gap_severity == GapSeverity.SIGNIFICANT]
        for gap in significant_gaps[:2]:
            if len(items) >= max_items:
                break
            order += 1
            items.append(
                LearningPathItem(
                    item_type="topic",
                    item_id=gap.topic_id,
                    title=gap.topic_name,
                    description=f"Significant knowledge gap",
                    difficulty="medium",
                    estimated_time_minutes=gap.estimated_time_minutes,
                    priority="high",
                    reason=f"Significant knowledge gap. Current mastery: {gap.current_mastery:.1f}%",
                    order=order,
                )
            )

        # Tier 3: In-progress topics (mastery 0.6-0.8)
        in_progress = [t for t in topic_results if 0.6 <= t.mastery_score < 0.8]
        for topic in in_progress:
            if len(items) >= max_items:
                break
            order += 1
            items.append(
                LearningPathItem(
                    item_type="topic",
                    item_id=topic.topic_id,
                    title=topic.topic_name,
                    description="Continue making progress",
                    difficulty="medium",
                    estimated_time_minutes=30,
                    priority="medium",
                    reason=f"You're making good progress! Current mastery: {topic.mastery_score * 100:.1f}%",
                    order=order,
                )
            )

        # Tier 4: Review items for mastered topics
        mastered = [t for t in topic_results if t.mastery_score >= 0.8]
        if mastered and len(items) < max_items:
            topic = mastered[0]
            order += 1
            items.append(
                LearningPathItem(
                    item_type="review",
                    item_id=topic.topic_id,
                    title=f"Review: {topic.topic_name}",
                    description="Reinforce mastered concepts",
                    difficulty="easy",
                    estimated_time_minutes=15,
                    priority="low",
                    reason="Reinforce mastered concepts with spaced repetition",
                    order=order,
                )
            )

        return items

    # ==================== INSTANCE METHODS (Use Repository) ====================

    def get_topic_mastery(
        self,
        user_id: str,
        topic_id: str,
        topic_name: str,
    ) -> TopicMasteryResult:
        """Get topic mastery using repository.

        Args:
            user_id: User identifier
            topic_id: Topic identifier
            topic_name: Topic name

        Returns:
            TopicMasteryResult

        Raises:
            ValueError: If repository not configured
        """
        if self._repo is None:
            raise ValueError("Repository not configured")

        topic_input = self._repo.get_topic_mastery_input(user_id, topic_id, topic_name)
        return self.calculate_topic_mastery(topic_input)

    def identify_knowledge_gaps(
        self,
        user_id: str,
        course_code: str,
        threshold: float = 0.6,
    ) -> List[KnowledgeGap]:
        """Identify topics below mastery threshold.

        Args:
            user_id: User identifier
            course_code: Course code
            threshold: Mastery threshold (default 0.6)

        Returns:
            List of KnowledgeGap sorted by severity and mastery

        Raises:
            ValueError: If repository not configured
        """
        if self._repo is None:
            raise ValueError("Repository not configured")

        topics = self._repo.get_all_topics_for_course(user_id, course_code)
        weak_areas = []

        for topic_input in topics:
            result = self.calculate_topic_mastery(topic_input)
            if result.mastery_score < threshold:
                gap = self.build_knowledge_gap(
                    topic_id=topic_input.topic_id,
                    topic_name=topic_input.topic_name,
                    mastery=result.mastery_score,
                    reviews=topic_input.exercise_reviews,
                    knowledge_item_names=None,  # Would need to fetch from repo
                )
                weak_areas.append(gap)

        # Sort by severity, then by mastery (lowest first)
        severity_order = {
            GapSeverity.CRITICAL: 0,
            GapSeverity.SIGNIFICANT: 1,
            GapSeverity.MINOR: 2,
        }
        weak_areas.sort(key=lambda x: (severity_order[x.gap_severity], x.current_mastery))

        return weak_areas
