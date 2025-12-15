"""
Mastery Aggregation Module for Examina.

Provides hierarchical mastery calculation and cascade updates:
    Exercise → Core Loop → Topic → Course

This module ensures mastery levels are consistent across the hierarchy
and automatically updates parent levels when child levels change.

Usage:
    from core.mastery_aggregator import MasteryAggregator
    from storage.database import Database

    with Database() as db:
        aggregator = MasteryAggregator(db)

        # Update after quiz attempt
        aggregator.update_exercise_mastery(exercise_id, quality, correct)

        # Cascade updates to parents
        aggregator.cascade_update(exercise_id)

        # Get aggregated mastery at any level
        topic_mastery = aggregator.get_topic_mastery(course_code, topic_id)
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from core.sm2 import SM2Algorithm


@dataclass
class MasteryStats:
    """Aggregated mastery statistics."""

    total_items: int
    mastered_count: int  # mastery_level = 'mastered'
    reviewing_count: int  # mastery_level = 'reviewing'
    learning_count: int  # mastery_level = 'learning'
    new_count: int  # mastery_level = 'new'

    average_mastery: float  # 0.0 - 1.0
    weighted_mastery: float  # weighted by importance/difficulty

    @property
    def mastery_distribution(self) -> Dict[str, int]:
        return {
            "new": self.new_count,
            "learning": self.learning_count,
            "reviewing": self.reviewing_count,
            "mastered": self.mastered_count,
        }

    @property
    def mastery_level(self) -> str:
        """Overall mastery level based on distribution."""
        if self.total_items == 0:
            return "new"

        mastered_ratio = self.mastered_count / self.total_items
        reviewing_ratio = self.reviewing_count / self.total_items

        if mastered_ratio >= 0.8:
            return "mastered"
        elif mastered_ratio + reviewing_ratio >= 0.6:
            return "reviewing"
        elif self.new_count < self.total_items * 0.5:
            return "learning"
        else:
            return "new"


class MasteryAggregator:
    """
    Aggregates mastery across the learning hierarchy.

    Hierarchy:
        Exercise (leaf) → Core Loop → Topic → Course (root)

    Each level's mastery is calculated from its children:
    - Core Loop mastery = aggregate of its exercise masteries
    - Topic mastery = aggregate of its core loop masteries
    - Course mastery = aggregate of its topic masteries
    """

    # Mastery level to numeric score mapping
    LEVEL_SCORES = {"new": 0.0, "learning": 0.33, "reviewing": 0.66, "mastered": 1.0}

    def __init__(self, db):
        """
        Initialize mastery aggregator.

        Args:
            db: Database instance (must be connected)
        """
        self.db = db
        self.sm2 = SM2Algorithm()

    # =========================================================================
    # Exercise Level (Leaf)
    # =========================================================================

    def get_exercise_mastery(self, exercise_id: str) -> Optional[Dict[str, Any]]:
        """
        Get mastery data for a single exercise.

        Args:
            exercise_id: Exercise ID

        Returns:
            Dict with mastery data or None if not found
        """
        row = self.db.conn.execute(
            """
            SELECT exercise_id, mastery_level, easiness_factor,
                   repetition_number, interval_days, total_reviews,
                   correct_reviews, last_reviewed_at
            FROM exercise_reviews
            WHERE exercise_id = ?
        """,
            (exercise_id,),
        ).fetchone()

        if not row:
            return None

        return {
            "exercise_id": row["exercise_id"],
            "mastery_level": row["mastery_level"],
            "mastery_score": self.LEVEL_SCORES.get(row["mastery_level"], 0.0),
            "easiness_factor": row["easiness_factor"],
            "repetition_number": row["repetition_number"],
            "interval_days": row["interval_days"],
            "total_reviews": row["total_reviews"],
            "correct_reviews": row["correct_reviews"],
            "correct_rate": row["correct_reviews"] / row["total_reviews"]
            if row["total_reviews"] > 0
            else 0.0,
            "last_reviewed_at": row["last_reviewed_at"],
        }

    def update_exercise_mastery(
        self, exercise_id: str, course_code: str, quality: int, correct: bool
    ) -> Dict[str, Any]:
        """
        Update exercise mastery after a quiz attempt using SM-2.

        Args:
            exercise_id: Exercise ID
            course_code: Course code
            quality: SM-2 quality score (0-5)
            correct: Whether the answer was correct

        Returns:
            Updated mastery data
        """
        # Get current state
        current = self.db.conn.execute(
            """
            SELECT easiness_factor, repetition_number, interval_days,
                   total_reviews, correct_reviews
            FROM exercise_reviews
            WHERE exercise_id = ?
        """,
            (exercise_id,),
        ).fetchone()

        if current:
            # Update existing record
            ef = current["easiness_factor"]
            rep = current["repetition_number"]
            interval = current["interval_days"]
            total = current["total_reviews"]
            correct_count = current["correct_reviews"]
        else:
            # New exercise - use defaults
            ef = 2.5
            rep = 0
            interval = 0
            total = 0
            correct_count = 0

        # Calculate new SM-2 values
        result = self.sm2.calculate_next_review(
            quality=quality, current_ef=ef, current_interval=interval, repetition=rep
        )

        # Update counts
        new_total = total + 1
        new_correct = correct_count + (1 if correct else 0)

        # Determine mastery level
        mastery_level = self.sm2.get_mastery_level(
            repetition=result["new_repetition"],
            interval_days=result["new_interval"],
            correct_count=new_correct,
            total_count=new_total,
        )

        # Upsert record
        self.db.conn.execute(
            """
            INSERT INTO exercise_reviews
                (exercise_id, course_code, easiness_factor, repetition_number,
                 interval_days, next_review_date, last_reviewed_at,
                 total_reviews, correct_reviews, mastery_level)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?)
            ON CONFLICT(exercise_id) DO UPDATE SET
                easiness_factor = excluded.easiness_factor,
                repetition_number = excluded.repetition_number,
                interval_days = excluded.interval_days,
                next_review_date = excluded.next_review_date,
                last_reviewed_at = CURRENT_TIMESTAMP,
                total_reviews = excluded.total_reviews,
                correct_reviews = excluded.correct_reviews,
                mastery_level = excluded.mastery_level
        """,
            (
                exercise_id,
                course_code,
                result["new_ef"],
                result["new_repetition"],
                result["new_interval"],
                result["next_review_date"].isoformat(),
                new_total,
                new_correct,
                mastery_level,
            ),
        )
        self.db.conn.commit()

        return {
            "exercise_id": exercise_id,
            "mastery_level": mastery_level,
            "mastery_score": self.LEVEL_SCORES[mastery_level],
            "easiness_factor": result["new_ef"],
            "repetition_number": result["new_repetition"],
            "interval_days": result["new_interval"],
            "next_review_date": result["next_review_date"],
            "total_reviews": new_total,
            "correct_reviews": new_correct,
        }

    # =========================================================================
    # Core Loop Level
    # =========================================================================

    def get_knowledge_item_mastery(self, knowledge_item_id: int) -> MasteryStats:
        """
        Calculate aggregated mastery for a core loop from its exercises.

        Args:
            knowledge_item_id: Core loop ID

        Returns:
            MasteryStats with aggregated data
        """
        # Get all exercises linked to this core loop
        rows = self.db.conn.execute(
            """
            SELECT er.mastery_level, er.easiness_factor,
                   er.correct_reviews, er.total_reviews
            FROM exercise_reviews er
            JOIN exercise_knowledge_items ecl ON er.exercise_id = ecl.exercise_id
            WHERE ecl.knowledge_item_id = ?
        """,
            (knowledge_item_id,),
        ).fetchall()

        if not rows:
            # No reviewed exercises - check how many exist
            count = self.db.conn.execute(
                """
                SELECT COUNT(*) FROM exercise_knowledge_items WHERE knowledge_item_id = ?
            """,
                (knowledge_item_id,),
            ).fetchone()[0]

            return MasteryStats(
                total_items=count,
                mastered_count=0,
                reviewing_count=0,
                learning_count=0,
                new_count=count,
                average_mastery=0.0,
                weighted_mastery=0.0,
            )

        # Count by mastery level
        counts = {"new": 0, "learning": 0, "reviewing": 0, "mastered": 0}
        total_score = 0.0

        for row in rows:
            level = row["mastery_level"] or "new"
            counts[level] = counts.get(level, 0) + 1
            total_score += self.LEVEL_SCORES.get(level, 0.0)

        total = len(rows)
        avg_mastery = total_score / total if total > 0 else 0.0

        return MasteryStats(
            total_items=total,
            mastered_count=counts["mastered"],
            reviewing_count=counts["reviewing"],
            learning_count=counts["learning"],
            new_count=counts["new"],
            average_mastery=avg_mastery,
            weighted_mastery=avg_mastery,  # TODO: weight by difficulty
        )

    def update_knowledge_item_mastery(self, knowledge_item_id: int, course_code: str) -> float:
        """
        Update student_progress table with aggregated core loop mastery.

        Args:
            knowledge_item_id: Core loop ID
            course_code: Course code

        Returns:
            New mastery score (0.0 - 1.0)
        """
        stats = self.get_knowledge_item_mastery(knowledge_item_id)

        # Get current attempts from student_progress
        current = self.db.conn.execute(
            """
            SELECT total_attempts, correct_attempts
            FROM student_progress
            WHERE knowledge_item_id = ? AND course_code = ?
        """,
            (knowledge_item_id, course_code),
        ).fetchone()

        total_attempts = current["total_attempts"] if current else 0
        correct_attempts = current["correct_attempts"] if current else 0

        # Calculate review interval based on mastery
        if stats.mastery_level == "mastered":
            interval = 30
        elif stats.mastery_level == "reviewing":
            interval = 7
        elif stats.mastery_level == "learning":
            interval = 3
        else:
            interval = 1

        # Upsert student_progress
        self.db.conn.execute(
            """
            INSERT INTO student_progress
                (course_code, knowledge_item_id, mastery_score, total_attempts,
                 correct_attempts, review_interval, last_practiced, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(course_code, knowledge_item_id) DO UPDATE SET
                mastery_score = excluded.mastery_score,
                review_interval = excluded.review_interval,
                last_practiced = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
        """,
            (
                course_code,
                knowledge_item_id,
                stats.average_mastery,
                total_attempts,
                correct_attempts,
                interval,
            ),
        )
        self.db.conn.commit()

        return stats.average_mastery

    # =========================================================================
    # Topic Level
    # =========================================================================

    def get_topic_mastery(self, topic_id: int) -> MasteryStats:
        """
        Calculate aggregated mastery for a topic from its core loops.

        Args:
            topic_id: Topic ID

        Returns:
            MasteryStats with aggregated data
        """
        # Get all core loops for this topic
        knowledge_items = self.db.conn.execute(
            """
            SELECT id FROM knowledge_items WHERE topic_id = ?
        """,
            (topic_id,),
        ).fetchall()

        if not knowledge_items:
            return MasteryStats(
                total_items=0,
                mastered_count=0,
                reviewing_count=0,
                learning_count=0,
                new_count=0,
                average_mastery=0.0,
                weighted_mastery=0.0,
            )

        # Aggregate core loop masteries
        counts = {"new": 0, "learning": 0, "reviewing": 0, "mastered": 0}
        total_score = 0.0

        for cl in knowledge_items:
            cl_stats = self.get_knowledge_item_mastery(cl["id"])
            level = cl_stats.mastery_level
            counts[level] += 1
            total_score += cl_stats.average_mastery

        total = len(knowledge_items)
        avg_mastery = total_score / total if total > 0 else 0.0

        return MasteryStats(
            total_items=total,
            mastered_count=counts["mastered"],
            reviewing_count=counts["reviewing"],
            learning_count=counts["learning"],
            new_count=counts["new"],
            average_mastery=avg_mastery,
            weighted_mastery=avg_mastery,
        )

    def update_topic_mastery(self, topic_id: int, course_code: str) -> float:
        """
        Update topic_mastery table with aggregated topic mastery.

        Args:
            topic_id: Topic ID
            course_code: Course code

        Returns:
            New mastery percentage (0.0 - 100.0)
        """
        stats = self.get_topic_mastery(topic_id)

        mastery_pct = stats.average_mastery * 100

        # Upsert topic_mastery
        self.db.conn.execute(
            """
            INSERT INTO topic_mastery
                (topic_id, course_code, exercises_total, exercises_mastered,
                 mastery_percentage, last_practiced_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(topic_id) DO UPDATE SET
                exercises_total = excluded.exercises_total,
                exercises_mastered = excluded.exercises_mastered,
                mastery_percentage = excluded.mastery_percentage,
                last_practiced_at = CURRENT_TIMESTAMP
        """,
            (topic_id, course_code, stats.total_items, stats.mastered_count, mastery_pct),
        )
        self.db.conn.commit()

        return mastery_pct

    # =========================================================================
    # Course Level
    # =========================================================================

    def get_course_mastery(self, course_code: str) -> MasteryStats:
        """
        Calculate aggregated mastery for a course from its topics.

        Args:
            course_code: Course code

        Returns:
            MasteryStats with aggregated data
        """
        # Get all topics for this course
        topics = self.db.conn.execute(
            """
            SELECT id FROM topics WHERE course_code = ?
        """,
            (course_code,),
        ).fetchall()

        if not topics:
            return MasteryStats(
                total_items=0,
                mastered_count=0,
                reviewing_count=0,
                learning_count=0,
                new_count=0,
                average_mastery=0.0,
                weighted_mastery=0.0,
            )

        # Aggregate topic masteries
        counts = {"new": 0, "learning": 0, "reviewing": 0, "mastered": 0}
        total_score = 0.0

        for topic in topics:
            t_stats = self.get_topic_mastery(topic["id"])
            level = t_stats.mastery_level
            counts[level] += 1
            total_score += t_stats.average_mastery

        total = len(topics)
        avg_mastery = total_score / total if total > 0 else 0.0

        return MasteryStats(
            total_items=total,
            mastered_count=counts["mastered"],
            reviewing_count=counts["reviewing"],
            learning_count=counts["learning"],
            new_count=counts["new"],
            average_mastery=avg_mastery,
            weighted_mastery=avg_mastery,
        )

    # =========================================================================
    # Cascade Updates
    # =========================================================================

    def cascade_update(self, exercise_id: str) -> Dict[str, Any]:
        """
        Cascade mastery update from exercise to core loop to topic.

        Call this after updating exercise mastery to propagate changes.

        Args:
            exercise_id: Exercise ID that was updated

        Returns:
            Dict with updated mastery at each level
        """
        # Get exercise's course and core loops
        exercise = self.db.conn.execute(
            """
            SELECT e.course_code, ecl.knowledge_item_id, cl.topic_id
            FROM exercises e
            JOIN exercise_knowledge_items ecl ON e.id = ecl.exercise_id
            JOIN knowledge_items cl ON ecl.knowledge_item_id = cl.id
            WHERE e.id = ?
        """,
            (exercise_id,),
        ).fetchone()

        if not exercise:
            return {"error": "Exercise not found or not linked to core loop"}

        course_code = exercise["course_code"]
        knowledge_item_id = exercise["knowledge_item_id"]
        topic_id = exercise["topic_id"]

        # Update core loop mastery
        cl_mastery = self.update_knowledge_item_mastery(knowledge_item_id, course_code)

        # Update topic mastery
        topic_mastery = self.update_topic_mastery(topic_id, course_code)

        # Get updated exercise mastery
        ex_mastery = self.get_exercise_mastery(exercise_id)

        return {
            "exercise": {
                "id": exercise_id,
                "mastery": ex_mastery["mastery_score"] if ex_mastery else 0.0,
                "level": ex_mastery["mastery_level"] if ex_mastery else "new",
            },
            "knowledge_item": {"id": knowledge_item_id, "mastery": cl_mastery},
            "topic": {"id": topic_id, "mastery_percentage": topic_mastery},
        }

    # =========================================================================
    # Bucket Queries (for mastery-based selection)
    # =========================================================================

    def get_exercises_by_mastery_level(
        self, course_code: str, level: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get exercises at a specific mastery level.

        Args:
            course_code: Course code
            level: Mastery level ('new', 'learning', 'reviewing', 'mastered')
            limit: Maximum number of exercises to return

        Returns:
            List of exercise dicts
        """
        if level == "new":
            # New = exercises not in exercise_reviews OR with mastery_level = 'new'
            rows = self.db.conn.execute(
                """
                SELECT e.id, e.text, e.difficulty
                FROM exercises e
                LEFT JOIN exercise_reviews er ON e.id = er.exercise_id
                WHERE e.course_code = ?
                  AND (er.exercise_id IS NULL OR er.mastery_level = 'new')
                LIMIT ?
            """,
                (course_code, limit),
            ).fetchall()
        else:
            rows = self.db.conn.execute(
                """
                SELECT e.id, e.text, e.difficulty, er.mastery_level,
                       er.easiness_factor, er.interval_days
                FROM exercises e
                JOIN exercise_reviews er ON e.id = er.exercise_id
                WHERE e.course_code = ? AND er.mastery_level = ?
                ORDER BY er.last_reviewed_at ASC
                LIMIT ?
            """,
                (course_code, level, limit),
            ).fetchall()

        return [dict(row) for row in rows]

    def get_mastery_distribution(self, course_code: str) -> Dict[str, int]:
        """
        Get distribution of exercises across mastery levels.

        Args:
            course_code: Course code

        Returns:
            Dict with counts per mastery level
        """
        # Count exercises with reviews
        reviewed = self.db.conn.execute(
            """
            SELECT mastery_level, COUNT(*) as count
            FROM exercise_reviews
            WHERE course_code = ?
            GROUP BY mastery_level
        """,
            (course_code,),
        ).fetchall()

        # Count total exercises
        total = self.db.conn.execute(
            """
            SELECT COUNT(*) FROM exercises WHERE course_code = ?
        """,
            (course_code,),
        ).fetchone()[0]

        # Build distribution
        dist = {"new": 0, "learning": 0, "reviewing": 0, "mastered": 0}
        reviewed_count = 0

        for row in reviewed:
            level = row["mastery_level"] or "new"
            dist[level] = row["count"]
            reviewed_count += row["count"]

        # Exercises not in exercise_reviews are 'new'
        dist["new"] += total - reviewed_count

        return dist

    def get_weak_knowledge_items(
        self, course_code: str, threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Get core loops with mastery below threshold.

        Args:
            course_code: Course code
            threshold: Mastery threshold (default 0.5)

        Returns:
            List of weak core loops with their mastery scores
        """
        rows = self.db.conn.execute(
            """
            SELECT sp.knowledge_item_id, sp.mastery_score, cl.name as knowledge_item_name,
                   t.name as topic_name
            FROM student_progress sp
            JOIN knowledge_items cl ON sp.knowledge_item_id = cl.id
            JOIN topics t ON cl.topic_id = t.id
            WHERE sp.course_code = ? AND sp.mastery_score < ?
            ORDER BY sp.mastery_score ASC
        """,
            (course_code, threshold),
        ).fetchall()

        return [dict(row) for row in rows]

    def get_exercises_due_for_review(
        self, course_code: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get exercises that are due for review (next_review_date <= today).

        Args:
            course_code: Course code
            limit: Maximum exercises to return

        Returns:
            List of exercises due for review
        """
        rows = self.db.conn.execute(
            """
            SELECT e.id, e.text, e.difficulty, er.mastery_level,
                   er.next_review_date, er.interval_days
            FROM exercises e
            JOIN exercise_reviews er ON e.id = er.exercise_id
            WHERE e.course_code = ?
              AND date(er.next_review_date) <= date('now')
            ORDER BY er.next_review_date ASC
            LIMIT ?
        """,
            (course_code, limit),
        ).fetchall()

        return [dict(row) for row in rows]
