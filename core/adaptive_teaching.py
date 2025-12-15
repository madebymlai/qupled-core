"""
Adaptive teaching system for Examina.
Personalizes explanations based on student's understanding level and learning progress.
"""

from typing import List, Dict, Optional, Any
from storage.database import Database


class AdaptiveTeachingManager:
    """Manages adaptive teaching features based on student progress and mastery."""

    def __init__(self, db: Optional[Database] = None):
        """Initialize adaptive teaching manager.

        Args:
            db: Database instance (creates new if not provided)
        """
        self.db = db
        self._external_db = db is not None

    def __enter__(self):
        """Context manager entry."""
        if not self._external_db:
            self.db = Database()
            self.db.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if not self._external_db and self.db:
            self.db.close()

    def get_recommended_depth(
        self,
        course_code: str,
        topic_name: Optional[str] = None,
        knowledge_item_name: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Determine optimal explanation depth based on mastery level.

        Args:
            course_code: Course code
            topic_name: Topic name (optional)
            knowledge_item_name: Core loop name (optional)
            user_id: User ID (reserved for future multi-user support)

        Returns:
            'basic', 'medium', or 'advanced'
        """
        mastery = self._calculate_mastery(course_code, topic_name, knowledge_item_name, user_id)

        # Mastery-to-Depth mapping:
        # - new/learning (mastery < 0.3): Use 'basic' depth
        # - reviewing (0.3 <= mastery < 0.7): Use 'medium' depth
        # - mastered (mastery >= 0.7): Use 'advanced' depth

        if mastery < 0.3:
            return "basic"
        elif mastery < 0.7:
            return "medium"
        else:
            return "advanced"

    def should_review_prerequisites(
        self, course_code: str, knowledge_item_name: str, user_id: Optional[str] = None
    ) -> bool:
        """Decide if prerequisite concepts should be explained.

        Args:
            course_code: Course code
            knowledge_item_name: Core loop name
            user_id: User ID (reserved for future multi-user support)

        Returns:
            True if prerequisites should be shown
        """
        mastery = self._calculate_mastery(
            course_code, knowledge_item_name=knowledge_item_name, user_id=user_id
        )

        # new/learning (mastery < 0.3): ALWAYS show prerequisites
        if mastery < 0.3:
            return True

        # reviewing (0.3 <= mastery < 0.7): Show prerequisites if recent failures
        if mastery < 0.7:
            return self._has_recent_failures(course_code, knowledge_item_name, user_id)

        # mastered (mastery >= 0.7): Skip prerequisites
        return False

    def detect_knowledge_gaps(
        self,
        course_code: str,
        knowledge_item_name: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict]:
        """Identify missing prerequisites or weak areas.

        Args:
            course_code: Course code
            knowledge_item_name: Core loop to check (optional, checks all if not provided)
            user_id: User ID (reserved for future multi-user support)

        Returns:
            List of knowledge gap dictionaries with:
            - gap: Topic/core loop name with low mastery
            - severity: 'high', 'medium', or 'low'
            - impact: List of dependent topics/core loops
            - recommendation: Suggested action
        """
        gaps = []

        with Database() as db:
            # Get all topics and their mastery
            topics = db.get_topics_by_course(course_code)

            for topic in topics:
                topic_id = topic["id"]
                topic_name = topic["name"]

                # Get core loops for this topic
                knowledge_items = db.get_knowledge_items_by_topic(topic_id)

                for loop in knowledge_items:
                    loop_id = loop["id"]
                    loop_name = loop["name"]

                    # Skip if specific core loop requested and this isn't it
                    if knowledge_item_name and loop_name != knowledge_item_name:
                        continue

                    # Get mastery for this core loop
                    mastery = self._calculate_mastery(
                        course_code, knowledge_item_name=loop_name, user_id=user_id
                    )

                    # Identify gaps (mastery < 0.5)
                    if mastery < 0.5:
                        # Determine severity
                        if mastery < 0.2:
                            severity = "high"
                        elif mastery < 0.35:
                            severity = "medium"
                        else:
                            severity = "low"

                        # Find dependent exercises/topics
                        impact = self._find_dependent_content(course_code, loop_id)

                        # Generate recommendation
                        if severity == "high":
                            recommendation = f"Review {loop_name} fundamentals before continuing with advanced topics"
                        elif severity == "medium":
                            recommendation = (
                                f"Practice {loop_name} exercises to strengthen understanding"
                            )
                        else:
                            recommendation = (
                                f"Consider reviewing {loop_name} if you encounter difficulties"
                            )

                        gaps.append(
                            {
                                "gap": loop_name,
                                "topic": topic_name,
                                "mastery": mastery,
                                "severity": severity,
                                "impact": impact,
                                "recommendation": recommendation,
                            }
                        )

        # Sort by severity and mastery (worst first)
        severity_order = {"high": 0, "medium": 1, "low": 2}
        gaps.sort(key=lambda x: (severity_order[x["severity"]], x["mastery"]))

        return gaps

    def check_prerequisite_mastery(
        self,
        course_code: str,
        knowledge_item_name: str,
        threshold: float = 0.5,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Check if prerequisites for a core loop are mastered.

        Args:
            course_code: Course code
            knowledge_item_name: Core loop to check prerequisites for
            threshold: Minimum mastery required (default 0.5)
            user_id: User ID (reserved for future multi-user support)

        Returns:
            Dict with:
            - ready: bool - True if all prerequisites are mastered
            - weak_prerequisites: List of weak prerequisite names
            - recommendation: str - What to do if not ready
        """
        weak_prerequisites = []

        with Database() as db:
            # Find prerequisite concepts for this core loop
            # First, get the core loop's topic
            knowledge_item = db.conn.execute(
                """
                SELECT cl.id, cl.topic_id, t.name as topic_name
                FROM knowledge_items cl
                JOIN topics t ON cl.topic_id = t.id
                WHERE cl.name = ? AND t.course_code = ?
            """,
                (knowledge_item_name, course_code),
            ).fetchone()

            if not knowledge_item:
                return {"ready": True, "weak_prerequisites": [], "recommendation": ""}

            topic_id = knowledge_item["topic_id"]

            # Get other core loops in the same topic that might be prerequisites
            # (simpler core loops that should be learned first)
            related_loops = db.conn.execute(
                """
                SELECT cl.id, cl.name
                FROM knowledge_items cl
                WHERE cl.topic_id = ? AND cl.name != ?
            """,
                (topic_id, knowledge_item_name),
            ).fetchall()

            # Check mastery of related core loops
            for loop in related_loops:
                loop_name = loop["name"]
                mastery = self._calculate_mastery(
                    course_code, knowledge_item_name=loop_name, user_id=user_id
                )

                if mastery < threshold:
                    weak_prerequisites.append(
                        {"name": loop_name, "mastery": mastery, "needed": threshold}
                    )

        if weak_prerequisites:
            names = [p["name"] for p in weak_prerequisites[:3]]
            recommendation = f"Consider reviewing first: {', '.join(names)}"
            return {
                "ready": False,
                "weak_prerequisites": weak_prerequisites,
                "recommendation": recommendation,
            }

        return {"ready": True, "weak_prerequisites": [], "recommendation": ""}

    def get_personalized_learning_path(
        self, course_code: str, user_id: Optional[str] = None, limit: int = 10
    ) -> List[Dict]:
        """Generate recommended study sequence based on current mastery.

        Priority:
        1. Overdue reviews (SM-2 scheduled reviews)
        2. Weak areas (low mastery topics)
        3. Due reviews
        4. New content (not yet attempted)

        Args:
            course_code: Course code
            user_id: User ID (reserved for future multi-user support)
            limit: Maximum number of items in path

        Returns:
            List of learning path items with:
            - priority: 1-N (lower is higher priority)
            - action: 'review', 'strengthen', 'learn', or 'practice'
            - knowledge_item: Core loop name (if applicable)
            - topic: Topic name
            - reason: Explanation why this is recommended
            - estimated_time: Estimated time in minutes
            - mastery: Current mastery level (0.0-1.0)
        """
        path = []
        priority = 1

        with Database() as db:
            # 1. OVERDUE REVIEWS (highest priority)
            overdue_reviews = db.conn.execute(
                """
                SELECT
                    sp.knowledge_item_id,
                    cl.name as knowledge_item_name,
                    t.name as topic_name,
                    sp.mastery_score,
                    sp.next_review,
                    sp.last_practiced,
                    JULIANDAY('now') - JULIANDAY(sp.next_review) as days_overdue
                FROM student_progress sp
                JOIN knowledge_items cl ON sp.knowledge_item_id = cl.id
                JOIN topics t ON cl.topic_id = t.id
                WHERE t.course_code = ?
                    AND sp.next_review IS NOT NULL
                    AND DATE(sp.next_review) < DATE('now')
                ORDER BY days_overdue DESC
                LIMIT ?
            """,
                (course_code, limit),
            ).fetchall()

            for review in overdue_reviews:
                days_overdue = int(review[6])
                path.append(
                    {
                        "priority": priority,
                        "action": "review",
                        "knowledge_item": review[1],
                        "topic": review[2],
                        "reason": f"Overdue by {days_overdue} day{'s' if days_overdue > 1 else ''}",
                        "estimated_time": 15,
                        "mastery": review[3],
                        "urgency": "high",
                    }
                )
                priority += 1

                if priority > limit:
                    return path

            # 2. WEAK AREAS (low mastery < 0.5)
            weak_areas = db.conn.execute(
                """
                SELECT
                    sp.knowledge_item_id,
                    cl.name as knowledge_item_name,
                    t.name as topic_name,
                    sp.mastery_score,
                    sp.total_attempts,
                    sp.last_practiced
                FROM student_progress sp
                JOIN knowledge_items cl ON sp.knowledge_item_id = cl.id
                JOIN topics t ON cl.topic_id = t.id
                WHERE t.course_code = ?
                    AND sp.mastery_score < 0.5
                    AND sp.total_attempts > 0
                ORDER BY sp.mastery_score ASC, sp.total_attempts ASC
                LIMIT ?
            """,
                (course_code, limit - priority + 1),
            ).fetchall()

            for weak in weak_areas:
                mastery_pct = int(weak[3] * 100)
                path.append(
                    {
                        "priority": priority,
                        "action": "strengthen",
                        "knowledge_item": weak[1],
                        "topic": weak[2],
                        "reason": f"Low mastery ({mastery_pct}%)",
                        "estimated_time": 20,
                        "mastery": weak[3],
                        "suggested_exercises": 3,
                        "urgency": "medium",
                    }
                )
                priority += 1

                if priority > limit:
                    return path

            # 3. DUE REVIEWS (scheduled for today)
            due_today = db.conn.execute(
                """
                SELECT
                    sp.knowledge_item_id,
                    cl.name as knowledge_item_name,
                    t.name as topic_name,
                    sp.mastery_score,
                    sp.last_practiced,
                    JULIANDAY('now') - JULIANDAY(sp.last_practiced) as days_since
                FROM student_progress sp
                JOIN knowledge_items cl ON sp.knowledge_item_id = cl.id
                JOIN topics t ON cl.topic_id = t.id
                WHERE t.course_code = ?
                    AND sp.next_review IS NOT NULL
                    AND DATE(sp.next_review) = DATE('now')
                ORDER BY sp.mastery_score ASC
                LIMIT ?
            """,
                (course_code, limit - priority + 1),
            ).fetchall()

            for due in due_today:
                days_since = int(due[5]) if due[5] else 0
                path.append(
                    {
                        "priority": priority,
                        "action": "review",
                        "knowledge_item": due[1],
                        "topic": due[2],
                        "reason": f"Due for review (last practiced {days_since} days ago)",
                        "estimated_time": 15,
                        "mastery": due[3],
                        "urgency": "medium",
                    }
                )
                priority += 1

                if priority > limit:
                    return path

            # 4. NEW CONTENT (not yet attempted)
            new_content = db.conn.execute(
                """
                SELECT
                    cl.id as knowledge_item_id,
                    cl.name as knowledge_item_name,
                    t.name as topic_name,
                    cl.exercise_count,
                    cl.difficulty_avg
                FROM knowledge_items cl
                JOIN topics t ON cl.topic_id = t.id
                LEFT JOIN student_progress sp ON cl.id = sp.knowledge_item_id AND sp.course_code = ?
                WHERE t.course_code = ?
                    AND (sp.total_attempts IS NULL OR sp.total_attempts = 0)
                    AND cl.exercise_count > 0
                ORDER BY cl.difficulty_avg ASC, cl.exercise_count DESC
                LIMIT ?
            """,
                (course_code, course_code, limit - priority + 1),
            ).fetchall()

            for new in new_content:
                difficulty = "easy" if new[4] < 1.5 else "medium" if new[4] < 2.5 else "hard"
                path.append(
                    {
                        "priority": priority,
                        "action": "learn",
                        "knowledge_item": new[1],
                        "topic": new[2],
                        "reason": f"New content ({new[3]} exercises available)",
                        "estimated_time": 25,
                        "mastery": 0.0,
                        "difficulty": difficulty,
                        "urgency": "low",
                    }
                )
                priority += 1

                if priority > limit:
                    return path

        return path

    def get_adaptive_recommendations(
        self, course_code: str, knowledge_item_name: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get adaptive recommendations for a specific core loop.

        Args:
            course_code: Course code
            knowledge_item_name: Core loop name
            user_id: User ID (reserved for future multi-user support)

        Returns:
            Dictionary with:
            - depth: Recommended explanation depth
            - show_prerequisites: Whether to show prerequisites
            - practice_count: Recommended number of practice exercises
            - focus_areas: List of specific areas to focus on
            - next_review: Next scheduled review date (if applicable)
        """
        mastery = self._calculate_mastery(
            course_code, knowledge_item_name=knowledge_item_name, user_id=user_id
        )
        depth = self.get_recommended_depth(
            course_code, knowledge_item_name=knowledge_item_name, user_id=user_id
        )
        show_prereqs = self.should_review_prerequisites(course_code, knowledge_item_name, user_id)

        # Calculate recommended practice count based on mastery
        if mastery < 0.3:
            practice_count = 5  # Intensive practice for beginners
        elif mastery < 0.7:
            practice_count = 3  # Moderate practice for intermediate
        else:
            practice_count = 1  # Light practice for advanced

        # Get focus areas (weak points)
        focus_areas = []
        gaps = self.detect_knowledge_gaps(course_code, knowledge_item_name, user_id)
        if gaps:
            focus_areas = [gap["recommendation"] for gap in gaps[:3]]

        # Get next review date
        next_review = None
        with Database() as db:
            # Find core loop ID
            loop_row = db.conn.execute(
                """
                SELECT cl.id
                FROM knowledge_items cl
                JOIN topics t ON cl.topic_id = t.id
                WHERE t.course_code = ? AND cl.name = ?
            """,
                (course_code, knowledge_item_name),
            ).fetchone()

            if loop_row:
                loop_id = loop_row[0]
                progress = db.conn.execute(
                    """
                    SELECT next_review
                    FROM student_progress
                    WHERE course_code = ? AND knowledge_item_id = ?
                """,
                    (course_code, loop_id),
                ).fetchone()

                if progress and progress[0]:
                    next_review = progress[0]

        return {
            "depth": depth,
            "show_prerequisites": show_prereqs,
            "practice_count": practice_count,
            "focus_areas": focus_areas,
            "next_review": next_review,
            "current_mastery": mastery,
        }

    def _calculate_mastery(
        self,
        course_code: str,
        topic_name: Optional[str] = None,
        knowledge_item_name: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> float:
        """Calculate mastery score for a topic or core loop.

        Uses existing student_progress data and quiz_attempts.

        Args:
            course_code: Course code
            topic_name: Topic name (optional)
            knowledge_item_name: Core loop name (optional)
            user_id: User ID (reserved for future)

        Returns:
            Mastery score between 0.0 and 1.0
        """
        with Database() as db:
            if knowledge_item_name:
                # Calculate mastery for specific core loop
                loop_row = db.conn.execute(
                    """
                    SELECT cl.id
                    FROM knowledge_items cl
                    JOIN topics t ON cl.topic_id = t.id
                    WHERE t.course_code = ? AND cl.name = ?
                """,
                    (course_code, knowledge_item_name),
                ).fetchone()

                if not loop_row:
                    return 0.0

                loop_id = loop_row[0]

                # Get student_progress data
                progress = db.conn.execute(
                    """
                    SELECT mastery_score, total_attempts, correct_attempts
                    FROM student_progress
                    WHERE course_code = ? AND knowledge_item_id = ?
                """,
                    (course_code, loop_id),
                ).fetchone()

                if not progress or progress[1] == 0:  # No attempts
                    return 0.0

                # Use existing mastery_score from student_progress
                return progress[0]

            elif topic_name:
                # Calculate average mastery for topic
                topic_row = db.conn.execute(
                    """
                    SELECT id FROM topics
                    WHERE course_code = ? AND name = ?
                """,
                    (course_code, topic_name),
                ).fetchone()

                if not topic_row:
                    return 0.0

                topic_id = topic_row[0]

                # Get average mastery across all core loops in topic
                avg_mastery = db.conn.execute(
                    """
                    SELECT AVG(sp.mastery_score)
                    FROM student_progress sp
                    JOIN knowledge_items cl ON sp.knowledge_item_id = cl.id
                    WHERE cl.topic_id = ? AND sp.total_attempts > 0
                """,
                    (topic_id,),
                ).fetchone()

                return avg_mastery[0] if avg_mastery[0] is not None else 0.0

            else:
                # Calculate overall course mastery
                avg_mastery = db.conn.execute(
                    """
                    SELECT AVG(sp.mastery_score)
                    FROM student_progress sp
                    JOIN knowledge_items cl ON sp.knowledge_item_id = cl.id
                    JOIN topics t ON cl.topic_id = t.id
                    WHERE t.course_code = ? AND sp.total_attempts > 0
                """,
                    (course_code,),
                ).fetchone()

                return avg_mastery[0] if avg_mastery[0] is not None else 0.0

    def _has_recent_failures(
        self, course_code: str, knowledge_item_name: str, user_id: Optional[str] = None
    ) -> bool:
        """Check if there have been recent failures for a core loop.

        Args:
            course_code: Course code
            knowledge_item_name: Core loop name
            user_id: User ID (reserved for future)

        Returns:
            True if recent failures detected
        """
        with Database() as db:
            # Find core loop ID
            loop_row = db.conn.execute(
                """
                SELECT cl.id
                FROM knowledge_items cl
                JOIN topics t ON cl.topic_id = t.id
                WHERE t.course_code = ? AND cl.name = ?
            """,
                (course_code, knowledge_item_name),
            ).fetchone()

            if not loop_row:
                return False

            loop_id = loop_row[0]

            # Check recent quiz attempts (last 5 attempts)
            recent_attempts = db.conn.execute(
                """
                SELECT qa.correct
                FROM quiz_attempts qa
                JOIN quiz_sessions qs ON qa.session_id = qs.id
                JOIN exercises e ON qa.exercise_id = e.id
                WHERE qs.course_code = ? AND e.knowledge_item_id = ?
                ORDER BY qa.attempted_at DESC
                LIMIT 5
            """,
                (course_code, loop_id),
            ).fetchall()

            if not recent_attempts:
                return False

            # If more than 40% of recent attempts were incorrect, show prerequisites
            incorrect_count = sum(1 for attempt in recent_attempts if not attempt[0])
            failure_rate = incorrect_count / len(recent_attempts)

            return failure_rate > 0.4

    def _find_dependent_content(self, course_code: str, knowledge_item_id: str) -> List[str]:
        """Find content that depends on a given core loop.

        Args:
            course_code: Course code
            knowledge_item_id: Core loop ID

        Returns:
            List of dependent topic/core loop names
        """
        # For now, return empty list
        # In future, could analyze prerequisite relationships
        # or find exercises that combine multiple core loops
        with Database() as db:
            # Find multi-procedure exercises that include this core loop
            dependent_exercises = db.conn.execute(
                """
                SELECT DISTINCT cl2.name
                FROM exercise_knowledge_items ecl1
                JOIN exercise_knowledge_items ecl2 ON ecl1.exercise_id = ecl2.exercise_id
                JOIN knowledge_items cl2 ON ecl2.knowledge_item_id = cl2.id
                JOIN exercises e ON ecl1.exercise_id = e.id
                WHERE ecl1.knowledge_item_id = ?
                    AND ecl2.knowledge_item_id != ?
                    AND e.course_code = ?
                LIMIT 5
            """,
                (knowledge_item_id, knowledge_item_id, course_code),
            ).fetchall()

            return [row[0] for row in dependent_exercises]

    def get_mastery_summary(
        self, course_code: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get summary of mastery levels across all topics.

        Args:
            course_code: Course code
            user_id: User ID (reserved for future)

        Returns:
            Dictionary with mastery breakdown
        """
        with Database() as db:
            topics = db.get_topics_by_course(course_code)
            summary = {
                "total_topics": len(topics),
                "mastered_topics": 0,
                "in_progress_topics": 0,
                "weak_topics": 0,
                "not_started_topics": 0,
                "overall_mastery": 0.0,
                "topic_details": [],
            }

            total_mastery = 0.0
            topics_with_progress = 0

            for topic in topics:
                mastery = self._calculate_mastery(
                    course_code, topic_name=topic["name"], user_id=user_id
                )

                # Classify topic
                if mastery >= 0.7:
                    status = "mastered"
                    summary["mastered_topics"] += 1
                elif mastery >= 0.3:
                    status = "in_progress"
                    summary["in_progress_topics"] += 1
                elif mastery > 0.0:
                    status = "weak"
                    summary["weak_topics"] += 1
                else:
                    status = "not_started"
                    summary["not_started_topics"] += 1

                if mastery > 0.0:
                    total_mastery += mastery
                    topics_with_progress += 1

                summary["topic_details"].append(
                    {"topic_name": topic["name"], "mastery": mastery, "status": status}
                )

            # Calculate overall mastery
            if topics_with_progress > 0:
                summary["overall_mastery"] = total_mastery / topics_with_progress

            # Sort topic details by mastery (ascending - weakest first)
            summary["topic_details"].sort(key=lambda x: x["mastery"])

            return summary
