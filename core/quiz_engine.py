"""
Quiz engine for Examina.
Manages quiz sessions, question selection, and spaced repetition.
"""

import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from config import Config
from core.sm2 import SM2Algorithm
from core.tutor import Tutor
from models.llm_manager import LLMManager



@dataclass
class QuizQuestion:
    """Represents a quiz question."""

    question_number: int
    exercise_id: str
    exercise_text: str
    knowledge_item_id: str
    knowledge_item_name: str
    difficulty: str
    knowledge_items: Optional[List[Dict[str, Any]]] = None
    user_answer: Optional[str] = None
    is_correct: Optional[bool] = None
    score: Optional[float] = None
    feedback: Optional[str] = None
    hints_requested: int = 0
    time_spent: Optional[int] = None  # seconds


@dataclass
class QuizSession:
    """Represents a quiz session."""

    session_id: str
    course_code: str
    quiz_type: str  # 'random', 'knowledge_item', 'review'
    questions: List[QuizQuestion]
    total_questions: int
    current_question: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_correct: int = 0
    score: float = 0.0
    knowledge_item_id: Optional[str] = None


class QuizEngine:
    """Manages quiz sessions and spaced repetition."""

    def __init__(self, llm_manager: Optional[LLMManager] = None, language: str = "en"):
        """Initialize quiz engine.

        Args:
            llm_manager: LLM manager for AI feedback
            language: Language for feedback ("en" or "it")
        """
        self.llm = llm_manager or LLMManager(provider="anthropic")
        self.tutor = Tutor(llm_manager=self.llm, language=language)
        self.language = language

    def create_quiz_session(
        self,
        course_code: str,
        num_questions: int = 10,
        topic: Optional[str] = None,
        knowledge_item: Optional[str] = None,
        difficulty: Optional[str] = None,
        review_only: bool = False,
        procedure_type: Optional[str] = None,
        multi_only: bool = False,
        tags: Optional[str] = None,
        adaptive: bool = False,
    ) -> QuizSession:
        """Create a new quiz session.

        Args:
            course_code: Course code
            num_questions: Number of questions
            topic: Optional topic filter
            knowledge_item: Optional core loop filter (ID or name pattern)
            difficulty: Optional difficulty filter
            review_only: If True, only include exercises due for review
            procedure_type: Optional procedure type (transformation, design, etc.)
            multi_only: If True, only include exercises with multiple procedures
            tags: Optional tag filter (comma-separated)
            adaptive: If True, select exercises based on mastery distribution
                      (40% weak, 40% learning, 20% strong)

        Returns:
            QuizSession object
        """
        session_id = str(uuid.uuid4())

        # Determine quiz type
        if adaptive:
            quiz_type = "adaptive"
        elif review_only:
            quiz_type = "review"
        elif multi_only:
            quiz_type = "multi_procedure"
        elif procedure_type:
            quiz_type = "procedure"
        elif knowledge_item:
            quiz_type = "knowledge_item"
        elif topic:
            quiz_type = "topic"
        else:
            quiz_type = "random"

        # Select exercises
        if adaptive:
            # Use mastery-based selection (40% weak, 40% learning, 20% strong)
            exercises = self._select_exercises_adaptive(
                course_code=course_code,
                num_questions=num_questions,
                topic=topic,
                knowledge_item=knowledge_item,
            )
        else:
            exercises = self._select_exercises(
                course_code=course_code,
                num_questions=num_questions,
                topic=topic,
                knowledge_item=knowledge_item,
                difficulty=difficulty,
                review_only=review_only,
                procedure_type=procedure_type,
                multi_only=multi_only,
                tags=tags,
            )

        if not exercises:
            raise ValueError("No exercises found matching the criteria")

        # Create quiz questions
        questions = []
        from storage.database import Database
        with Database() as db:
            for i, exercise in enumerate(exercises, 1):
                # Fetch all core loops for this exercise
                knowledge_items = db.get_knowledge_items_for_exercise(exercise["id"])

                # Use first core loop for backward compatibility
                primary_loop_id = exercise.get("knowledge_item_id", "")
                primary_loop_name = exercise.get("knowledge_item_name", "Unknown")

                # If we have core loops from junction table, use the first one as primary
                if knowledge_items:
                    primary_loop_id = knowledge_items[0].get("id", primary_loop_id)
                    primary_loop_name = knowledge_items[0].get("name", primary_loop_name)

                questions.append(
                    QuizQuestion(
                        question_number=i,
                        exercise_id=exercise["id"],
                        exercise_text=exercise["text"],
                        knowledge_item_id=primary_loop_id,
                        knowledge_item_name=primary_loop_name,
                        topic_name=exercise.get("topic_name", "Unknown"),
                        difficulty=exercise.get("difficulty", "medium"),
                        knowledge_items=knowledge_items if knowledge_items else None,
                    )
                )

        # Get topic_id and knowledge_item_id for session metadata
        topic_id = None
        knowledge_item_id = None

        if topic and exercises:
            from storage.database import Database
            with Database() as db:
                topic_row = db.conn.execute(
                    "SELECT id FROM topics WHERE course_code = ? AND name LIKE ?",
                    (course_code, f"%{topic}%"),
                ).fetchone()
                if topic_row:
                    topic_id = topic_row[0]

        if knowledge_item and exercises:
            knowledge_item_id = exercises[0].get("knowledge_item_id")

        return QuizSession(
            session_id=session_id,
            course_code=course_code,
            quiz_type=quiz_type,
            questions=questions,
            total_questions=len(questions),
            topic_id=topic_id,
            knowledge_item_id=knowledge_item_id,
            started_at=datetime.now(),
        )

    def _select_exercises(
        self,
        course_code: str,
        num_questions: int,
        topic: Optional[str],
        knowledge_item: Optional[str],
        difficulty: Optional[str],
        review_only: bool,
        procedure_type: Optional[str] = None,
        multi_only: bool = False,
        tags: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Select exercises for the quiz.

        Args:
            course_code: Course code
            num_questions: Number of questions needed
            topic: Optional topic filter
            knowledge_item: Optional core loop filter (ID or name pattern)
            difficulty: Optional difficulty filter
            review_only: Only exercises due for review
            procedure_type: Optional procedure type filter (transformation, design, etc.)
            multi_only: Only exercises with multiple procedures
            tags: Optional tag filter (comma-separated)

        Returns:
            List of exercise dictionaries
        """
        from storage.database import Database
        with Database() as db:
            # Build query using junction table for multi-procedure support
            query = """
                SELECT DISTINCT
                    e.id,
                    e.text,
                    e.difficulty,
                    e.knowledge_item_id,
                    e.tags,
                    t.name as topic_name,
                    sp.next_review,
                    sp.mastery_score,
                    GROUP_CONCAT(DISTINCT cl.name) as knowledge_item_names
                FROM exercises e
                LEFT JOIN topics t ON e.topic_id = t.id
                LEFT JOIN exercise_knowledge_items ecl ON e.id = ecl.exercise_id
                LEFT JOIN knowledge_items cl ON ecl.knowledge_item_id = cl.id
                LEFT JOIN student_progress sp ON e.knowledge_item_id = sp.knowledge_item_id AND sp.course_code = ?
                WHERE e.course_code = ?
                    AND e.analyzed = 1
                    AND e.low_confidence_skipped = 0
            """
            params = [course_code, course_code]

            # Apply filters
            if topic:
                query += " AND t.name LIKE ?"
                params.append(f"%{topic}%")

            if knowledge_item:
                # Support both ID and name pattern matching via junction table
                query += """ AND e.id IN (
                    SELECT exercise_id FROM exercise_knowledge_items ecl2
                    JOIN knowledge_items cl2 ON ecl2.knowledge_item_id = cl2.id
                    WHERE cl2.id = ? OR cl2.name LIKE ?
                )"""
                params.append(knowledge_item)
                params.append(f"%{knowledge_item}%")

            if difficulty:
                query += " AND e.difficulty = ?"
                params.append(difficulty)

            if procedure_type:
                # Filter by procedure type using tags
                query += " AND e.tags LIKE ?"
                params.append(f'%"{procedure_type}"%')

            if tags:
                # Filter by any of the provided tags (comma-separated)
                tag_list = [t.strip() for t in tags.split(",")]
                tag_conditions = " OR ".join(["e.tags LIKE ?" for _ in tag_list])
                query += f" AND ({tag_conditions})"
                for tag in tag_list:
                    params.append(f'%"{tag}"%')

            if review_only:
                query += " AND sp.next_review IS NOT NULL AND DATE(sp.next_review) <= DATE('now')"

            # Group by exercise to aggregate core loop names
            query += " GROUP BY e.id"

            # Filter by multi-procedure exercises (HAVING clause after GROUP BY)
            if multi_only:
                query += " HAVING COUNT(DISTINCT ecl.knowledge_item_id) > 1"

            # Execute query
            results = db.conn.execute(query, params).fetchall()

            if not results:
                return []

            # Convert to dictionaries
            exercises = [dict(row) for row in results]

            # Prioritize review exercises if available
            if review_only or (len(exercises) > num_questions):
                # Sort by review priority (overdue first, then by mastery score)
                def review_priority(ex):
                    next_review = ex.get("next_review")
                    mastery = ex.get("mastery_score", 0)

                    if review_only and next_review:
                        # Overdue exercises first
                        review_date = datetime.fromisoformat(next_review)
                        days_overdue = (datetime.now() - review_date).days
                        return (-days_overdue, mastery)  # More overdue = higher priority

                    # New or low mastery exercises
                    return (0, mastery)  # Lower mastery = higher priority

                exercises.sort(key=review_priority)

            # Limit to requested number
            selected = exercises[:num_questions]

            # Shuffle if not review mode (keep review order for overdue items)
            if not review_only:
                random.shuffle(selected)

            return selected

    def _select_exercises_adaptive(
        self,
        course_code: str,
        num_questions: int,
        topic: Optional[str] = None,
        knowledge_item: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Select exercises based on mastery distribution.

        Distribution:
        - 40% weak (mastery < 0.4) - reinforce gaps
        - 40% learning (0.4-0.7) - build proficiency
        - 20% strong (> 0.7) - maintain mastery

        Args:
            course_code: Course code
            num_questions: Number of questions needed
            topic: Optional topic filter
            knowledge_item: Optional core loop filter

        Returns:
            List of exercise dictionaries balanced by mastery
        """
        from core.mastery_aggregator import MasteryAggregator

        # Calculate distribution
        n_weak = max(1, int(num_questions * 0.4))
        n_learning = max(1, int(num_questions * 0.4))
        n_strong = num_questions - n_weak - n_learning

        selected = []

        from storage.database import Database
        from core.mastery_aggregator import MasteryAggregator
        with Database() as db:
            MasteryAggregator(db)

            # Base query for exercises with mastery info
            base_query = """
                SELECT DISTINCT
                    e.id, e.text, e.difficulty, e.knowledge_item_id,
                    t.name as topic_name,
                    COALESCE(er.mastery_level, 'new') as mastery_level,
                    CASE
                        WHEN er.mastery_level = 'mastered' THEN 1.0
                        WHEN er.mastery_level = 'reviewing' THEN 0.66
                        WHEN er.mastery_level = 'learning' THEN 0.33
                        ELSE 0.0
                    END as mastery_score
                FROM exercises e
                LEFT JOIN topics t ON e.topic_id = t.id
                LEFT JOIN exercise_reviews er ON e.id = er.exercise_id
                WHERE e.course_code = ?
                    AND e.analyzed = 1
                    AND e.low_confidence_skipped = 0
            """
            params = [course_code]

            if topic:
                base_query += " AND t.name LIKE ?"
                params.append(f"%{topic}%")

            if knowledge_item:
                base_query += """ AND e.id IN (
                    SELECT exercise_id FROM exercise_knowledge_items ecl
                    JOIN knowledge_items cl ON ecl.knowledge_item_id = cl.id
                    WHERE cl.id = ? OR cl.name LIKE ?
                )"""
                params.append(knowledge_item)
                params.append(f"%{knowledge_item}%")

            # Get weak exercises (new or learning with low score)
            weak_query = (
                base_query
                + """
                AND (er.mastery_level IS NULL
                     OR er.mastery_level = 'new'
                     OR er.mastery_level = 'learning')
                ORDER BY RANDOM()
                LIMIT ?
            """
            )
            weak_params = params + [n_weak]
            weak_results = db.conn.execute(weak_query, weak_params).fetchall()
            selected.extend([dict(r) for r in weak_results])

            # Get learning exercises (reviewing, working on it)
            learning_query = (
                base_query
                + """
                AND er.mastery_level = 'reviewing'
                ORDER BY RANDOM()
                LIMIT ?
            """
            )
            learning_params = params + [n_learning]
            learning_results = db.conn.execute(learning_query, learning_params).fetchall()
            selected.extend([dict(r) for r in learning_results])

            # Get strong exercises (mastered, for maintenance)
            strong_query = (
                base_query
                + """
                AND er.mastery_level = 'mastered'
                ORDER BY RANDOM()
                LIMIT ?
            """
            )
            strong_params = params + [n_strong]
            strong_results = db.conn.execute(strong_query, strong_params).fetchall()
            selected.extend([dict(r) for r in strong_results])

            # If we don't have enough in each category, fill from any available
            if len(selected) < num_questions:
                existing_ids = {ex["id"] for ex in selected}
                fill_query = (
                    base_query
                    + """
                    ORDER BY RANDOM()
                    LIMIT ?
                """
                )
                fill_params = params + [num_questions * 2]
                fill_results = db.conn.execute(fill_query, fill_params).fetchall()

                for row in fill_results:
                    if row["id"] not in existing_ids:
                        selected.append(dict(row))
                        existing_ids.add(row["id"])
                        if len(selected) >= num_questions:
                            break

            # Shuffle final selection
            random.shuffle(selected)

            return selected[:num_questions]

    def evaluate_answer(
        self,
        session: QuizSession,
        question: QuizQuestion,
        user_answer: str,
        provide_hints: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate a user's answer using AI.

        Args:
            session: Quiz session
            question: Current question
            user_answer: User's answer
            provide_hints: Whether to provide hints

        Returns:
            Dictionary with:
            - is_correct: Boolean
            - score: Float (0-1)
            - feedback: String with AI feedback
            - mistakes: List of identified mistakes
        """
        # Use tutor to check answer
        feedback_response = self.tutor.check_answer(
            exercise_id=question.exercise_id, user_answer=user_answer, provide_hints=provide_hints
        )

        if not feedback_response.success:
            return {
                "is_correct": False,
                "score": 0.0,
                "feedback": "Error evaluating answer: " + feedback_response.content,
                "mistakes": [],
            }

        # Parse feedback to determine correctness
        # Look for keywords in the feedback
        feedback_text = feedback_response.content.lower()

        # Simple heuristic: look for positive/negative indicators
        positive_indicators = ["correct", "right", "excellent", "perfect", "well done", "great"]
        negative_indicators = ["incorrect", "wrong", "mistake", "error", "not quite", "missing"]

        positive_count = sum(1 for word in positive_indicators if word in feedback_text)
        negative_count = sum(1 for word in negative_indicators if word in feedback_text)

        # Determine if correct (basic heuristic)
        is_correct = positive_count > negative_count

        # Score based on correctness and hints
        if is_correct:
            score = 1.0 - (question.hints_requested * Config.HINT_PENALTY)
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
        else:
            # Partial credit for partially correct answers
            if positive_count > 0:
                score = 0.5
            else:
                score = 0.0

        return {
            "is_correct": is_correct,
            "score": score,
            "feedback": feedback_response.content,
            "mistakes": [],  # Could extract from feedback in future
        }

    def complete_session(self, session: QuizSession) -> None:
        """Complete a quiz session and update progress.

        Args:
            session: Quiz session to complete
        """
        session.completed_at = datetime.now()

        # Calculate total score
        answered_questions = [q for q in session.questions if q.score is not None]
        if answered_questions:
            session.total_correct = sum(1 for q in answered_questions if q.is_correct)
            session.score = sum(q.score for q in answered_questions) / len(answered_questions)

        # Calculate total time
        if session.started_at and session.completed_at:
            int((session.completed_at - session.started_at).total_seconds())

        # Store session in database
        from storage.database import Database
        with Database() as db:
            db.conn.execute(
                """
                INSERT INTO quiz_sessions
                (id, course_code, quiz_type, filter_topic_id, filter_knowledge_item_id,
                 total_questions, created_at, completed_at, correct_answers,
                 score_percentage, filter_difficulty)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    session.session_id,
                    session.course_code,
                    session.quiz_type,
                    session.topic_id,
                    session.knowledge_item_id,
                    session.total_questions,
                    session.started_at.isoformat(),
                    session.completed_at.isoformat(),
                    session.total_correct,
                    session.score,
                    None,  # filter_difficulty
                ),
            )

            # Store individual answers
            for question in session.questions:
                if question.user_answer is not None:
                    db.conn.execute(
                        """
                        INSERT INTO quiz_attempts
                        (session_id, exercise_id, user_answer,
                         correct, time_taken_seconds, hint_used, feedback)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            session.session_id,
                            question.exercise_id,
                            question.user_answer,
                            1 if question.is_correct else 0,
                            int(question.time_spent) if question.time_spent else 0,
                            1 if question.hints_requested else 0,
                            question.feedback or "",
                        ),
                    )

            # Update student progress for each core loop
            self._update_progress(session, db)

            db.conn.commit()

    def _update_progress(self, session: QuizSession, db: Any) -> None:
        """Update student progress based on quiz results.

        Uses spaced repetition (SM-2 algorithm) to schedule reviews.
        Now includes exercise-level mastery updates with cascade propagation.

        Args:
            session: Completed quiz session
            db: Database connection
        """
        # Initialize mastery aggregator for cascade updates
        from core.mastery_aggregator import MasteryAggregator
        aggregator = MasteryAggregator(db)
        sm2 = SM2Algorithm()

        # Track exercises updated for cascade
        updated_exercises = set()

        # First: Update exercise-level mastery for each answered question
        for question in session.questions:
            if question.score is None:
                continue

            # Calculate SM-2 quality from score and performance
            quality = sm2.get_review_quality_from_score(
                correct=question.is_correct or False,
                time_taken=question.time_spent or 180,
                hint_used=question.hints_requested > 0,
                expected_time=180,  # 3 minutes expected
            )

            # Update exercise mastery
            aggregator.update_exercise_mastery(
                exercise_id=question.exercise_id,
                course_code=session.course_code,
                quality=quality,
                correct=question.is_correct or False,
            )
            updated_exercises.add(question.exercise_id)

        # Cascade updates for all affected exercises
        for exercise_id in updated_exercises:
            aggregator.cascade_update(exercise_id)

        # Group questions by core loop (for backward compatibility)
        knowledge_item_performance = {}
        for question in session.questions:
            if question.score is None:
                continue

            loop_id = question.knowledge_item_id
            if not loop_id:  # Skip if no core loop assigned
                continue
            if loop_id not in knowledge_item_performance:
                knowledge_item_performance[loop_id] = {
                    "attempts": 0,
                    "correct": 0,
                    "total_score": 0.0,
                }

            knowledge_item_performance[loop_id]["attempts"] += 1
            if question.is_correct:
                knowledge_item_performance[loop_id]["correct"] += 1
            knowledge_item_performance[loop_id]["total_score"] += question.score

        # Update progress for each core loop
        for loop_id, perf in knowledge_item_performance.items():
            # Get existing progress
            existing = db.conn.execute(
                """
                SELECT * FROM student_progress
                WHERE course_code = ? AND knowledge_item_id = ?
            """,
                (session.course_code, loop_id),
            ).fetchone()

            if existing:
                # Update existing progress
                total_attempts = existing["total_attempts"] + perf["attempts"]
                correct_attempts = existing["correct_attempts"] + perf["correct"]

                # Calculate new mastery score (exponential moving average)
                old_mastery = existing["mastery_score"]
                new_accuracy = perf["correct"] / perf["attempts"]
                alpha = 0.3  # Weight for new data
                mastery_score = old_mastery * (1 - alpha) + new_accuracy * alpha

                # SM-2 algorithm for next review
                quality = perf["total_score"] / perf["attempts"]  # 0-1 score
                old_interval = existing["review_interval"]
                new_interval = self._calculate_sm2_interval(quality, old_interval)

                next_review = datetime.now() + timedelta(days=new_interval)

                db.conn.execute(
                    """
                    UPDATE student_progress
                    SET total_attempts = ?,
                        correct_attempts = ?,
                        mastery_score = ?,
                        last_practiced = CURRENT_TIMESTAMP,
                        next_review = ?,
                        review_interval = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE course_code = ? AND knowledge_item_id = ?
                """,
                    (
                        total_attempts,
                        correct_attempts,
                        mastery_score,
                        next_review.isoformat(),
                        new_interval,
                        session.course_code,
                        loop_id,
                    ),
                )
            else:
                # Create new progress entry
                total_attempts = perf["attempts"]
                correct_attempts = perf["correct"]
                mastery_score = perf["correct"] / perf["attempts"]

                # Initial review interval
                quality = perf["total_score"] / perf["attempts"]
                new_interval = self._calculate_sm2_interval(quality, 1)
                next_review = datetime.now() + timedelta(days=new_interval)

                db.conn.execute(
                    """
                    INSERT INTO student_progress
                    (course_code, knowledge_item_id, total_attempts, correct_attempts,
                     mastery_score, last_practiced, next_review, review_interval)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
                """,
                    (
                        session.course_code,
                        loop_id,
                        total_attempts,
                        correct_attempts,
                        mastery_score,
                        next_review.isoformat(),
                        new_interval,
                    ),
                )

    def _calculate_sm2_interval(self, quality: float, old_interval: int) -> int:
        """Calculate next review interval using SM-2 algorithm.

        Args:
            quality: Quality of recall (0-1)
            old_interval: Previous interval in days

        Returns:
            New interval in days
        """
        # Convert quality (0-1) to SM-2 grade (0-5)
        grade = int(quality * 5)

        if grade < 3:
            # Failed recall - reset to 1 day
            return 1
        else:
            # Successful recall - increase interval
            if old_interval == 1:
                return 3
            elif old_interval == 3:
                return 7
            else:
                # Exponential growth
                easiness = 1.3 + (grade - 3) * 0.1  # 1.3 to 1.6
                new_interval = int(old_interval * easiness)
                return min(new_interval, 180)  # Cap at 6 months
