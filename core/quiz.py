"""
Quiz Manager for Examina's Phase 5 quiz system.
Handles quiz creation, question selection, answer evaluation, and SM-2 integration.
"""

import uuid
import json
import random
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from storage.database import Database
from core.sm2 import SM2Algorithm
from core.tutor import Tutor
from models.llm_manager import LLMManager
from config import Config


@dataclass
class QuizQuestion:
    """Represents a quiz question."""
    exercise_id: str
    question_number: int
    text: str
    difficulty: Optional[str]
    knowledge_item_id: Optional[str]
    topic_id: Optional[int]
    answered: bool = False


class QuizManager:
    """Manages quiz sessions, question selection, and answer evaluation."""

    def __init__(self, llm_manager: Optional[LLMManager] = None, language: str = "en"):
        """Initialize QuizManager.

        Args:
            llm_manager: LLM manager instance
            language: Language for AI feedback ("en" or "it")
        """
        self.llm = llm_manager or LLMManager(provider="anthropic")
        self.language = language
        self.tutor = Tutor(llm_manager=self.llm, language=language)
        self.sm2 = SM2Algorithm()

    def create_quiz(self, course_code: str, quiz_type: str = 'random',
                   question_count: int = 10, topic_id: Optional[int] = None,
                   knowledge_item_id: Optional[str] = None,
                   difficulty: Optional[str] = None,
                   prioritize_due: bool = True) -> str:
        """Create a new quiz session.

        Args:
            course_code: Course code
            quiz_type: Type of quiz ('random', 'topic', 'knowledge_item', 'review')
            question_count: Number of questions
            topic_id: Filter by topic (for 'topic' quiz type)
            knowledge_item_id: Filter by core loop (for 'knowledge_item' quiz type)
            difficulty: Filter by difficulty ('easy', 'medium', 'hard')
            prioritize_due: Prioritize exercises due for review

        Returns:
            session_id: Unique quiz session ID

        Raises:
            ValueError: If no exercises match the filters
        """
        # Generate unique session ID
        session_id = str(uuid.uuid4())

        with Database() as db:
            # Verify course exists
            course = db.get_course(course_code)
            if not course:
                raise ValueError(f"Course {course_code} not found")

            # Get exercises based on quiz type and filters
            exercises = self._select_exercises(
                db, course_code, quiz_type, topic_id, knowledge_item_id,
                difficulty, prioritize_due, question_count
            )

            if not exercises:
                raise ValueError(
                    f"No exercises found matching criteria for quiz type '{quiz_type}'"
                )

            # Limit to requested count
            if len(exercises) > question_count:
                exercises = exercises[:question_count]

            # Create quiz session in database
            db.conn.execute("""
                INSERT INTO quiz_sessions
                (id, course_code, quiz_type, topic_id, knowledge_item_id,
                 total_questions, started_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (session_id, course_code, quiz_type, topic_id, knowledge_item_id,
                  len(exercises)))

            # Store questions in quiz_answers table (initially unanswered)
            for idx, exercise in enumerate(exercises, 1):
                db.conn.execute("""
                    INSERT INTO quiz_answers
                    (session_id, exercise_id, question_number, answered_at)
                    VALUES (?, ?, ?, NULL)
                """, (session_id, exercise['id'], idx))

        return session_id

    def _select_exercises(self, db: Database, course_code: str,
                         quiz_type: str, topic_id: Optional[int],
                         knowledge_item_id: Optional[str], difficulty: Optional[str],
                         prioritize_due: bool, question_count: int) -> List[Dict[str, Any]]:
        """Select exercises for the quiz based on filters.

        Args:
            db: Database connection
            course_code: Course code
            quiz_type: Type of quiz
            topic_id: Topic filter
            knowledge_item_id: Core loop filter
            difficulty: Difficulty filter
            prioritize_due: Whether to prioritize due exercises
            question_count: Number of questions needed

        Returns:
            List of exercise dictionaries
        """
        # Build query based on quiz type
        query = "SELECT DISTINCT e.* FROM exercises e"
        params = [course_code]
        conditions = ["e.course_code = ?"]

        # Add join for SM-2 progress if prioritizing due exercises
        if prioritize_due and quiz_type in ['random', 'review']:
            query += " LEFT JOIN student_progress sp ON e.knowledge_item_id = sp.knowledge_item_id"

        # Filter by quiz type
        if quiz_type == 'topic' and topic_id:
            conditions.append("e.topic_id = ?")
            params.append(topic_id)
        elif quiz_type == 'knowledge_item' and knowledge_item_id:
            conditions.append("e.knowledge_item_id = ?")
            params.append(knowledge_item_id)
        elif quiz_type == 'review':
            # Only exercises with core loops that are due for review
            conditions.append("e.knowledge_item_id IS NOT NULL")
            conditions.append(
                "(sp.next_review IS NULL OR sp.next_review <= datetime('now'))"
            )

        # Additional filters
        if difficulty:
            conditions.append("e.difficulty = ?")
            params.append(difficulty)

        # Only include analyzed exercises with core loops
        conditions.append("e.analyzed = 1")
        conditions.append("e.knowledge_item_id IS NOT NULL")

        # Build final query
        query += " WHERE " + " AND ".join(conditions)

        # Add ordering - prioritize due exercises if requested
        if prioritize_due and quiz_type == 'review':
            query += " ORDER BY sp.next_review ASC NULLS FIRST"
        else:
            query += " ORDER BY RANDOM()"

        # Add limit with some buffer for randomization later
        query += f" LIMIT {question_count * 3}"

        # Execute query
        cursor = db.conn.execute(query, params)
        exercises = []

        for row in cursor.fetchall():
            exercise = dict(row)
            # Parse JSON fields
            if exercise.get('image_paths'):
                exercise['image_paths'] = json.loads(exercise['image_paths'])
            if exercise.get('variations'):
                exercise['variations'] = json.loads(exercise['variations'])
            if exercise.get('analysis_metadata'):
                exercise['analysis_metadata'] = json.loads(exercise['analysis_metadata'])
            exercises.append(exercise)

        # If prioritizing due and we have enough exercises, sort by due date and review count
        if prioritize_due and len(exercises) >= question_count:
            exercises = self._prioritize_due_exercises(db, exercises, question_count)
        else:
            # Otherwise, just shuffle for randomness
            random.shuffle(exercises)

        return exercises

    def _prioritize_due_exercises(self, db: Database, exercises: List[Dict[str, Any]],
                                  count: int) -> List[Dict[str, Any]]:
        """Prioritize exercises based on SM-2 review schedule.

        Args:
            db: Database connection
            exercises: List of candidate exercises
            count: Number of exercises needed

        Returns:
            Prioritized and randomized list of exercises
        """
        # Get SM-2 data for each exercise's core loop
        exercise_scores = []
        current_time = datetime.now()

        for ex in exercises:
            knowledge_item_id = ex.get('knowledge_item_id')
            if not knowledge_item_id:
                exercise_scores.append((ex, 0))  # No priority
                continue

            # Get progress data
            cursor = db.conn.execute("""
                SELECT next_review, review_interval, total_attempts
                FROM student_progress
                WHERE knowledge_item_id = ?
            """, (knowledge_item_id,))

            row = cursor.fetchone()

            if not row or row['next_review'] is None:
                # Never reviewed - high priority
                priority = 1000
            else:
                next_review = datetime.fromisoformat(row['next_review'])
                days_overdue = (current_time - next_review).days

                # Calculate priority score
                # Higher score = higher priority
                if days_overdue > 0:
                    # Overdue - very high priority
                    priority = 100 + days_overdue
                else:
                    # Not yet due - lower priority
                    priority = max(0, 50 + days_overdue)

            exercise_scores.append((ex, priority))

        # Sort by priority (descending) with some randomization
        exercise_scores.sort(key=lambda x: x[1] + random.uniform(-10, 10), reverse=True)

        # Return top exercises
        return [ex for ex, _ in exercise_scores[:count]]

    def get_next_question(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get next unanswered question in quiz.

        Args:
            session_id: Quiz session ID

        Returns:
            Dictionary with question data or None if quiz is complete
        """
        with Database() as db:
            # Find next unanswered question
            cursor = db.conn.execute("""
                SELECT qa.question_number, qa.exercise_id, e.text, e.difficulty,
                       e.knowledge_item_id, e.topic_id, e.has_images, e.image_paths
                FROM quiz_answers qa
                JOIN exercises e ON qa.exercise_id = e.id
                WHERE qa.session_id = ? AND qa.student_answer IS NULL
                ORDER BY qa.question_number
                LIMIT 1
            """, (session_id,))

            row = cursor.fetchone()
            if not row:
                return None  # No more questions

            # Get total questions
            total = db.conn.execute("""
                SELECT total_questions FROM quiz_sessions WHERE id = ?
            """, (session_id,)).fetchone()

            question = {
                'question_number': row['question_number'],
                'total_questions': total['total_questions'] if total else 0,
                'exercise_id': row['exercise_id'],
                'text': row['text'],
                'difficulty': row['difficulty'],
                'knowledge_item_id': row['knowledge_item_id'],
                'topic_id': row['topic_id'],
                'has_images': bool(row['has_images']),
                'image_paths': json.loads(row['image_paths']) if row['image_paths'] else []
            }

            return question

    def submit_answer(self, session_id: str, exercise_id: str,
                     user_answer: str, time_taken: int,
                     hint_used: bool = False) -> Dict[str, Any]:
        """Submit answer and get AI feedback with SM-2 update.

        Args:
            session_id: Quiz session ID
            exercise_id: Exercise ID
            user_answer: User's answer text
            time_taken: Time taken in seconds
            hint_used: Whether hints were used

        Returns:
            Dictionary with:
                - correct: bool
                - score: float (0.0 to 1.0)
                - feedback: str (AI-generated feedback)
                - sm2_update: dict with SM-2 parameters
                - remaining_questions: int
        """
        with Database() as db:
            # Check answer using AI tutor
            tutor_response = self.tutor.check_answer(
                exercise_id=exercise_id,
                user_answer=user_answer,
                provide_hints=False  # No hints after submission
            )

            if not tutor_response.success:
                return {
                    'correct': False,
                    'score': 0.0,
                    'feedback': f"Error evaluating answer: {tutor_response.content}",
                    'sm2_update': {},
                    'remaining_questions': 0
                }

            # Parse feedback to extract score
            feedback_text = tutor_response.content
            score, is_correct = self._extract_score_from_feedback(feedback_text)

            # Apply hint penalty if configured (currently 0.0 in config)
            if hint_used and Config.HINT_PENALTY > 0:
                score = max(0.0, score - Config.HINT_PENALTY)

            # Convert score to SM-2 quality
            quality = self.sm2.convert_score_to_quality(
                score=score,
                hint_used=hint_used
            )

            # Get or create SM-2 progress for this exercise's core loop
            exercise = db.get_exercise(exercise_id)
            knowledge_item_id = exercise.get('knowledge_item_id') if exercise else None

            sm2_update = {}
            if knowledge_item_id:
                sm2_update = self._update_sm2_progress(
                    db, knowledge_item_id, exercise['course_code'],
                    quality, is_correct
                )

            # Store answer in database
            db.conn.execute("""
                UPDATE quiz_answers
                SET student_answer = ?,
                    is_correct = ?,
                    score = ?,
                    hint_used = ?,
                    answered_at = CURRENT_TIMESTAMP,
                    time_spent = ?
                WHERE session_id = ? AND exercise_id = ?
            """, (user_answer, is_correct, score, hint_used,
                  time_taken, session_id, exercise_id))

            # Get remaining questions count
            remaining = db.conn.execute("""
                SELECT COUNT(*) as count
                FROM quiz_answers
                WHERE session_id = ? AND student_answer IS NULL
            """, (session_id,)).fetchone()

            return {
                'correct': is_correct,
                'score': score,
                'feedback': feedback_text,
                'sm2_update': sm2_update,
                'remaining_questions': remaining['count'] if remaining else 0
            }

    def _extract_score_from_feedback(self, feedback: str) -> tuple[float, bool]:
        """Extract numerical score from AI feedback.

        Args:
            feedback: AI-generated feedback text

        Returns:
            Tuple of (score, is_correct)
        """
        # Use simple heuristics to determine correctness from feedback
        feedback_lower = feedback.lower()

        # Keywords indicating correctness
        correct_keywords = ['correct', 'right', 'excellent', 'perfect', 'good job',
                           'well done', 'corretto', 'giusto', 'ottimo', 'perfetto']
        incorrect_keywords = ['incorrect', 'wrong', 'mistake', 'error', 'not quite',
                             'sbagliato', 'errore', 'non corretto']

        # Check for keywords
        is_correct = any(keyword in feedback_lower for keyword in correct_keywords)
        is_incorrect = any(keyword in feedback_lower for keyword in incorrect_keywords)

        # Determine score based on feedback
        if is_correct and not is_incorrect:
            # Fully correct
            score = 0.9 + random.uniform(0, 0.1)  # 0.9-1.0
        elif is_incorrect:
            # Incorrect but might have partial credit
            if 'partially' in feedback_lower or 'partial' in feedback_lower:
                score = 0.5 + random.uniform(0, 0.2)  # 0.5-0.7
            else:
                score = 0.0 + random.uniform(0, 0.3)  # 0.0-0.3
        else:
            # Ambiguous - assume partially correct
            score = 0.6 + random.uniform(0, 0.2)  # 0.6-0.8

        return (round(score, 2), score >= 0.7)

    def _update_sm2_progress(self, db: Database, knowledge_item_id: str,
                            course_code: str, quality: int,
                            is_correct: bool) -> Dict[str, Any]:
        """Update SM-2 progress for a core loop.

        Args:
            db: Database connection
            knowledge_item_id: Core loop ID
            course_code: Course code
            quality: SM-2 quality rating (0-5)
            is_correct: Whether answer was correct

        Returns:
            Dictionary with SM-2 update information
        """
        # Get current progress
        cursor = db.conn.execute("""
            SELECT *
            FROM student_progress
            WHERE course_code = ? AND knowledge_item_id = ?
        """, (course_code, knowledge_item_id))

        row = cursor.fetchone()

        if row:
            # Existing progress
            current_ef = 2.5  # Default if not tracked
            repetition = row['review_interval'] or 0
            previous_interval = row['review_interval'] or 0
            total_attempts = row['total_attempts'] or 0
            correct_attempts = row['correct_attempts'] or 0
        else:
            # New progress
            current_ef = 2.5
            repetition = 0
            previous_interval = 0
            total_attempts = 0
            correct_attempts = 0

        # Calculate SM-2 update
        sm2_result = self.sm2.calculate(
            quality=quality,
            easiness_factor=current_ef,
            repetition_number=repetition,
            previous_interval=previous_interval
        )

        # Update progress
        new_attempts = total_attempts + 1
        new_correct = correct_attempts + (1 if is_correct else 0)
        mastery_score = new_correct / new_attempts if new_attempts > 0 else 0.0

        if row:
            # Update existing
            db.conn.execute("""
                UPDATE student_progress
                SET total_attempts = ?,
                    correct_attempts = ?,
                    mastery_score = ?,
                    last_practiced = CURRENT_TIMESTAMP,
                    next_review = ?,
                    review_interval = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE course_code = ? AND knowledge_item_id = ?
            """, (new_attempts, new_correct, mastery_score,
                  sm2_result.next_review_date.isoformat(),
                  sm2_result.interval_days,
                  course_code, knowledge_item_id))
        else:
            # Insert new
            db.conn.execute("""
                INSERT INTO student_progress
                (course_code, knowledge_item_id, total_attempts, correct_attempts,
                 mastery_score, last_practiced, next_review, review_interval)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
            """, (course_code, knowledge_item_id, new_attempts, new_correct,
                  mastery_score, sm2_result.next_review_date.isoformat(),
                  sm2_result.interval_days))

        return {
            'easiness_factor': sm2_result.easiness_factor,
            'repetition_number': sm2_result.repetition_number,
            'interval_days': sm2_result.interval_days,
            'next_review': sm2_result.next_review_date.isoformat(),
            'quality': quality,
            'mastery_score': mastery_score
        }

    def complete_quiz(self, session_id: str) -> Dict[str, Any]:
        """Complete quiz and calculate final score.

        Args:
            session_id: Quiz session ID

        Returns:
            Dictionary with quiz statistics
        """
        with Database() as db:
            # Calculate statistics
            stats = db.conn.execute("""
                SELECT
                    COUNT(*) as total_questions,
                    SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as total_correct,
                    AVG(score) as avg_score,
                    SUM(time_spent) as total_time
                FROM quiz_answers
                WHERE session_id = ? AND student_answer IS NOT NULL
            """, (session_id,)).fetchone()

            if not stats or stats['total_questions'] == 0:
                return {
                    'completed': False,
                    'error': 'No answers found for this quiz'
                }

            total_questions = stats['total_questions']
            total_correct = stats['total_correct'] or 0
            avg_score = stats['avg_score'] or 0.0
            total_time = stats['total_time'] or 0

            # Final score as percentage
            final_score = (avg_score * 100) if avg_score else 0.0

            # Update quiz session
            db.conn.execute("""
                UPDATE quiz_sessions
                SET completed_at = CURRENT_TIMESTAMP,
                    total_correct = ?,
                    score = ?,
                    time_spent = ?
                WHERE id = ?
            """, (total_correct, final_score, total_time, session_id))

            # Get difficulty breakdown
            difficulty_stats = db.conn.execute("""
                SELECT
                    e.difficulty,
                    COUNT(*) as count,
                    SUM(CASE WHEN qa.is_correct THEN 1 ELSE 0 END) as correct
                FROM quiz_answers qa
                JOIN exercises e ON qa.exercise_id = e.id
                WHERE qa.session_id = ?
                GROUP BY e.difficulty
            """, (session_id,)).fetchall()

            difficulty_breakdown = {}
            for row in difficulty_stats:
                difficulty = row['difficulty'] or 'unknown'
                difficulty_breakdown[difficulty] = {
                    'total': row['count'],
                    'correct': row['correct'] or 0,
                    'percentage': (row['correct'] or 0) / row['count'] * 100
                }

            return {
                'completed': True,
                'session_id': session_id,
                'total_questions': total_questions,
                'total_correct': total_correct,
                'final_score': round(final_score, 1),
                'average_score': round(avg_score, 2),
                'total_time_seconds': total_time,
                'difficulty_breakdown': difficulty_breakdown,
                'passed': final_score >= 60.0  # 60% passing threshold
            }

    def get_quiz_status(self, session_id: str) -> Dict[str, Any]:
        """Get current quiz status.

        Args:
            session_id: Quiz session ID

        Returns:
            Dictionary with quiz status information
        """
        with Database() as db:
            # Get session info
            session = db.conn.execute("""
                SELECT * FROM quiz_sessions WHERE id = ?
            """, (session_id,)).fetchone()

            if not session:
                return {'error': 'Quiz session not found'}

            # Get progress
            answered = db.conn.execute("""
                SELECT COUNT(*) as count
                FROM quiz_answers
                WHERE session_id = ? AND student_answer IS NOT NULL
            """, (session_id,)).fetchone()

            total = session['total_questions']
            answered_count = answered['count'] if answered else 0

            return {
                'session_id': session_id,
                'course_code': session['course_code'],
                'quiz_type': session['quiz_type'],
                'total_questions': total,
                'answered': answered_count,
                'remaining': total - answered_count,
                'started_at': session['started_at'],
                'completed': session['completed_at'] is not None,
                'completed_at': session['completed_at']
            }
