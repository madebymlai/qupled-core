#!/usr/bin/env python3
"""
Test script for QuizManager functionality.
"""

from core.quiz import QuizManager
from core.sm2 import SM2Algorithm
from storage.database import Database
from datetime import datetime


def test_sm2_algorithm():
    """Test SM-2 algorithm calculations."""
    print("=" * 60)
    print("Testing SM-2 Algorithm")
    print("=" * 60)

    sm2 = SM2Algorithm()

    # Test 1: Perfect answer
    result = sm2.calculate(quality=5, easiness_factor=2.5, repetition_number=0)
    print(f"\nTest 1 - Perfect Answer (quality=5):")
    print(f"  Easiness Factor: {result.easiness_factor}")
    print(f"  Repetition: {result.repetition_number}")
    print(f"  Interval: {result.interval_days} days")
    print(f"  Next Review: {result.next_review_date.strftime('%Y-%m-%d')}")

    # Test 2: Incorrect answer
    result = sm2.calculate(quality=1, easiness_factor=2.5, repetition_number=2)
    print(f"\nTest 2 - Incorrect Answer (quality=1):")
    print(f"  Easiness Factor: {result.easiness_factor}")
    print(f"  Repetition: {result.repetition_number} (should reset to 0)")
    print(f"  Interval: {result.interval_days} days (should be 1)")

    # Test 3: Score to quality conversion
    print(f"\nTest 3 - Score to Quality Conversion:")
    test_scores = [1.0, 0.9, 0.75, 0.6, 0.4, 0.1]
    for score in test_scores:
        quality = sm2.convert_score_to_quality(score, hint_used=False)
        print(f"  Score {score:.1f} -> Quality {quality}")

    # Test 4: Hint penalty
    print(f"\nTest 4 - Hint Usage Impact:")
    quality_no_hint = sm2.convert_score_to_quality(0.9, hint_used=False)
    quality_with_hint = sm2.convert_score_to_quality(0.9, hint_used=True)
    print(f"  Score 0.9 without hint: Quality {quality_no_hint}")
    print(f"  Score 0.9 with hint: Quality {quality_with_hint}")


def test_quiz_creation():
    """Test quiz creation and question selection."""
    print("\n" + "=" * 60)
    print("Testing Quiz Creation")
    print("=" * 60)

    # Check if database has courses and exercises
    with Database() as db:
        courses = db.get_all_courses()
        if not courses:
            print("\n[SKIP] No courses found in database. Please add courses first.")
            return

        course_code = courses[0]['code']
        print(f"\nUsing course: {course_code} - {courses[0]['name']}")

        # Get exercise count
        exercises = db.get_exercises_by_course(course_code, analyzed_only=True)
        exercises_with_knowledge_items = [ex for ex in exercises if ex.get('knowledge_item_id')]

        print(f"Total analyzed exercises: {len(exercises)}")
        print(f"Exercises with core loops: {len(exercises_with_knowledge_items)}")

        if len(exercises_with_knowledge_items) < 5:
            print("\n[SKIP] Not enough exercises with core loops for testing.")
            return

    # Create quiz manager
    quiz_manager = QuizManager(language="en")

    # Test 1: Random quiz
    try:
        print("\nTest 1 - Creating Random Quiz:")
        session_id = quiz_manager.create_quiz(
            course_code=course_code,
            quiz_type='random',
            question_count=5,
            prioritize_due=False
        )
        print(f"  ✓ Quiz created: {session_id}")

        # Get quiz status
        status = quiz_manager.get_quiz_status(session_id)
        print(f"  Total questions: {status['total_questions']}")
        print(f"  Answered: {status['answered']}")
        print(f"  Remaining: {status['remaining']}")

        # Get first question
        question = quiz_manager.get_next_question(session_id)
        if question:
            print(f"\n  First question (#{question['question_number']}):")
            print(f"  Difficulty: {question['difficulty']}")
            print(f"  Text preview: {question['text'][:100]}...")

    except ValueError as e:
        print(f"  ✗ Error: {e}")

    # Test 2: Review quiz (if any progress exists)
    try:
        print("\nTest 2 - Creating Review Quiz:")
        session_id = quiz_manager.create_quiz(
            course_code=course_code,
            quiz_type='review',
            question_count=3,
            prioritize_due=True
        )
        print(f"  ✓ Review quiz created: {session_id}")

        status = quiz_manager.get_quiz_status(session_id)
        print(f"  Questions due for review: {status['total_questions']}")

    except ValueError as e:
        print(f"  ✗ Error: {e}")

    # Test 3: Difficulty filter
    try:
        print("\nTest 3 - Creating Quiz with Difficulty Filter:")
        session_id = quiz_manager.create_quiz(
            course_code=course_code,
            quiz_type='random',
            question_count=3,
            difficulty='medium',
            prioritize_due=False
        )
        print(f"  ✓ Medium difficulty quiz created: {session_id}")

    except ValueError as e:
        print(f"  ✗ Error: {e}")


def test_answer_submission():
    """Test answer submission and evaluation (mock test)."""
    print("\n" + "=" * 60)
    print("Testing Answer Submission (Mock)")
    print("=" * 60)

    with Database() as db:
        courses = db.get_all_courses()
        if not courses:
            print("\n[SKIP] No courses found.")
            return

        course_code = courses[0]['code']
        exercises = db.get_exercises_by_course(course_code, analyzed_only=True)
        exercises_with_knowledge_items = [ex for ex in exercises if ex.get('knowledge_item_id')]

        if not exercises_with_knowledge_items:
            print("\n[SKIP] No exercises with core loops.")
            return

    quiz_manager = QuizManager(language="en")

    try:
        # Create a small quiz
        session_id = quiz_manager.create_quiz(
            course_code=course_code,
            quiz_type='random',
            question_count=1,
            prioritize_due=False
        )

        question = quiz_manager.get_next_question(session_id)
        if not question:
            print("\n[SKIP] No question available.")
            return

        print(f"\nQuestion: {question['text'][:150]}...")
        print(f"\nSimulating answer submission...")

        # Note: This will make actual API call to LLM for evaluation
        # Comment out if you don't want to use API credits
        print("[INFO] Skipping actual submission to avoid API usage.")
        print("[INFO] In production, this would call:")
        print("  - quiz_manager.submit_answer()")
        print("  - AI tutor for evaluation")
        print("  - SM-2 algorithm for scheduling")

        # Uncomment to test real submission:
        # result = quiz_manager.submit_answer(
        #     session_id=session_id,
        #     exercise_id=question['exercise_id'],
        #     user_answer="Test answer",
        #     time_taken=60,
        #     hint_used=False
        # )
        # print(f"\nResult:")
        # print(f"  Correct: {result['correct']}")
        # print(f"  Score: {result['score']}")
        # print(f"  Feedback: {result['feedback'][:200]}...")

    except Exception as e:
        print(f"\n✗ Error: {e}")


def test_quiz_completion():
    """Test quiz completion and statistics."""
    print("\n" + "=" * 60)
    print("Testing Quiz Completion")
    print("=" * 60)

    with Database() as db:
        # Find any completed quiz sessions
        cursor = db.conn.execute("""
            SELECT id, course_code, total_questions, total_correct, score
            FROM quiz_sessions
            WHERE completed_at IS NOT NULL
            ORDER BY completed_at DESC
            LIMIT 1
        """)

        row = cursor.fetchone()
        if not row:
            print("\n[SKIP] No completed quizzes found.")
            print("[INFO] Complete a quiz first to test this functionality.")
            return

        session_id = row['id']
        print(f"\nAnalyzing completed quiz: {session_id}")
        print(f"Course: {row['course_code']}")
        print(f"Questions: {row['total_questions']}")
        print(f"Correct: {row['total_correct']}")
        print(f"Score: {row['score']:.1f}%")

    quiz_manager = QuizManager(language="en")
    stats = quiz_manager.complete_quiz(session_id)

    if stats.get('error'):
        print(f"\n✗ Error: {stats['error']}")
    else:
        print(f"\nQuiz Statistics:")
        print(f"  Total Questions: {stats['total_questions']}")
        print(f"  Correct Answers: {stats['total_correct']}")
        print(f"  Final Score: {stats['final_score']:.1f}%")
        print(f"  Passed: {'Yes' if stats['passed'] else 'No'}")
        if stats.get('difficulty_breakdown'):
            print(f"\n  Difficulty Breakdown:")
            for difficulty, data in stats['difficulty_breakdown'].items():
                print(f"    {difficulty}: {data['correct']}/{data['total']} ({data['percentage']:.1f}%)")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("EXAMINA QUIZ MANAGER TEST SUITE")
    print("=" * 60)

    test_sm2_algorithm()
    test_quiz_creation()
    test_answer_submission()
    test_quiz_completion()

    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
