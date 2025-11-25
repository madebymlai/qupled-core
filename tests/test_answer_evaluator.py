"""
Unit tests for AnswerEvaluator.

Tests cover QUIZ mode (structured scoring) and LEARN mode (pedagogical feedback)
without requiring actual LLM calls - uses mock LLM responses.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.answer_evaluator import (
    AnswerEvaluator,
    EvaluationMode,
    EvaluationResult,
)


# ============================================================================
# Mock LLM for Testing
# ============================================================================

class MockLLM:
    """Mock LLM that returns predefined responses."""

    def __init__(self, response: str = "", should_fail: bool = False):
        self.response = response
        self.should_fail = should_fail
        self.last_prompt = None

    def generate(self, prompt: str, **kwargs) -> str:
        self.last_prompt = prompt
        if self.should_fail:
            raise Exception("Mock LLM failure")
        return self.response


# ============================================================================
# Test EvaluationMode Enum
# ============================================================================

def test_evaluation_mode_values():
    """Test EvaluationMode enum values."""
    assert EvaluationMode.QUIZ.value == "quiz"
    assert EvaluationMode.LEARN.value == "learn"
    print("✓ test_evaluation_mode_values passed")


# ============================================================================
# Test QUIZ Mode Evaluation
# ============================================================================

def test_evaluate_quiz_correct_json():
    """Test QUIZ mode with valid JSON response."""
    mock_llm = MockLLM(
        response='{"is_correct": true, "score": 0.95, "feedback": "Excellent answer!"}'
    )
    evaluator = AnswerEvaluator(mock_llm)

    result = evaluator.evaluate(
        question="What is 2+2?",
        student_answer="4",
        mode=EvaluationMode.QUIZ,
        expected_solution="4",
    )

    assert result.is_correct is True
    assert result.score == 0.95
    assert result.feedback == "Excellent answer!"
    print("✓ test_evaluate_quiz_correct_json passed")


def test_evaluate_quiz_incorrect_json():
    """Test QUIZ mode with incorrect answer JSON response."""
    mock_llm = MockLLM(
        response='{"is_correct": false, "score": 0.2, "feedback": "Not quite right."}'
    )
    evaluator = AnswerEvaluator(mock_llm)

    result = evaluator.evaluate(
        question="What is 2+2?",
        student_answer="5",
        mode=EvaluationMode.QUIZ,
        expected_solution="4",
    )

    assert result.is_correct is False
    assert result.score == 0.2
    assert result.feedback == "Not quite right."
    print("✓ test_evaluate_quiz_incorrect_json passed")


def test_evaluate_quiz_json_with_extra_text():
    """Test QUIZ mode extracts JSON from response with extra text."""
    mock_llm = MockLLM(
        response='Here is my evaluation: {"is_correct": true, "score": 1.0, "feedback": "Perfect!"} Hope this helps!'
    )
    evaluator = AnswerEvaluator(mock_llm)

    result = evaluator.evaluate(
        question="What is the capital of France?",
        student_answer="Paris",
        mode=EvaluationMode.QUIZ,
        expected_solution="Paris",
    )

    assert result.is_correct is True
    assert result.score == 1.0
    assert result.feedback == "Perfect!"
    print("✓ test_evaluate_quiz_json_with_extra_text passed")


def test_evaluate_quiz_invalid_json_fallback():
    """Test QUIZ mode falls back to keyword matching on invalid JSON."""
    mock_llm = MockLLM(
        response="This is not valid JSON at all"
    )
    evaluator = AnswerEvaluator(mock_llm)

    result = evaluator.evaluate(
        question="What is the capital of France?",
        student_answer="Paris is the capital",
        mode=EvaluationMode.QUIZ,
        expected_solution="Paris",
    )

    # Fallback should use keyword matching - "Paris" is in the answer
    assert result.is_correct is True
    assert result.score == 0.7  # Fallback score for correct
    assert "This is not valid JSON" in result.feedback
    print("✓ test_evaluate_quiz_invalid_json_fallback passed")


def test_evaluate_quiz_llm_failure():
    """Test QUIZ mode handles LLM failure gracefully."""
    mock_llm = MockLLM(should_fail=True)
    evaluator = AnswerEvaluator(mock_llm)

    result = evaluator.evaluate(
        question="What is 2+2?",
        student_answer="four which equals 4",
        mode=EvaluationMode.QUIZ,
        expected_solution="4",
    )

    # Should use fallback evaluation
    assert result.feedback == "Answer recorded. Manual review may be needed."
    assert result.score is not None
    print("✓ test_evaluate_quiz_llm_failure passed")


def test_evaluate_quiz_no_solution():
    """Test QUIZ mode with no expected solution."""
    mock_llm = MockLLM(should_fail=True)
    evaluator = AnswerEvaluator(mock_llm)

    result = evaluator.evaluate(
        question="Explain your reasoning",
        student_answer="Because it makes sense",
        mode=EvaluationMode.QUIZ,
        expected_solution=None,
    )

    # Fallback with no solution
    assert result.is_correct is False
    assert result.score == 0.5
    print("✓ test_evaluate_quiz_no_solution passed")


# ============================================================================
# Test LEARN Mode Evaluation
# ============================================================================

def test_evaluate_learn_basic():
    """Test LEARN mode returns pedagogical feedback."""
    mock_llm = MockLLM(
        response="Great attempt! Your answer shows understanding of the concept."
    )
    evaluator = AnswerEvaluator(mock_llm)

    result = evaluator.evaluate(
        question="Explain photosynthesis",
        student_answer="Plants use sunlight to make food",
        mode=EvaluationMode.LEARN,
    )

    # LEARN mode doesn't score
    assert result.is_correct is None
    assert result.score is None
    assert "Great attempt" in result.feedback
    print("✓ test_evaluate_learn_basic passed")


def test_evaluate_learn_with_hints():
    """Test LEARN mode includes hint instruction when requested."""
    mock_llm = MockLLM(
        response="Close! Hint: Think about what else plants need besides sunlight."
    )
    evaluator = AnswerEvaluator(mock_llm)

    result = evaluator.evaluate(
        question="What do plants need for photosynthesis?",
        student_answer="Sunlight",
        mode=EvaluationMode.LEARN,
        expected_solution="Sunlight, water, and carbon dioxide",
        provide_hints=True,
    )

    assert "Hint" in result.feedback
    # Check that hint instruction was in the prompt
    assert "provide a hint" in mock_llm.last_prompt.lower()
    print("✓ test_evaluate_learn_with_hints passed")


def test_evaluate_learn_llm_failure():
    """Test LEARN mode handles LLM failure gracefully."""
    mock_llm = MockLLM(should_fail=True)
    evaluator = AnswerEvaluator(mock_llm)

    result = evaluator.evaluate(
        question="What is gravity?",
        student_answer="A force",
        mode=EvaluationMode.LEARN,
    )

    assert result.is_correct is None
    assert result.score is None
    assert "Unable to evaluate" in result.feedback
    print("✓ test_evaluate_learn_llm_failure passed")


# ============================================================================
# Test Fallback Evaluation
# ============================================================================

def test_fallback_keyword_match_full():
    """Test keyword matching with multiple matches."""
    mock_llm = MockLLM(should_fail=True)
    evaluator = AnswerEvaluator(mock_llm)

    result = evaluator.evaluate(
        question="What is the process?",
        student_answer="The process involves initialization, computation, and finalization",
        mode=EvaluationMode.QUIZ,
        expected_solution="Process involves initialization computation finalization steps",
    )

    # Multiple keyword matches should give good score
    assert result.score >= 0.5
    print("✓ test_fallback_keyword_match_full passed")


def test_fallback_keyword_match_none():
    """Test keyword matching with no matches."""
    mock_llm = MockLLM(should_fail=True)
    evaluator = AnswerEvaluator(mock_llm)

    result = evaluator.evaluate(
        question="What is the capital?",
        student_answer="I don't know",
        mode=EvaluationMode.QUIZ,
        expected_solution="Paris is the capital of France",
    )

    # No keyword matches should give low score
    assert result.is_correct is False
    assert result.score < 0.5
    print("✓ test_fallback_keyword_match_none passed")


def test_simple_keyword_match():
    """Test static keyword matching method."""
    # Direct test of _simple_keyword_match
    assert AnswerEvaluator._simple_keyword_match("Paris", "Paris") is True
    assert AnswerEvaluator._simple_keyword_match("paris city", "Paris") is True
    assert AnswerEvaluator._simple_keyword_match("London", "Paris") is False
    assert AnswerEvaluator._simple_keyword_match("anything", None) is False
    assert AnswerEvaluator._simple_keyword_match("", "Paris") is False
    print("✓ test_simple_keyword_match passed")


# ============================================================================
# Test EvaluationResult
# ============================================================================

def test_evaluation_result_frozen():
    """Test EvaluationResult is immutable."""
    result = EvaluationResult(
        is_correct=True,
        score=0.9,
        feedback="Good!",
    )

    try:
        result.score = 0.5  # Should raise
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass  # Expected - dataclass is frozen

    print("✓ test_evaluation_result_frozen passed")


def test_evaluation_result_default_hint():
    """Test EvaluationResult default hint is None."""
    result = EvaluationResult(
        is_correct=True,
        score=1.0,
        feedback="Perfect!",
    )
    assert result.hint is None
    print("✓ test_evaluation_result_default_hint passed")


# ============================================================================
# Test Prompt Generation
# ============================================================================

def test_quiz_prompt_includes_question():
    """Test QUIZ prompt includes all required information."""
    mock_llm = MockLLM(response='{"is_correct": true, "score": 1.0, "feedback": "OK"}')
    evaluator = AnswerEvaluator(mock_llm)

    evaluator.evaluate(
        question="Test question here",
        student_answer="Test answer",
        mode=EvaluationMode.QUIZ,
        expected_solution="Expected solution",
    )

    assert "Test question here" in mock_llm.last_prompt
    assert "Test answer" in mock_llm.last_prompt
    assert "Expected solution" in mock_llm.last_prompt
    print("✓ test_quiz_prompt_includes_question passed")


def test_learn_prompt_no_hints_by_default():
    """Test LEARN prompt doesn't include hint instruction by default."""
    mock_llm = MockLLM(response="Feedback")
    evaluator = AnswerEvaluator(mock_llm)

    evaluator.evaluate(
        question="Test",
        student_answer="Answer",
        mode=EvaluationMode.LEARN,
        expected_solution="Solution",
        provide_hints=False,
    )

    assert "provide a hint" not in mock_llm.last_prompt.lower()
    print("✓ test_learn_prompt_no_hints_by_default passed")


# ============================================================================
# Main runner
# ============================================================================

def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running AnswerEvaluator tests")
    print("=" * 60 + "\n")

    # EvaluationMode tests
    test_evaluation_mode_values()

    # QUIZ mode tests
    test_evaluate_quiz_correct_json()
    test_evaluate_quiz_incorrect_json()
    test_evaluate_quiz_json_with_extra_text()
    test_evaluate_quiz_invalid_json_fallback()
    test_evaluate_quiz_llm_failure()
    test_evaluate_quiz_no_solution()

    # LEARN mode tests
    test_evaluate_learn_basic()
    test_evaluate_learn_with_hints()
    test_evaluate_learn_llm_failure()

    # Fallback tests
    test_fallback_keyword_match_full()
    test_fallback_keyword_match_none()
    test_simple_keyword_match()

    # EvaluationResult tests
    test_evaluation_result_frozen()
    test_evaluation_result_default_hint()

    # Prompt tests
    test_quiz_prompt_includes_question()
    test_learn_prompt_no_hints_by_default()

    print("\n" + "=" * 60)
    print("All AnswerEvaluator tests passed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all_tests()
