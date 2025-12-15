"""Database-agnostic answer evaluation.

This module provides answer evaluation logic that can be used with any LLM backend.
Supports three modes:
- QUIZ: Structured scoring with JSON output (is_correct, score, feedback)
- LEARN: Pedagogical feedback focused on learning (hints, encouragement)
- RECALL: Compare student explanation to reference content (recall_score, points, misconceptions)

Extracted from examina-cloud/backend/app/api/v1/quiz.py and learn.py to enable:
- Reuse across CLI and web
- Consistent evaluation behavior
- Clean separation of evaluation logic from data access
"""

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol


class EvaluationMode(Enum):
    """Mode for answer evaluation."""

    QUIZ = "quiz"  # Structured scoring with JSON output
    LEARN = "learn"  # Pedagogical feedback, hints, encouragement
    RECALL = "recall"  # Compare student explanation to reference content


@dataclass(frozen=True)
class EvaluationResult:
    """Result of answer evaluation.

    Attributes:
        is_correct: Whether the answer is substantially correct (QUIZ mode)
        score: Score from 0.0 to 1.0 (QUIZ mode)
        feedback: Feedback text explaining the evaluation
        hint: Optional hint if answer was wrong and hints requested (LEARN mode)
    """

    is_correct: Optional[bool]
    score: Optional[float]
    feedback: str
    hint: Optional[str] = None


@dataclass
class RecallEvaluationResult:
    """Result of recall evaluation.

    Attributes:
        recall_score: Score from 0.0 to 1.0 indicating recall quality
        correct_points: List of key points correctly explained
        missed_points: List of important points not mentioned
        misconceptions: List of incorrect statements (empty if none)
        feedback: Summary of recall quality (2-3 sentences)
        success: Whether evaluation succeeded (True) or used fallback (False)
    """

    recall_score: float
    correct_points: list[str]
    missed_points: list[str]
    misconceptions: list[str]
    feedback: str
    success: bool = True


RECALL_EVALUATION_PROMPT = """
Compare this student's explanation of a concept to the **reference material**.

Concept: {concept_name}
Reference Content:
{reference_content}

Student's Explanation:
{student_explanation}

Evaluate their recall and provide a JSON response with:
- **"recall_score"**: float from 0.0 to 1.0
- **"correct_points"**: list of key points they **correctly explained**
- **"missed_points"**: list of important points they **did not mention**
- **"misconceptions"**: list of any **incorrect statements** (empty if none)
- **"feedback"**: 2-3 sentence summary

Be **encouraging but honest**. Focus on **key concepts**, not exact wording.
Respond **ONLY** with valid JSON.
"""


class LLMInterface(Protocol):
    """Protocol for LLM generation.

    Any LLM manager implementing this interface can be used with AnswerEvaluator.
    """

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The prompt to send to the LLM
            model: Optional model override
            system: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            json_mode: Whether to request JSON output

        Returns:
            The generated text response
        """
        ...


class AnswerEvaluator:
    """Database-agnostic answer evaluator.

    Provides three evaluation modes:
    - QUIZ: Returns structured JSON with is_correct, score, feedback
    - LEARN: Returns pedagogical feedback with optional hints
    - RECALL: Returns recall evaluation with score, points, and misconceptions

    Usage:
        evaluator = AnswerEvaluator(llm_manager)

        # Quiz mode
        result = evaluator.evaluate(
            question="What is 2+2?",
            student_answer="4",
            mode=EvaluationMode.QUIZ,
            expected_solution="4",
        )

        # Recall mode
        recall_result = evaluator.evaluate_recall(
            concept_name="Pythagorean Theorem",
            reference_content="The theorem states...",
            student_explanation="I think it says...",
        )
    """

    def __init__(self, llm: LLMInterface):
        """Initialize with LLM interface.

        Args:
            llm: LLM manager implementing generate() method
        """
        self._llm = llm

    # ==================== PUBLIC METHODS ====================

    def evaluate(
        self,
        question: str,
        student_answer: str,
        mode: EvaluationMode = EvaluationMode.QUIZ,
        expected_solution: Optional[str] = None,
        provide_hints: bool = False,
    ) -> EvaluationResult:
        """Evaluate a student answer.

        Args:
            question: The question text
            student_answer: The student's answer
            mode: QUIZ for scoring, LEARN for pedagogical feedback
            expected_solution: Expected solution (optional but improves accuracy)
            provide_hints: Whether to include hints in LEARN mode

        Returns:
            EvaluationResult with feedback and optionally score
        """
        if mode == EvaluationMode.QUIZ:
            return self._evaluate_quiz(
                question=question,
                student_answer=student_answer,
                expected_solution=expected_solution,
            )
        else:
            return self._evaluate_learn(
                question=question,
                student_answer=student_answer,
                expected_solution=expected_solution,
                provide_hints=provide_hints,
            )

    def evaluate_recall(
        self,
        concept_name: str,
        reference_content: str,
        student_explanation: str,
    ) -> RecallEvaluationResult:
        """Evaluate a student's recall of a concept by comparing their explanation
        to the reference content.

        Args:
            concept_name: Name of the concept being reviewed
            reference_content: The actual concept content (theory/explanation)
            student_explanation: What the student wrote from memory

        Returns:
            RecallEvaluationResult with score, points, and feedback
        """
        prompt = RECALL_EVALUATION_PROMPT.format(
            concept_name=concept_name,
            reference_content=reference_content,
            student_explanation=student_explanation,
        )

        try:
            response = self._llm.generate(prompt)
            return self._parse_recall_response(
                response=response,
                student_explanation=student_explanation,
                reference_content=reference_content,
            )
        except Exception:
            # Fallback to simple keyword comparison
            return self._fallback_recall_evaluation(
                student_explanation=student_explanation,
                reference_content=reference_content,
            )

    # ==================== PRIVATE METHODS ====================

    def _evaluate_quiz(
        self,
        question: str,
        student_answer: str,
        expected_solution: Optional[str],
    ) -> EvaluationResult:
        """Evaluate in QUIZ mode with structured scoring.

        Returns JSON with is_correct, score, feedback.
        Falls back to keyword matching if JSON parsing fails.
        """
        solution_text = expected_solution or "No solution provided"

        prompt = f"""Evaluate this student answer to a quiz question.

Question: {question}

Expected Solution: {solution_text}

Student Answer: {student_answer}

Provide a JSON response with:
- **"is_correct"**: true/false (true if answer is **substantially correct**)
- **"score"**: 0.0 to 1.0 (0 = completely wrong, 1 = perfect)
- **"feedback"**: brief feedback explaining the evaluation

Respond **ONLY** with valid JSON, no other text."""

        try:
            response = self._llm.generate(prompt)
            return self._parse_quiz_response(
                response=response,
                student_answer=student_answer,
                expected_solution=expected_solution,
            )
        except Exception:
            # Fallback to simple keyword comparison
            return self._fallback_quiz_evaluation(
                student_answer=student_answer,
                expected_solution=expected_solution,
            )

    def _evaluate_learn(
        self,
        question: str,
        student_answer: str,
        expected_solution: Optional[str],
        provide_hints: bool,
    ) -> EvaluationResult:
        """Evaluate in LEARN mode with pedagogical feedback.

        Returns encouraging feedback with optional hints.
        """
        hint_instruction = ""
        if provide_hints and expected_solution:
            hint_instruction = (
                f"\n\nIf the answer is wrong, provide a hint to guide the "
                f"student towards the correct answer. The expected solution "
                f"is: {expected_solution}"
            )

        prompt = f"""Evaluate this student's practice answer.

Question: {question}

Student Answer: {student_answer}
{hint_instruction}

Provide **helpful, encouraging** feedback. If the answer is **correct**, acknowledge it.
If **incorrect**, explain what's missing or wrong **without giving away the full solution**
unless hints are requested.

Keep your response **concise** (2-4 sentences)."""

        try:
            feedback = self._llm.generate(prompt)
            return EvaluationResult(
                is_correct=None,  # LEARN mode doesn't score
                score=None,
                feedback=feedback,
                hint=None,  # Hint is embedded in feedback if requested
            )
        except Exception as e:
            return EvaluationResult(
                is_correct=None,
                score=None,
                feedback=f"Unable to evaluate answer: {str(e)}",
                hint=None,
            )

    def _parse_quiz_response(
        self,
        response: str,
        student_answer: str,
        expected_solution: Optional[str],
    ) -> EvaluationResult:
        """Parse JSON response from quiz evaluation.

        Falls back to keyword matching if JSON parsing fails.
        """
        # Try to extract JSON from response
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)

        if json_match:
            try:
                evaluation = json.loads(json_match.group())
                return EvaluationResult(
                    is_correct=evaluation.get("is_correct", False),
                    score=float(evaluation.get("score", 0.0)),
                    feedback=evaluation.get("feedback", "Answer evaluated."),
                )
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: use response as feedback, do simple matching
        is_correct = self._simple_keyword_match(
            student_answer=student_answer,
            expected_solution=expected_solution,
        )
        return EvaluationResult(
            is_correct=is_correct,
            score=0.7 if is_correct else 0.0,
            feedback=response if response else "Answer evaluated.",
        )

    def _fallback_quiz_evaluation(
        self,
        student_answer: str,
        expected_solution: Optional[str],
    ) -> EvaluationResult:
        """Fallback evaluation when LLM fails.

        Uses keyword matching with more sophisticated scoring.
        """
        if not expected_solution:
            return EvaluationResult(
                is_correct=False,
                score=0.5,
                feedback="Answer recorded. Manual review may be needed.",
            )

        answer_lower = student_answer.lower()
        solution_lower = expected_solution.lower()

        # Count matching keywords (words > 3 chars)
        solution_words = [w for w in solution_lower.split()[:10] if len(w) > 3]
        if solution_words:
            matches = sum(1 for word in solution_words if word in answer_lower)
            score = min(matches / max(len(solution_words), 5), 1.0)
        else:
            score = 0.5

        is_correct = score >= 0.5

        return EvaluationResult(
            is_correct=is_correct,
            score=score,
            feedback="Answer recorded. Manual review may be needed.",
        )

    @staticmethod
    def _simple_keyword_match(
        student_answer: str,
        expected_solution: Optional[str],
    ) -> bool:
        """Simple keyword matching for fallback evaluation."""
        if not expected_solution:
            return False

        answer_lower = student_answer.lower()
        solution_lower = expected_solution.lower()

        # Check if any of first 5 solution words appear in answer
        solution_words = solution_lower.split()[:5]
        return any(word in answer_lower for word in solution_words if word)

    def _parse_recall_response(
        self,
        response: str,
        student_explanation: str,
        reference_content: str,
    ) -> RecallEvaluationResult:
        """Parse JSON response from recall evaluation.

        Falls back to keyword matching if JSON parsing fails.
        """
        # Try to extract JSON from response
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)

        if json_match:
            try:
                evaluation = json.loads(json_match.group())
                return RecallEvaluationResult(
                    recall_score=float(evaluation.get("recall_score", 0.0)),
                    correct_points=evaluation.get("correct_points", []),
                    missed_points=evaluation.get("missed_points", []),
                    misconceptions=evaluation.get("misconceptions", []),
                    feedback=evaluation.get("feedback", "Recall evaluated."),
                    success=True,
                )
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: use simple keyword comparison
        return self._fallback_recall_evaluation(
            student_explanation=student_explanation,
            reference_content=reference_content,
        )

    def _fallback_recall_evaluation(
        self,
        student_explanation: str,
        reference_content: str,
    ) -> RecallEvaluationResult:
        """Fallback evaluation when LLM fails.

        Uses keyword matching to approximate recall quality.
        """
        explanation_lower = student_explanation.lower()
        reference_lower = reference_content.lower()

        # Extract keywords (words > 3 chars) from reference
        reference_words = [w for w in reference_lower.split() if len(w) > 3]
        unique_reference_words = list(dict.fromkeys(reference_words))[:20]  # First 20 unique

        if not unique_reference_words:
            return RecallEvaluationResult(
                recall_score=0.5,
                correct_points=[],
                missed_points=["Unable to evaluate - reference content too short"],
                misconceptions=[],
                feedback="Recall recorded. Manual review may be needed.",
                success=False,
            )

        # Count matches
        matches = sum(1 for word in unique_reference_words if word in explanation_lower)
        recall_score = min(matches / len(unique_reference_words), 1.0)

        # Determine feedback based on score
        if recall_score >= 0.7:
            feedback = (
                "Good recall of the key concepts. Keep practicing to maintain this knowledge."
            )
        elif recall_score >= 0.4:
            feedback = "Partial recall. Review the reference material to fill in the gaps."
        else:
            feedback = "Limited recall. It would be helpful to review this concept again."

        return RecallEvaluationResult(
            recall_score=recall_score,
            correct_points=[],
            missed_points=[],
            misconceptions=[],
            feedback=feedback,
            success=False,
        )
