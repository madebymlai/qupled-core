"""Review Engine - Exercise-based review with LLM evaluation.

This module provides:
- Exercise generation based on knowledge items + linked exercises
- Answer evaluation with partial credit

Used by examina-cloud for Review Mode v2.
"""

import json
from dataclasses import dataclass
from typing import Optional, Protocol


class LLMInterface(Protocol):
    """Protocol for LLM generation."""

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        """Generate a response from the LLM."""
        ...


@dataclass
class ExerciseExample:
    """Example exercise with optional solution."""

    text: str
    solution: Optional[str] = None
    source_type: str = "practice"  # "exam" or "practice"
    image_context: Optional[str] = None  # Vision LLM description of associated diagram
    exercise_context: Optional[str] = None  # Parent exercise text for sub-exercises


@dataclass
class GeneratedExercise:
    """Generated review exercise."""

    exercise_text: str
    expected_answer: str
    exercise_type: str  # calculation, short_answer, explanation, scenario


@dataclass
class ReviewEvaluation:
    """Result of review answer evaluation."""

    score: float  # 0.0 - 1.0
    is_correct: bool  # True if score >= 0.7
    feedback: str
    correct_answer: str


def calculate_mastery(average_score: float, review_count: int) -> float:
    """Calculate mastery for a single concept.

    Formula: mastery = average_score * min(review_count / 6, 1.0)

    Progression (for exam prep):
    - 3 reviews at 70% = 35% mastery (bronze)
    - 4 reviews at 75% = 50% mastery (silver)
    - 5 reviews at 80% = 67% mastery (gold)
    - 6 reviews at 90% = 90% mastery (diamond)

    Args:
        average_score: Average score from all reviews (0.0 - 1.0)
        review_count: Total number of reviews done

    Returns:
        Mastery percentage (0.0 - 1.0)
    """
    if review_count == 0:
        return 0.0
    confidence = min(review_count / 6, 1.0)
    return average_score * confidence


class ReviewEngine:
    """Engine for exercise-based review with LLM evaluation.

    Usage:
        engine = ReviewEngine(llm_manager, language="en")  # or "it", "de", etc.

        # Generate exercise
        exercise = engine.generate_exercise(
            knowledge_item_name="Base Conversion",
            learning_approach="procedural",
            examples=[...],
            recent_exercises=["Convert 45 to binary..."],
        )

        # Evaluate answer
        result = engine.evaluate_answer(
            exercise_text=exercise.exercise_text,
            expected_answer=exercise.expected_answer,
            student_answer="101101",
            exercise_type=exercise.exercise_type,
        )

        # Map to SM2
        quality = score_to_quality(result.score)
    """

    # Uses LEARNING_APPROACHES from analyzer.py for consistent definitions

    def __init__(self, llm: LLMInterface, language: str = "en"):
        """Initialize with LLM interface.

        Args:
            llm: LLM manager implementing generate() method
            language: Output language (ISO 639-1 code, e.g., "en", "it", "de")
        """
        self._llm = llm
        self._language = language
        self._reasoner_model = "deepseek-reasoner"

    def _language_instruction(self) -> str:
        """Generate language instruction for prompts."""
        return f"the language with ISO 639-1 code '{self._language}'"

    def generate_exercise(
        self,
        knowledge_item_name: str,
        learning_approach: str,
        examples: list[ExerciseExample],
        recent_exercises: Optional[list[str]] = None,
    ) -> GeneratedExercise:
        """Generate a review exercise based on knowledge item and examples.

        Args:
            knowledge_item_name: Name of the knowledge item being reviewed
            learning_approach: One of procedural, conceptual, factual, analytical
            examples: Real exam/practice exercises as examples
            recent_exercises: Recent generated exercises to avoid (for variety)

        Returns:
            GeneratedExercise with text, expected answer, and type
        """
        # Separate by source type - exams are primary
        exam_ex = [ex for ex in examples if ex.source_type == "exam"]
        practice_ex = [ex for ex in examples if ex.source_type == "practice"]

        # Priority: exam primary, practice context
        if exam_ex:
            primary = exam_ex
        else:
            primary = practice_ex

        # Format examples
        primary_text = self._format_examples(primary[:3])

        system = "You are a teacher making a new exercise for a test."
        prompt = f"""Create a similar exercise about {knowledge_item_name}.

Examples:
{primary_text}

IMPORTANT: Output in {self._language_instruction()}. Use LaTeX: $...$ inline, $$...$$ display.

Return JSON:
{{
  "exercise_text": "the exercise",
  "expected_answer": "brief solution"
}}"""

        try:
            response = self._llm.generate(
                prompt, model=self._reasoner_model, system=system, max_tokens=2048
            )
            response_text = response.text if hasattr(response, "text") else str(response)
            return self._parse_exercise_response(response_text, knowledge_item_name)
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"Exercise generation failed: {e}")
            return GeneratedExercise(
                exercise_text=f"Explain the key concepts of {knowledge_item_name}.",
                expected_answer="A clear explanation of the main concepts.",
                exercise_type="explanation",
            )

    def evaluate_answer(
        self,
        exercise_text: str,
        expected_answer: str,
        student_answer: str,
        exercise_type: str = "explanation",
    ) -> ReviewEvaluation:
        """Evaluate student's answer to a review exercise.

        Args:
            exercise_text: The exercise question
            expected_answer: Expected solution
            student_answer: Student's submitted answer
            exercise_type: Type of exercise (unused, kept for API compatibility)

        Returns:
            ReviewEvaluation with score, feedback, and correct answer
        """
        system = "You are a teacher correcting your student's work."
        prompt = f"""Exercise: {exercise_text}
Expected: {expected_answer}
Student: {student_answer}

IMPORTANT: Respond in {self._language_instruction()}.

Return JSON:
{{
  "score": 0.0-1.0,
  "is_correct": true/false,
  "feedback": "brief feedback"
}}"""

        try:
            response = self._llm.generate(prompt, model=self._reasoner_model, system=system)
            response_text = response.text if hasattr(response, "text") else str(response)
            return self._parse_evaluation_response(response_text, expected_answer, student_answer)
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"Answer evaluation failed: {e}")
            return self._fallback_evaluation(expected_answer, student_answer)

    def _format_examples(self, examples: list[ExerciseExample]) -> str:
        """Format exercise examples for the prompt."""
        if not examples:
            return "No examples provided."

        formatted = []
        for i, ex in enumerate(examples, 1):
            # Include parent context for sub-exercises
            if ex.exercise_context:
                text = f"Example {i}:\nContext: {ex.exercise_context}\nSub-exercise: {ex.text}"
            else:
                text = f"Example {i}:\n{ex.text}"
            if ex.image_context:
                text += f"\n[IMAGE CONTEXT: {ex.image_context}]"
            if ex.solution:
                text += f"\nSolution: {ex.solution}"
            formatted.append(text)

        return "\n\n".join(formatted)

    def _parse_exercise_response(
        self,
        response: str,
        knowledge_item_name: str,
    ) -> GeneratedExercise:
        """Parse JSON response from exercise generation."""
        # Try to parse the entire response as JSON first
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "exercise_text" in data:
                return GeneratedExercise(
                    exercise_text=data.get("exercise_text", ""),
                    expected_answer=data.get("expected_answer", ""),
                    exercise_type=data.get("exercise_type", "explanation"),
                )
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in response (handles markdown code blocks)
        # Find the outermost { } pair by counting braces
        start_idx = response.find("{")
        if start_idx != -1:
            brace_count = 0
            for i, char in enumerate(response[start_idx:], start_idx):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = response[start_idx : i + 1]
                        try:
                            data = json.loads(json_str)
                            if isinstance(data, dict) and "exercise_text" in data:
                                return GeneratedExercise(
                                    exercise_text=data.get("exercise_text", ""),
                                    expected_answer=data.get("expected_answer", ""),
                                    exercise_type=data.get("exercise_type", "explanation"),
                                )
                        except json.JSONDecodeError:
                            pass
                        break

        # Fallback: use response as exercise text
        return GeneratedExercise(
            exercise_text=response[:500] if response else f"Explain {knowledge_item_name}.",
            expected_answer="See reference material.",
            exercise_type="explanation",
        )

    def _parse_evaluation_response(
        self,
        response: str,
        expected_answer: str,
        student_answer: str,
    ) -> ReviewEvaluation:
        """Parse JSON response from evaluation."""
        # Try to parse the entire response as JSON first
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "score" in data:
                score = float(data.get("score", 0.0))
                return ReviewEvaluation(
                    score=score,
                    is_correct=data.get("is_correct", score >= 0.7),
                    feedback=data.get("feedback", "Answer evaluated."),
                    correct_answer=data.get("correct_answer", expected_answer),
                )
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in response (handles markdown code blocks)
        # Find the outermost { } pair by counting braces
        start_idx = response.find("{")
        if start_idx != -1:
            brace_count = 0
            for i, char in enumerate(response[start_idx:], start_idx):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = response[start_idx : i + 1]
                        try:
                            data = json.loads(json_str)
                            if isinstance(data, dict) and "score" in data:
                                score = float(data.get("score", 0.0))
                                return ReviewEvaluation(
                                    score=score,
                                    is_correct=data.get("is_correct", score >= 0.7),
                                    feedback=data.get("feedback", "Answer evaluated."),
                                    correct_answer=data.get("correct_answer", expected_answer),
                                )
                        except json.JSONDecodeError:
                            pass
                        break

        # Fallback
        return self._fallback_evaluation(expected_answer, student_answer)

    async def evaluate_stream(
        self,
        exercise_text: str,
        expected_answer: str,
        student_answer: str,
        exercise_type: str = "explanation",
    ):
        """Stream evaluation feedback.

        Yields chunks of feedback text as they are generated.

        Args:
            exercise_text: The exercise question
            expected_answer: Expected solution
            student_answer: Student's submitted answer
            exercise_type: Type of exercise (unused, kept for API compatibility)

        Yields:
            String chunks of feedback
        """
        system = "You are a teacher correcting your student's work."
        prompt = f"""Exercise: {exercise_text}
Expected: {expected_answer}
Student: {student_answer}

IMPORTANT: Respond in {self._language_instruction()}.

Provide feedback on the student's answer. Be encouraging but accurate.
Start with whether the answer is correct or not, then explain what was good or what needs improvement."""

        async for chunk in self._llm.generate_stream(prompt, model=self._reasoner_model, system=system):
            yield chunk

    def _fallback_evaluation(
        self,
        expected_answer: str,
        student_answer: str,
    ) -> ReviewEvaluation:
        """Fallback evaluation using keyword matching."""
        if not student_answer.strip():
            return ReviewEvaluation(
                score=0.0,
                is_correct=False,
                feedback="No answer provided.",
                correct_answer=expected_answer,
            )

        if not expected_answer:
            return ReviewEvaluation(
                score=0.5,
                is_correct=False,
                feedback="Answer recorded for review.",
                correct_answer="",
            )

        # Simple keyword matching
        answer_lower = student_answer.lower()
        expected_lower = expected_answer.lower()

        # Extract keywords
        keywords = [w for w in expected_lower.split() if len(w) > 3][:10]
        if keywords:
            matches = sum(1 for kw in keywords if kw in answer_lower)
            score = min(matches / len(keywords), 1.0)
        else:
            score = 0.5

        return ReviewEvaluation(
            score=score,
            is_correct=score >= 0.7,
            feedback="Answer evaluated using keyword matching.",
            correct_answer=expected_answer,
        )
