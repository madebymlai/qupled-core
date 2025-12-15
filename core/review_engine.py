"""Review Engine - Exercise-based review with LLM evaluation.

This module provides:
- Exercise generation based on knowledge items + linked exercises
- Answer evaluation with partial credit
- Score to SM2 quality mapping

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


def score_to_quality(score: float) -> int:
    """Map LLM evaluation score to SM2 quality rating.

    Pass = quality >= 3 = score >= 70%
    Fail = quality < 3 = score < 70% -> interval resets to 1

    Args:
        score: Score from 0.0 to 1.0

    Returns:
        SM2 quality rating from 0 to 5
    """
    if score >= 0.9:
        return 5  # Excellent - interval grows fast
    elif score >= 0.8:
        return 4  # Good - interval grows
    elif score >= 0.7:
        return 3  # Pass - interval grows (minimum to pass)
    elif score >= 0.5:
        return 2  # Close but fail - reset
    elif score >= 0.3:
        return 1  # Wrong - reset
    else:
        return 0  # Blackout - reset


def calculate_mastery(average_score: float, review_count: int) -> float:
    """Calculate mastery for a single concept.

    Formula: mastery = average_score * min(review_count / 3, 1.0)

    Args:
        average_score: Average score from all reviews (0.0 - 1.0)
        review_count: Total number of reviews done

    Returns:
        Mastery percentage (0.0 - 1.0)
    """
    if review_count == 0:
        return 0.0
    confidence = min(review_count / 3, 1.0)
    return average_score * confidence


class ReviewEngine:
    """Engine for exercise-based review with LLM evaluation.

    Usage:
        engine = ReviewEngine(llm_manager)

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

    # Learning approach to exercise type mapping
    APPROACH_PROMPTS = {
        "procedural": "Generate a CALCULATION exercise. Student must show step-by-step work.",
        "conceptual": "Generate an EXPLANATION exercise. Student must explain WHY.",
        "factual": "Generate a RECALL exercise. Student must recall facts/definitions.",
        "analytical": "Generate a SCENARIO exercise. Student must analyze a situation.",
    }

    def __init__(self, llm: LLMInterface):
        """Initialize with LLM interface.

        Args:
            llm: LLM manager implementing generate() method
        """
        self._llm = llm

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
            context = practice_ex
        else:
            primary = practice_ex
            context = []

        # Format examples
        primary_text = self._format_examples(primary[:3])
        context_text = self._format_examples(context[:2]) if context else ""

        # Build avoid section
        avoid_text = ""
        if recent_exercises:
            avoid_text = "\n\nAVOID THESE RECENT EXERCISES (use different values/scenarios):\n"
            for i, ex in enumerate(recent_exercises[-5:], 1):
                avoid_text += f"{i}. {ex[:200]}...\n" if len(ex) > 200 else f"{i}. {ex}\n"

        approach_prompt = self.APPROACH_PROMPTS.get(
            learning_approach, self.APPROACH_PROMPTS["conceptual"]
        )

        prompt = f"""You are creating a review exercise for a student preparing for an exam.

CONCEPT: {knowledge_item_name}
TYPE: {approach_prompt}

REAL EXAM/PRACTICE EXAMPLES (study these carefully):
{primary_text}

{"ADDITIONAL CONTEXT:" + chr(10) + context_text if context_text else ""}
{avoid_text}

IGNORE ALL OF THE FOLLOWING IN EXAMPLES - these are NOT exercises:
- Exam headers (NOME, COGNOME, MATRICOLA, student ID fields)
- Exam instructions (rules about pens, calculators, paper)
- Page numbers, dates, professor names
- Any administrative text that isn't an actual question

LOOK FOR THE ACTUAL EXERCISE which typically:
- Asks to calculate, convert, explain, or analyze something
- Contains mathematical notation, formulas, or specific values
- Has a clear question or task to complete

CRITICAL REQUIREMENTS:
1. Generate in the SAME LANGUAGE as the examples above
2. Your exercise MUST match the complexity and style of ACTUAL exercises (not headers)
3. Use DIFFERENT numbers, variables, or scenarios - never copy
4. Include realistic edge cases or tricks that appear in real exams
5. If examples use specific notation/formatting, match it exactly
6. The exercise should take 2-5 minutes to solve (not trivial, not lengthy)
7. Match the EXACT difficulty level - if examples are hard, yours must be hard

DO NOT:
- Copy exam headers, instructions, or administrative text
- Generate trivial "textbook example" problems
- Make it easier than the real examples
- Use round/obvious numbers if examples don't
- Skip the tricky parts that make exam questions challenging
- Change the language from the examples

LaTeX: Use $...$ for inline math, $$...$$ for display math.

Return valid JSON:
{{
  "exercise_text": "The complete exercise with LaTeX math",
  "expected_answer": "Brief final answer + key steps (not verbose)",
  "exercise_type": "calculation|short_answer|explanation|scenario"
}}"""

        try:
            response = self._llm.generate(prompt, json_mode=True)
            # Handle LLMResponse object or string
            response_text = response.text if hasattr(response, "text") else str(response)
            return self._parse_exercise_response(response_text, knowledge_item_name)
        except Exception as e:
            # Fallback: create simple exercise
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
        exercise_type: str,
    ) -> ReviewEvaluation:
        """Evaluate student's answer to a review exercise.

        Args:
            exercise_text: The exercise question
            expected_answer: Expected solution
            student_answer: Student's submitted answer
            exercise_type: Type of exercise (calculation, explanation, etc.)

        Returns:
            ReviewEvaluation with score, feedback, and correct answer
        """
        prompt = f"""You are evaluating a student's answer to an exam review exercise.

EXERCISE:
{exercise_text}

EXPECTED ANSWER:
{expected_answer}

STUDENT'S ANSWER:
{student_answer}

EVALUATION RULES:

1. LANGUAGE: Evaluate in the same language as the exercise. If exercise is Italian, respond in Italian.

2. EQUIVALENT ANSWERS - Accept these as correct:
   - Different notation: λ=3, lambda=3, \\lambda=3 are equivalent
   - Different order: "3 and 5" = "5 and 3" for unordered answers
   - Simplified vs unsimplified: 2/4 = 1/2 = 0.5
   - Different units if convertible: 100cm = 1m
   - Minor typos in text answers if meaning is clear

3. PARTIAL CREDIT RULES:
   - 90-100%: Fully correct, possibly minor notation differences
   - 70-89%: Correct approach and final answer, minor errors in steps
   - 50-69%: Correct approach but wrong final answer, OR correct answer but wrong/missing steps
   - 30-49%: Partially correct approach, significant errors
   - 10-29%: Shows some understanding but mostly wrong
   - 0-9%: Completely wrong or blank

4. EXERCISE TYPE CONSIDERATIONS:
   - calculation: Steps matter. Correct answer with no steps = max 70%
   - explanation: Clarity and completeness matter. Key concepts must be mentioned.
   - short_answer: Focus on correctness of final answer
   - scenario: Reasoning matters as much as conclusion

5. BE FAIR:
   - Don't penalize for extra correct information
   - Don't penalize for different valid approaches
   - Don't penalize for formatting differences

Return valid JSON:
{{
  "score": 0.0-1.0,
  "is_correct": true/false (true if score >= 0.7),
  "feedback": "Brief explanation of what was right/wrong",
  "correct_answer": "The expected answer for reference"
}}"""

        try:
            response = self._llm.generate(prompt, json_mode=True)
            # Handle LLMResponse object or string
            response_text = response.text if hasattr(response, "text") else str(response)
            return self._parse_evaluation_response(response_text, expected_answer, student_answer)
        except Exception as e:
            # Fallback evaluation
            import logging

            logging.getLogger(__name__).warning(f"Answer evaluation failed: {e}")
            return self._fallback_evaluation(expected_answer, student_answer)

    def _format_examples(self, examples: list[ExerciseExample]) -> str:
        """Format exercise examples for the prompt."""
        if not examples:
            return "No examples provided."

        formatted = []
        for i, ex in enumerate(examples, 1):
            text = f"Example {i}:\n{ex.text}"
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
