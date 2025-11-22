"""
Interactive AI tutor for Examina.
Provides learning, practice, and exercise generation features.
"""

import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from models.llm_manager import LLMManager
from storage.database import Database
from config import Config


@dataclass
class TutorResponse:
    """Response from tutor."""
    content: str
    success: bool
    metadata: Optional[Dict[str, Any]] = None


class Tutor:
    """AI tutor for learning core loops and practicing exercises."""

    def __init__(self, llm_manager: Optional[LLMManager] = None, language: str = "en"):
        """Initialize tutor.

        Args:
            llm_manager: LLM manager instance
            language: Output language ("en" or "it")
        """
        self.llm = llm_manager or LLMManager(provider="anthropic")
        self.language = language

    def learn(self, course_code: str, core_loop_id: str) -> TutorResponse:
        """Explain a core loop with theory and procedure.

        Args:
            course_code: Course code
            core_loop_id: Core loop ID to learn

        Returns:
            TutorResponse with explanation
        """
        with Database() as db:
            # Get core loop details
            core_loop = db.conn.execute("""
                SELECT cl.*, t.name as topic_name
                FROM core_loops cl
                JOIN topics t ON cl.topic_id = t.id
                WHERE cl.id = ? AND t.course_code = ?
            """, (core_loop_id, course_code)).fetchone()

            if not core_loop:
                return TutorResponse(
                    content="Core loop not found.",
                    success=False
                )

            # Get example exercises
            exercises = db.get_exercises_by_course(course_code)
            examples = [ex for ex in exercises if ex.get('core_loop_id') == core_loop_id][:3]

        # Build learning prompt
        prompt = self._build_learn_prompt(
            core_loop=dict(core_loop),
            examples=examples
        )

        # Call LLM
        response = self.llm.generate(
            prompt=prompt,
            model=self.llm.primary_model,
            temperature=0.3,
            max_tokens=2000
        )

        if not response.success:
            return TutorResponse(
                content=f"Failed to generate explanation: {response.error}",
                success=False
            )

        return TutorResponse(
            content=response.text,
            success=True,
            metadata={
                "core_loop": core_loop_id,
                "examples_count": len(examples)
            }
        )

    def practice(self, course_code: str, topic: Optional[str] = None,
                 difficulty: Optional[str] = None) -> TutorResponse:
        """Get a practice exercise.

        Args:
            course_code: Course code
            topic: Optional topic filter
            difficulty: Optional difficulty filter (easy|medium|hard)

        Returns:
            TutorResponse with exercise
        """
        with Database() as db:
            # Get exercises
            exercises = db.get_exercises_by_course(course_code)

            # Filter by topic if specified
            if topic:
                topic_id = db.conn.execute(
                    "SELECT id FROM topics WHERE course_code = ? AND name LIKE ?",
                    (course_code, f"%{topic}%")
                ).fetchone()

                if topic_id:
                    exercises = [ex for ex in exercises if ex.get('topic_id') == topic_id[0]]

            # Filter by difficulty if specified
            if difficulty:
                exercises = [ex for ex in exercises if ex.get('difficulty') == difficulty]

            # Filter out exercises without core loops
            exercises = [ex for ex in exercises if ex.get('core_loop_id')]

            if not exercises:
                return TutorResponse(
                    content="No exercises found matching criteria.",
                    success=False
                )

            # Pick random exercise
            exercise = random.choice(exercises)

        return TutorResponse(
            content=exercise['text'],
            success=True,
            metadata={
                "exercise_id": exercise['id'],
                "core_loop_id": exercise.get('core_loop_id'),
                "difficulty": exercise.get('difficulty'),
                "topic_id": exercise.get('topic_id')
            }
        )

    def check_answer(self, exercise_id: str, user_answer: str,
                    provide_hints: bool = False) -> TutorResponse:
        """Check user's answer and provide feedback.

        Args:
            exercise_id: Exercise ID
            user_answer: User's answer
            provide_hints: Whether to provide hints if wrong

        Returns:
            TutorResponse with feedback
        """
        with Database() as db:
            # Get exercise and core loop
            exercise = db.conn.execute("""
                SELECT e.*, cl.procedure, cl.name as core_loop_name
                FROM exercises e
                LEFT JOIN core_loops cl ON e.core_loop_id = cl.id
                WHERE e.id = ?
            """, (exercise_id,)).fetchone()

            if not exercise:
                return TutorResponse(
                    content="Exercise not found.",
                    success=False
                )

        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(
            exercise_text=exercise['text'],
            user_answer=user_answer,
            procedure=exercise['procedure'],
            provide_hints=provide_hints
        )

        # Call LLM
        response = self.llm.generate(
            prompt=prompt,
            model=self.llm.primary_model,
            temperature=0.3,
            max_tokens=1500
        )

        if not response.success:
            return TutorResponse(
                content=f"Failed to evaluate answer: {response.error}",
                success=False
            )

        return TutorResponse(
            content=response.text,
            success=True,
            metadata={
                "exercise_id": exercise_id,
                "has_hints": provide_hints
            }
        )

    def generate(self, course_code: str, core_loop_id: str,
                difficulty: str = "medium") -> TutorResponse:
        """Generate a new exercise variation.

        Args:
            course_code: Course code
            core_loop_id: Core loop to generate for
            difficulty: Difficulty level (easy|medium|hard)

        Returns:
            TutorResponse with new exercise
        """
        with Database() as db:
            # Get core loop
            core_loop = db.conn.execute("""
                SELECT cl.*, t.name as topic_name
                FROM core_loops cl
                JOIN topics t ON cl.topic_id = t.id
                WHERE cl.id = ? AND t.course_code = ?
            """, (core_loop_id, course_code)).fetchone()

            if not core_loop:
                return TutorResponse(
                    content="Core loop not found.",
                    success=False
                )

            # Get example exercises
            exercises = db.get_exercises_by_course(course_code)
            examples = [ex for ex in exercises if ex.get('core_loop_id') == core_loop_id][:5]

            if not examples:
                return TutorResponse(
                    content="No example exercises found for this core loop.",
                    success=False
                )

        # Build generation prompt
        prompt = self._build_generation_prompt(
            core_loop=dict(core_loop),
            examples=examples,
            difficulty=difficulty
        )

        # Call LLM
        response = self.llm.generate(
            prompt=prompt,
            model=self.llm.primary_model,
            temperature=0.7,  # Higher temperature for creativity
            max_tokens=1500
        )

        if not response.success:
            return TutorResponse(
                content=f"Failed to generate exercise: {response.error}",
                success=False
            )

        return TutorResponse(
            content=response.text,
            success=True,
            metadata={
                "core_loop": core_loop_id,
                "difficulty": difficulty,
                "based_on_examples": len(examples)
            }
        )

    def _build_learn_prompt(self, core_loop: Dict[str, Any],
                           examples: List[Dict[str, Any]]) -> str:
        """Build prompt for learning explanation."""
        language_instruction = {
            "it": "Rispondi in ITALIANO.",
            "en": "Respond in ENGLISH."
        }

        prompt = f"""{language_instruction.get(self.language, language_instruction["en"])}

You are an AI tutor helping students learn problem-solving procedures.

TOPIC: {core_loop.get('topic_name', 'Unknown')}
CORE PROCEDURE: {core_loop['name']}

SOLVING STEPS:
{self._format_procedure(core_loop.get('procedure'))}

EXAMPLE EXERCISES:
{self._format_examples(examples)}

Your task:
1. Explain the theory behind this procedure
2. Walk through each step with clear explanations
3. Show how to apply it using the examples
4. Provide tips and common mistakes to avoid

Make it pedagogical and clear for students learning this for the first time.
"""
        return prompt

    def _build_evaluation_prompt(self, exercise_text: str, user_answer: str,
                                 procedure: Optional[str], provide_hints: bool) -> str:
        """Build prompt for answer evaluation."""
        language_instruction = {
            "it": "Rispondi in ITALIANO.",
            "en": "Respond in ENGLISH."
        }

        hint_instruction = ""
        if provide_hints:
            hint_instruction = "\n3. Provide progressive hints to guide them toward the solution"

        prompt = f"""{language_instruction.get(self.language, language_instruction["en"])}

You are an AI tutor evaluating a student's answer.

EXERCISE:
{exercise_text}

STUDENT'S ANSWER:
{user_answer}

EXPECTED PROCEDURE:
{self._format_procedure(procedure)}

Your task:
1. Evaluate if the answer is correct or partially correct
2. Identify what's right and what's wrong{hint_instruction}
4. Be encouraging and constructive

Respond in a friendly, pedagogical tone.
"""
        return prompt

    def _build_generation_prompt(self, core_loop: Dict[str, Any],
                                 examples: List[Dict[str, Any]],
                                 difficulty: str) -> str:
        """Build prompt for exercise generation."""
        language_instruction = {
            "it": "Crea un esercizio in ITALIANO.",
            "en": "Create an exercise in ENGLISH."
        }

        prompt = f"""{language_instruction.get(self.language, language_instruction["en"])}

You are creating a new practice exercise.

PROCEDURE TO PRACTICE: {core_loop['name']}
TOPIC: {core_loop.get('topic_name', 'Unknown')}
DIFFICULTY: {difficulty}

SOLVING STEPS:
{self._format_procedure(core_loop.get('procedure'))}

EXAMPLE EXERCISES:
{self._format_examples(examples, limit=3)}

Your task:
Create a NEW exercise that:
1. Tests the same procedure/core loop
2. Has similar structure to the examples
3. Matches the requested difficulty level
4. Uses different numbers/scenarios than the examples
5. Is clear and well-formatted

Generate ONLY the exercise text, not the solution.
"""
        return prompt

    def _format_procedure(self, procedure: Optional[str]) -> str:
        """Format procedure steps for display."""
        if not procedure:
            return "No procedure available."

        try:
            import json
            steps = json.loads(procedure)
            if isinstance(steps, list):
                return "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
        except:
            pass

        return str(procedure)

    def _format_examples(self, examples: List[Dict[str, Any]], limit: int = 3) -> str:
        """Format example exercises."""
        if not examples:
            return "No examples available."

        formatted = []
        for i, ex in enumerate(examples[:limit], 1):
            text = ex['text'][:300] + "..." if len(ex['text']) > 300 else ex['text']
            formatted.append(f"Example {i}:\n{text}\n")

        return "\n".join(formatted)
