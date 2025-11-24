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
from core.concept_explainer import ConceptExplainer
from core.study_strategies import StudyStrategyManager
from core.proof_tutor import ProofTutor


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
        self.concept_explainer = ConceptExplainer(llm_manager=self.llm, language=language)
        self.strategy_manager = StudyStrategyManager(language=language)
        self.proof_tutor = ProofTutor(llm_manager=self.llm, language=language)

    def learn(self, course_code: str,
              core_loop_id: str,
              explain_concepts: bool = True,
              depth: str = "medium",
              adaptive: bool = True,
              include_study_strategy: bool = False,
              show_solutions: bool = True) -> TutorResponse:
        """Explain a core loop with theory and procedure.

        Args:
            course_code: Course code
            core_loop_id: Core loop ID to learn
            explain_concepts: Whether to include prerequisite concepts (default: True)
            depth: Explanation depth - basic, medium, advanced (default: medium)
            adaptive: Enable adaptive teaching (auto-select depth and prerequisites based on mastery)
            include_study_strategy: Whether to include metacognitive study strategy (default: False)
            show_solutions: Whether to show official solutions for exercises (default: True)

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

            # Get example exercises (with solutions if available)
            exercises = db.get_exercises_by_course(course_code)
            examples = [ex for ex in exercises if ex.get('core_loop_id') == core_loop_id][:3]

            # Track exercises with solutions for later display
            exercises_with_solutions = []
            if show_solutions:
                for ex in examples:
                    if ex.get('solution') and ex.get('solution').strip():
                        exercises_with_solutions.append({
                            'exercise_number': ex.get('exercise_number', 'Unknown'),
                            'solution': ex.get('solution'),
                            'source_pdf': ex.get('source_pdf', '')
                        })

        core_loop_dict = dict(core_loop)

        # Check if this is a proof exercise (check first example)
        if examples and self.proof_tutor.is_proof_exercise(examples[0].get('text', '')):
            # Use proof-specific learning
            return self._learn_proof(course_code, core_loop_id, examples[0], explain_concepts, depth, adaptive)
        core_loop_name = core_loop_dict.get('name', '')

        # Adaptive teaching: Auto-select depth and prerequisites based on mastery
        adaptive_recommendations = None
        if adaptive:
            from core.adaptive_teaching import AdaptiveTeachingManager

            with AdaptiveTeachingManager() as atm:
                adaptive_recommendations = atm.get_adaptive_recommendations(
                    course_code, core_loop_name
                )

                # Override depth and explain_concepts if adaptive
                depth = adaptive_recommendations['depth']
                explain_concepts = adaptive_recommendations['show_prerequisites']

        # Get prerequisite concepts
        prerequisite_text = ""
        if explain_concepts:
            prerequisite_text = self.concept_explainer.explain_prerequisites(
                core_loop_name, depth=depth
            )

        # Build enhanced learning prompt with deep reasoning
        prompt = self._build_enhanced_learn_prompt(
            core_loop=core_loop_dict,
            examples=examples,
            depth=depth
        )

        # Call LLM for deep explanation
        response = self.llm.generate(
            prompt=prompt,
            model=self.llm.primary_model,
            temperature=0.3,
            max_tokens=3500  # Increased for deeper explanations
        )

        if not response.success:
            return TutorResponse(
                content=f"Failed to generate explanation: {response.error}",
                success=False
            )

        # Combine prerequisite concepts with LLM explanation
        full_content = []

        if prerequisite_text:
            full_content.append(prerequisite_text)
            full_content.append("\n" + "=" * 60 + "\n")

        full_content.append(response.text)

        # Add study strategy if requested
        if include_study_strategy:
            strategy = self.strategy_manager.get_strategy_for_core_loop(
                core_loop_name,
                difficulty=depth
            )
            if strategy:
                full_content.append("\n" + "=" * 60 + "\n")
                full_content.append(self.strategy_manager.format_strategy_output(strategy, core_loop_name))

        # Add adaptive recommendations at end
        if adaptive and adaptive_recommendations:
            full_content.append("\n" + "=" * 60 + "\n")
            full_content.append(self._format_adaptive_recommendations(adaptive_recommendations))

        # Add official solutions section if available
        if exercises_with_solutions:
            full_content.append("\n" + "=" * 60 + "\n")
            full_content.append(self._format_official_solutions(exercises_with_solutions))

        return TutorResponse(
            content="\n".join(full_content),
            success=True,
            metadata={
                "core_loop": core_loop_id,
                "examples_count": len(examples),
                "includes_prerequisites": explain_concepts,
                "depth": depth,
                "adaptive": adaptive,
                "recommendations": adaptive_recommendations,
                "includes_study_strategy": include_study_strategy,
                "has_solutions": len(exercises_with_solutions) > 0,
                "solutions_count": len(exercises_with_solutions)
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

    def _build_enhanced_learn_prompt(self, core_loop: Dict[str, Any],
                                     examples: List[Dict[str, Any]],
                                     depth: str = "medium") -> str:
        """Build enhanced prompt with deep WHY reasoning."""
        language_instruction = {
            "it": "Rispondi in ITALIANO.",
            "en": "Respond in ENGLISH."
        }

        depth_instructions = {
            "basic": "Keep explanations simple and concise. Focus on the core concepts.",
            "medium": "Provide balanced explanations with WHY reasoning and practical examples.",
            "advanced": "Give comprehensive explanations with deep reasoning, edge cases, and optimization strategies."
        }

        depth_instruction = depth_instructions.get(depth, depth_instructions["medium"])

        prompt = f"""{language_instruction.get(self.language, language_instruction["en"])}

You are an expert educator helping students DEEPLY understand a problem-solving procedure.

TOPIC: {core_loop.get('topic_name', 'Unknown')}
PROCEDURE: {core_loop['name']}
EXPLANATION DEPTH: {depth}

PROCEDURE STEPS:
{self._format_procedure(core_loop.get('procedure'))}

EXAMPLE EXERCISES:
{self._format_examples(examples)}

{depth_instruction}

Your task is to provide a COMPREHENSIVE, PEDAGOGICAL explanation that goes beyond just listing steps:

## 1. BIG PICTURE (The "What" and "Why it matters")
- What is this procedure solving?
- Why is this problem important?
- When would you use this in practice?
- What makes this approach effective?

## 2. STEP-BY-STEP BREAKDOWN (The "How" with reasoning)
For EACH step in the procedure, explain:
- **WHAT**: What are you doing in this step?
- **WHY**: Why is this step necessary? What problem does it solve?
- **HOW**: How do you perform this step concretely?
- **REASONING**: What's the underlying logic? Why does this method work?
- **VALIDATION**: How do you know you've done it correctly?

## 3. COMMON PITFALLS (Mistakes and how to avoid them)
- What mistakes do students typically make at each step?
- Why do these mistakes happen (what's the misconception)?
- How can you avoid or catch these mistakes?
- What are red flags that something went wrong?

## 4. DECISION-MAKING GUIDANCE (When and how to apply)
- When should you use this procedure (what signals/patterns)?
- When should you NOT use it (what are limitations)?
- How does this connect to related concepts?
- What variations or alternatives exist?

## 5. PRACTICE STRATEGY (How to master this)
- What's the best way to practice this skill?
- What should you focus on first?
- How can you test your understanding?
- What resources help deepen mastery?

Use:
- Concrete examples throughout (not just abstract descriptions)
- Analogies to familiar concepts when helpful
- "You" language to engage the student
- Clear structure with headers
- Progressive complexity (build from simple to complex)
- Practical tips from experience

Make your explanation conversational, engaging, and genuinely helpful for someone trying to learn this for the first time or deepen their understanding.
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

    def _format_adaptive_recommendations(self, recommendations: Dict[str, Any]) -> str:
        """Format adaptive teaching recommendations.

        Args:
            recommendations: Recommendations dictionary from AdaptiveTeachingManager

        Returns:
            Formatted string with recommendations
        """
        language_headers = {
            "it": {
                "title": "📚 RACCOMANDAZIONI DI STUDIO PERSONALIZZATE",
                "mastery": "Livello di padronanza attuale",
                "practice": "Esercizi consigliati per la pratica",
                "focus": "Aree su cui concentrarsi",
                "next_review": "Prossima revisione programmata"
            },
            "en": {
                "title": "📚 PERSONALIZED STUDY RECOMMENDATIONS",
                "mastery": "Current mastery level",
                "practice": "Recommended practice exercises",
                "focus": "Focus areas",
                "next_review": "Next scheduled review"
            }
        }

        headers = language_headers.get(self.language, language_headers["en"])
        lines = [f"\n{headers['title']}\n"]

        # Current mastery
        mastery = recommendations.get('current_mastery', 0.0)
        mastery_pct = int(mastery * 100)
        mastery_emoji = "🟢" if mastery >= 0.7 else "🟡" if mastery >= 0.3 else "🔴"
        lines.append(f"{mastery_emoji} {headers['mastery']}: {mastery_pct}%")

        # Practice recommendations
        practice_count = recommendations.get('practice_count', 3)
        lines.append(f"\n✍️  {headers['practice']}: {practice_count}")

        # Focus areas
        focus_areas = recommendations.get('focus_areas', [])
        if focus_areas:
            lines.append(f"\n🎯 {headers['focus']}:")
            for area in focus_areas:
                lines.append(f"   • {area}")

        # Next review
        next_review = recommendations.get('next_review')
        if next_review:
            from datetime import datetime
            try:
                review_date = datetime.fromisoformat(next_review)
                lines.append(f"\n📅 {headers['next_review']}: {review_date.strftime('%Y-%m-%d')}")
            except:
                pass

        return "\n".join(lines)

    def _format_official_solutions(self, exercises_with_solutions: List[Dict[str, Any]]) -> str:
        """Format official solutions section.

        Args:
            exercises_with_solutions: List of exercises with solutions

        Returns:
            Formatted string with solutions
        """
        language_headers = {
            "it": {
                "title": "SOLUZIONI UFFICIALI",
                "available": "Sono disponibili soluzioni ufficiali per alcuni esercizi di esempio:",
                "exercise": "Esercizio",
                "from": "da",
                "note": "Nota: Questa soluzione e stata estratta automaticamente dal PDF dell'esame."
            },
            "en": {
                "title": "OFFICIAL SOLUTIONS",
                "available": "Official solutions are available for some example exercises:",
                "exercise": "Exercise",
                "from": "from",
                "note": "Note: This solution was automatically extracted from the exam PDF."
            }
        }

        headers = language_headers.get(self.language, language_headers["en"])
        lines = [f"\n{headers['title']}\n"]
        lines.append(headers['available'])
        lines.append("")

        for i, ex_sol in enumerate(exercises_with_solutions, 1):
            exercise_num = ex_sol.get('exercise_number', f'#{i}')
            source_pdf = ex_sol.get('source_pdf', '')
            solution = ex_sol.get('solution', '')

            lines.append(f"{headers['exercise']} {exercise_num}")
            if source_pdf:
                lines.append(f"({headers['from']} {source_pdf})")
            lines.append("")
            lines.append(solution)
            lines.append("")
            lines.append(f"[{headers['note']}]")
            lines.append("")

            if i < len(exercises_with_solutions):
                lines.append("-" * 40)
                lines.append("")

        return "\n".join(lines)

    def _learn_proof(self, course_code: str, core_loop_id: str, example_exercise: Dict[str, Any],
                     explain_concepts: bool, depth: str, adaptive: bool) -> TutorResponse:
        """Handle proof-specific learning.

        Args:
            course_code: Course code
            core_loop_id: Core loop ID
            example_exercise: Example proof exercise
            explain_concepts: Whether to include prerequisites
            depth: Explanation depth
            adaptive: Whether to use adaptive teaching

        Returns:
            TutorResponse with proof explanation
        """
        exercise_text = example_exercise.get('text', '')
        exercise_id = example_exercise.get('id', '')

        # Get proof-specific explanation
        proof_explanation = self.proof_tutor.learn_proof(course_code, exercise_id, exercise_text)

        # Optionally add prerequisite concepts
        full_content = []

        if explain_concepts:
            # Extract core loop name for concept explanation
            with Database() as db:
                core_loop = db.conn.execute(
                    "SELECT name FROM core_loops WHERE id = ?",
                    (core_loop_id,)
                ).fetchone()
                if core_loop:
                    core_loop_name = core_loop['name']
                    prerequisite_text = self.concept_explainer.explain_prerequisites(
                        core_loop_name, depth=depth
                    )
                    if prerequisite_text:
                        full_content.append(prerequisite_text)
                        full_content.append("\n" + "=" * 60 + "\n")

        full_content.append(proof_explanation)

        return TutorResponse(
            content="\n".join(full_content),
            success=True,
            metadata={
                "core_loop": core_loop_id,
                "is_proof": True,
                "depth": depth,
                "includes_prerequisites": explain_concepts
            }
        )
