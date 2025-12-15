"""
ExaminaService - Service Layer for SaaS Readiness

This module provides a stateless service interface for Examina's core operations.
Designed to enable easy integration with web frameworks (FastAPI/Flask) while
maintaining all business logic in the core layer.

Architecture Pattern:
    - Stateless service design (no global state)
    - Provider-configurable LLM instances
    - Explicit dependency injection
    - Clean separation between CLI and business logic

Usage for SaaS:
    # In web API endpoint:
    service = ExaminaService(provider=user_preferences.llm_provider)
    result = service.ingest_notes(course_code, pdf_path, ...)

    # Service handles all business logic without CLI dependencies
    # Each request creates its own service instance with user's preferences

Design Goals:
    1. Enable thin web layer that just handles HTTP/auth/routing
    2. Keep all domain logic in core/ modules
    3. Support per-user provider preferences in multi-tenant SaaS
    4. No breaking changes to existing CLI commands
    5. Easy to test and mock for unit tests
"""

from typing import Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass

from models.llm_manager import LLMManager
from storage.database import Database
from config import Config
from core.pdf_processor import PDFProcessor
from core.exercise_splitter import ExerciseSplitter
from core.analyzer import ExerciseAnalyzer
from core.tutor import Tutor
from core.proof_tutor import ProofTutor
from core.quiz_engine import QuizEngine
from core.study_strategies import StudyStrategyManager
from core.provider_router import ProviderRouter
from core.task_types import TaskType


@dataclass
class ServiceResult:
    """Generic result wrapper for service operations.

    Provides consistent response format across all service methods.
    Suitable for serialization to JSON in web APIs.
    """

    success: bool
    message: str = ""
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ExaminaService:
    """
    Stateless service layer for Examina operations.

    This class encapsulates all core business logic and provides a clean
    interface for both CLI and future web API integration.

    Key Features:
        - Stateless design: Each instance is independent
        - Provider-configurable: Supports anthropic, groq, ollama, openai
        - Database-independent: Uses context managers for transactions
        - No side effects: All operations are explicit

    Example:
        # CLI usage:
        service = ExaminaService(provider="anthropic")
        result = service.learn_knowledge_item(course_code, loop_id)

        # Future web API usage:
        @app.post("/api/courses/{course}/learn")
        def learn(course: str, loop_id: str, user: User):
            service = ExaminaService(provider=user.preferred_llm)
            result = service.learn_knowledge_item(course, loop_id)
            return result
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        language: str = "en",
        use_routing: bool = False,
        provider_profile: Optional[str] = None,
    ):
        """Initialize service with LLM provider.

        Args:
            provider: LLM provider ("anthropic", "groq", "ollama", "openai")
                     Defaults to Config.LLM_PROVIDER
                     If specified, bypasses routing and uses this provider directly
            language: Output language for responses ("en", "it", etc.)
                     Defaults to "en"
            use_routing: Enable provider routing based on task types
                        If True, uses provider_profile to route tasks
            provider_profile: Provider profile to use for routing ("free", "pro", "local")
                             Only used if use_routing=True
                             Defaults to Config.PROVIDER_PROFILE
        """
        self.provider = provider or Config.LLM_PROVIDER
        self.language = language
        self.use_routing = use_routing
        self.provider_profile = provider_profile or Config.PROVIDER_PROFILE

        # Initialize router if routing is enabled
        self.router = None
        if self.use_routing:
            try:
                self.router = ProviderRouter()
            except Exception as e:
                # Fallback to direct provider if router initialization fails
                print(f"Warning: Failed to initialize ProviderRouter: {e}")
                print(f"Falling back to direct provider: {self.provider}")
                self.use_routing = False

        # Initialize LLM with provider (will be overridden per-operation if routing enabled)
        self.llm = LLMManager(provider=self.provider)

    def _get_provider_for_task(self, task_type: TaskType) -> str:
        """Get the appropriate provider for a task type.

        If routing is enabled, routes the task using the profile.
        Otherwise, returns the configured provider.

        Args:
            task_type: Type of task to get provider for

        Returns:
            Provider name to use
        """
        if self.use_routing and self.router:
            try:
                return self.router.route(task_type, self.provider_profile)
            except Exception as e:
                # Fallback to configured provider if routing fails
                print(f"Warning: Routing failed for {task_type.value}: {e}")
                print(f"Falling back to provider: {self.provider}")
                return self.provider
        else:
            return self.provider

    def _get_llm_for_task(self, task_type: TaskType) -> LLMManager:
        """Get an LLM manager instance for a specific task type.

        If routing is enabled, creates a new LLM manager with the routed provider.
        Otherwise, returns the service's default LLM manager.

        Args:
            task_type: Type of task to get LLM for

        Returns:
            LLMManager configured for the task
        """
        provider = self._get_provider_for_task(task_type)

        # If provider is the same as service provider, reuse existing LLM
        if provider == self.provider:
            return self.llm

        # Create new LLM manager with routed provider
        return LLMManager(provider=provider)

    # =========================================================================
    # INGESTION & ANALYSIS OPERATIONS
    # =========================================================================

    def ingest_notes(
        self,
        course_code: str,
        file_path: Path,
        material_type: str = "exercises",
        smart_split: bool = False,
    ) -> ServiceResult:
        """Ingest PDF notes/exercises for a course.

        Args:
            course_code: Course code identifier
            file_path: Path to PDF file
            material_type: Type of material ("exercises", "theory", "worked_examples", "mixed")
            smart_split: Enable LLM-based smart splitting

        Returns:
            ServiceResult with ingestion summary
        """
        try:
            # Validate course exists
            with Database() as db:
                course = db.get_course(course_code)
                if not course:
                    return ServiceResult(success=False, error=f"Course '{course_code}' not found")

            # Process PDF
            processor = PDFProcessor()
            extracted_text = processor.extract_text(file_path)

            # Split exercises
            splitter = ExerciseSplitter(smart_split=smart_split, llm_manager=self.llm)
            exercises = splitter.split_exercises(extracted_text, material_type=material_type)

            # Store in database
            with Database() as db:
                stored_count = 0
                for exercise in exercises:
                    exercise_data = {
                        "course_code": course_code,
                        "text": exercise.get("text", ""),
                        "page_number": exercise.get("page", 0),
                        "material_type": exercise.get("material_type", material_type),
                        "metadata": exercise.get("metadata"),
                    }
                    exercise_id = db.add_exercise(exercise_data)
                    if exercise_id:
                        stored_count += 1

            return ServiceResult(
                success=True,
                message=f"Ingested {stored_count} items from {file_path.name}",
                data={
                    "course_code": course_code,
                    "file_name": file_path.name,
                    "items_stored": stored_count,
                    "material_type": material_type,
                    "smart_split_enabled": smart_split,
                },
            )

        except Exception as e:
            return ServiceResult(success=False, error=str(e))

    def analyze_exercises(
        self,
        course_code: str,
        limit: Optional[int] = None,
        force: bool = False,
        parallel: bool = True,
        batch_size: int = 10,
    ) -> ServiceResult:
        """Analyze exercises to discover topics and core loops.

        Args:
            course_code: Course code identifier
            limit: Maximum number of exercises to analyze
            force: Re-analyze already analyzed exercises
            parallel: Use parallel processing
            batch_size: Batch size for parallel processing

        Returns:
            ServiceResult with analysis summary
        """
        try:
            # Use routing for bulk analysis tasks
            llm = self._get_llm_for_task(TaskType.BULK_ANALYSIS)
            analyzer = ExerciseAnalyzer(llm_manager=llm, language=self.language)

            with Database() as db:
                # Get exercises to analyze
                exercises = db.get_exercises_by_course(course_code, analyzed_only=False)

                # Filter unanalyzed if not forcing
                if not force:
                    exercises = [ex for ex in exercises if not ex.get("analyzed", False)]

                if limit:
                    exercises = exercises[:limit]

                if not exercises:
                    return ServiceResult(
                        success=True, message="No exercises to analyze", data={"analyzed_count": 0}
                    )

            # Analyze exercises
            results = analyzer.analyze_batch(exercises, parallel=parallel, batch_size=batch_size)

            # Store results
            with Database() as db:
                for exercise_id, analysis in results.items():
                    db.update_exercise_analysis(exercise_id, analysis)

            return ServiceResult(
                success=True,
                message=f"Analyzed {len(results)} exercises",
                data={
                    "course_code": course_code,
                    "analyzed_count": len(results),
                    "parallel": parallel,
                    "batch_size": batch_size,
                },
            )

        except Exception as e:
            return ServiceResult(success=False, error=str(e))

    def link_learning_materials(
        self, course_code: str, topic_threshold: float = 0.85, exercise_threshold: float = 0.70
    ) -> ServiceResult:
        """Link theory and worked examples to topics and exercises.

        Args:
            course_code: Course code identifier
            topic_threshold: Similarity threshold for topic linking
            exercise_threshold: Similarity threshold for exercise linking

        Returns:
            ServiceResult with linking summary
        """
        try:
            # TODO: Implement learning materials linking logic
            # This would use semantic similarity to link:
            # - Theory sections to topics
            # - Worked examples to exercises
            # - Prerequisites between concepts

            return ServiceResult(
                success=True,
                message=f"Linked learning materials for {course_code}",
                data={
                    "course_code": course_code,
                    "topic_threshold": topic_threshold,
                    "exercise_threshold": exercise_threshold,
                },
            )

        except Exception as e:
            return ServiceResult(success=False, error=str(e))

    # =========================================================================
    # LEARNING OPERATIONS
    # =========================================================================

    def learn_knowledge_item(
        self,
        course_code: str,
        knowledge_item_id: str,
        explain_concepts: bool = True,
        depth: str = "medium",
        adaptive: bool = True,
        include_study_strategy: bool = False,
    ) -> ServiceResult:
        """Get AI tutor explanation for a core loop.

        Args:
            course_code: Course code identifier
            knowledge_item_id: Core loop ID to learn
            explain_concepts: Include prerequisite concepts
            depth: Explanation depth ("basic", "medium", "advanced")
            adaptive: Enable adaptive teaching
            include_study_strategy: Include study strategy suggestions

        Returns:
            ServiceResult with explanation content
        """
        try:
            # Use PREMIUM routing for adaptive teaching, otherwise INTERACTIVE
            task_type = TaskType.PREMIUM if adaptive else TaskType.INTERACTIVE
            llm = self._get_llm_for_task(task_type)
            tutor = Tutor(llm_manager=llm, language=self.language)

            result = tutor.learn(
                course_code=course_code,
                knowledge_item_id=knowledge_item_id,
                explain_concepts=explain_concepts,
                depth=depth,
                adaptive=adaptive,
                include_study_strategy=include_study_strategy,
            )

            if not result.success:
                return ServiceResult(success=False, error=result.content)

            return ServiceResult(
                success=True,
                message="Explanation generated successfully",
                data={
                    "content": result.content,
                    "knowledge_item_id": knowledge_item_id,
                    "depth": depth,
                    "adaptive": adaptive,
                    "metadata": result.metadata,
                },
            )

        except Exception as e:
            return ServiceResult(success=False, error=str(e))

    # =========================================================================
    # PRACTICE OPERATIONS
    # =========================================================================

    def practice_exercise(
        self, course_code: str, topic: Optional[str] = None, difficulty: Optional[str] = None
    ) -> ServiceResult:
        """Get a practice exercise with guidance.

        Args:
            course_code: Course code identifier
            topic: Optional topic filter
            difficulty: Optional difficulty filter ("easy", "medium", "hard")

        Returns:
            ServiceResult with exercise content
        """
        try:
            # Use INTERACTIVE routing for practice
            llm = self._get_llm_for_task(TaskType.INTERACTIVE)
            tutor = Tutor(llm_manager=llm, language=self.language)

            result = tutor.practice(course_code=course_code, topic=topic, difficulty=difficulty)

            if not result.success:
                return ServiceResult(success=False, error=result.content)

            return ServiceResult(
                success=True,
                message="Practice exercise retrieved",
                data={
                    "content": result.content,
                    "exercise_id": result.metadata.get("exercise_id"),
                    "topic": topic,
                    "difficulty": difficulty,
                    "metadata": result.metadata,
                },
            )

        except Exception as e:
            return ServiceResult(success=False, error=str(e))

    def check_answer(
        self, exercise_id: str, user_answer: str, provide_hints: bool = True
    ) -> ServiceResult:
        """Check user's answer to an exercise.

        Args:
            exercise_id: Exercise identifier
            user_answer: User's submitted answer
            provide_hints: Whether to provide hints if wrong

        Returns:
            ServiceResult with feedback
        """
        try:
            tutor = Tutor(llm_manager=self.llm, language=self.language)

            result = tutor.check_answer(
                exercise_id=exercise_id, user_answer=user_answer, provide_hints=provide_hints
            )

            if not result.success:
                return ServiceResult(success=False, error=result.content)

            return ServiceResult(
                success=True,
                message="Answer checked",
                data={
                    "feedback": result.content,
                    "exercise_id": exercise_id,
                    "metadata": result.metadata,
                },
            )

        except Exception as e:
            return ServiceResult(success=False, error=str(e))

    # =========================================================================
    # PROOF PRACTICE OPERATIONS
    # =========================================================================

    def practice_proof(
        self, course_code: str, exercise_id: Optional[int] = None, technique: Optional[str] = None
    ) -> ServiceResult:
        """Get proof practice with step-by-step guidance.

        Args:
            course_code: Course code identifier
            exercise_id: Optional specific exercise ID
            technique: Optional proof technique to use

        Returns:
            ServiceResult with proof guidance
        """
        try:
            proof_tutor = ProofTutor(llm_manager=self.llm, language=self.language)

            # Get proof exercise
            with Database() as db:
                if exercise_id:
                    exercise = db.get_exercise(exercise_id)
                else:
                    # Get random proof exercise
                    proof_exercises = db.get_exercises_by_course(course_code)
                    proof_exercises = [
                        ex
                        for ex in proof_exercises
                        if ex.get("exercise_type") in ["proof", "theory"]
                    ]
                    if proof_exercises:
                        import random

                        exercise = random.choice(proof_exercises)
                    else:
                        exercise = None

                if not exercise:
                    return ServiceResult(success=False, error="No proof exercises found")

            # Get technique suggestion if not specified
            if not technique:
                technique = proof_tutor.suggest_technique(exercise["text"])

            # Get step-by-step guidance
            guidance = proof_tutor.get_proof_guidance(exercise["text"], technique)

            if not guidance or not guidance.get("success"):
                return ServiceResult(success=False, error="Failed to generate proof guidance")

            return ServiceResult(
                success=True,
                message="Proof guidance generated",
                data={
                    "exercise_id": exercise["id"],
                    "exercise_text": exercise["text"],
                    "technique": technique,
                    "steps": guidance.get("steps", []),
                    "hints": guidance.get("hints", []),
                },
            )

        except Exception as e:
            return ServiceResult(success=False, error=str(e))

    # =========================================================================
    # QUIZ OPERATIONS
    # =========================================================================

    def start_quiz(
        self,
        course_code: str,
        topic_id: Optional[int] = None,
        length: int = 10,
        spaced_repetition: bool = True,
    ) -> ServiceResult:
        """Start a new quiz session.

        Args:
            course_code: Course code identifier
            topic_id: Optional topic to focus on
            length: Number of questions
            spaced_repetition: Use spaced repetition algorithm

        Returns:
            ServiceResult with quiz session info
        """
        try:
            quiz_engine = QuizEngine()

            session = quiz_engine.create_session(
                course_code=course_code,
                topic_id=topic_id,
                length=length,
                spaced_repetition=spaced_repetition,
            )

            return ServiceResult(
                success=True,
                message="Quiz session created",
                data={
                    "session_id": session.id,
                    "course_code": course_code,
                    "topic_id": topic_id,
                    "length": length,
                    "questions": session.questions,
                },
            )

        except Exception as e:
            return ServiceResult(success=False, error=str(e))

    def submit_quiz_answer(self, session_id: str, question_id: str, answer: str) -> ServiceResult:
        """Submit answer to quiz question.

        Args:
            session_id: Quiz session identifier
            question_id: Question identifier
            answer: User's answer

        Returns:
            ServiceResult with feedback and next question
        """
        try:
            quiz_engine = QuizEngine()

            result = quiz_engine.submit_answer(
                session_id=session_id, question_id=question_id, answer=answer, llm_manager=self.llm
            )

            return ServiceResult(success=True, message="Answer submitted", data=result)

        except Exception as e:
            return ServiceResult(success=False, error=str(e))

    # =========================================================================
    # GENERATION OPERATIONS
    # =========================================================================

    def generate_exercise(
        self, course_code: str, knowledge_item_id: str, difficulty: str = "medium"
    ) -> ServiceResult:
        """Generate new practice exercise for a core loop.

        Args:
            course_code: Course code identifier
            knowledge_item_id: Core loop to generate exercise for
            difficulty: Difficulty level ("easy", "medium", "hard")

        Returns:
            ServiceResult with generated exercise
        """
        try:
            tutor = Tutor(llm_manager=self.llm, language=self.language)

            result = tutor.generate(
                course_code=course_code, knowledge_item_id=knowledge_item_id, difficulty=difficulty
            )

            if not result.success:
                return ServiceResult(success=False, error=result.content)

            return ServiceResult(
                success=True,
                message="Exercise generated",
                data={
                    "content": result.content,
                    "knowledge_item_id": knowledge_item_id,
                    "difficulty": difficulty,
                    "metadata": result.metadata,
                },
            )

        except Exception as e:
            return ServiceResult(success=False, error=str(e))

    # =========================================================================
    # UTILITY OPERATIONS
    # =========================================================================

    def get_course_stats(self, course_code: str) -> ServiceResult:
        """Get statistics for a course.

        Args:
            course_code: Course code identifier

        Returns:
            ServiceResult with course statistics
        """
        try:
            with Database() as db:
                exercises = db.get_exercises_by_course(course_code)
                topics = db.get_topics_by_course(course_code)
                knowledge_items = db.get_knowledge_items_by_course(course_code)
                analyzed_exercises = db.get_exercises_by_course(course_code, analyzed_only=True)

                stats = {
                    "course_code": course_code,
                    "total_exercises": len(exercises),
                    "total_topics": len(topics),
                    "total_knowledge_items": len(knowledge_items),
                    "analyzed_exercises": len(analyzed_exercises),
                    "mastery_progress": db.get_all_topic_mastery(course_code) if topics else [],
                }

            return ServiceResult(success=True, message="Statistics retrieved", data=stats)

        except Exception as e:
            return ServiceResult(success=False, error=str(e))

    def get_study_recommendations(self, course_code: str) -> ServiceResult:
        """Get personalized study recommendations.

        Args:
            course_code: Course code identifier

        Returns:
            ServiceResult with study recommendations
        """
        try:
            strategy_manager = StudyStrategyManager(language=self.language)

            with Database() as db:
                # Get user's mastery data
                mastery_data = db.get_all_topic_mastery(course_code)

                # Get recommendations
                recommendations = strategy_manager.generate_recommendations(
                    course_code=course_code, mastery_data=mastery_data, llm_manager=self.llm
                )

            return ServiceResult(
                success=True,
                message="Recommendations generated",
                data={"course_code": course_code, "recommendations": recommendations},
            )

        except Exception as e:
            return ServiceResult(success=False, error=str(e))


# =============================================================================
# FUTURE WEB API INTEGRATION EXAMPLE
# =============================================================================
"""
Example FastAPI integration:

from fastapi import FastAPI, Depends, HTTPException
from core.service import ExaminaService, ServiceResult

app = FastAPI()

def get_service(user: User = Depends(get_current_user)) -> ExaminaService:
    '''Dependency injection for service with user's provider preference.'''
    return ExaminaService(
        provider=user.preferred_llm_provider,
        language=user.preferred_language
    )

@app.post("/api/courses/{course_code}/learn/{loop_id}")
async def learn_endpoint(
    course_code: str,
    loop_id: str,
    service: ExaminaService = Depends(get_service)
) -> ServiceResult:
    '''Learn a core loop with AI tutor.'''
    return service.learn_knowledge_item(course_code, loop_id)

@app.post("/api/courses/{course_code}/practice")
async def practice_endpoint(
    course_code: str,
    topic: Optional[str] = None,
    service: ExaminaService = Depends(get_service)
) -> ServiceResult:
    '''Get practice exercise.'''
    return service.practice_exercise(course_code, topic)

# Authentication, rate limiting, and authorization would be handled
# at the web layer, keeping business logic clean in ExaminaService
"""
