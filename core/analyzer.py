"""
AI-powered exercise analyzer for Examina.
Extracts knowledge items from exercises for spaced repetition learning.
"""

import json
import logging
import re
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed

from models.llm_manager import LLMManager, LLMResponse
from storage.database import Database
from config import Config

logger = logging.getLogger(__name__)

# Knowledge item types - what kind of knowledge is being tested
KNOWLEDGE_TYPES = [
    "procedure",     # Step-by-step method to solve a problem
    "algorithm",     # Formal algorithm with specific steps
    "definition",    # Formal definition of a concept
    "theorem",       # Mathematical/scientific theorem
    "proof",         # Proof of a theorem or property
    "derivation",    # Derivation of a formula or result
    "formula",       # Mathematical formula to remember
    "fact",          # Factual information to memorize
    "key_concept",   # Important concept to understand
]

# Learning approaches - how to best teach this knowledge
LEARNING_APPROACHES = [
    "procedural",    # Step-by-step problem solving
    "conceptual",    # Understanding principles and "why"
    "factual",       # Memorizing facts/terminology
    "analytical",    # Critical thinking, evaluating evidence
]

# Type checking imports (avoid circular dependencies)
if TYPE_CHECKING:
    from core.procedure_cache import ProcedureCache, CacheHit


@dataclass
class KnowledgeItemInfo:
    """Unified knowledge item extracted from exercise analysis.

    The primary format for knowledge extraction. Replaces ProcedureInfo.
    """
    name: str  # snake_case identifier
    knowledge_type: Optional[str] = None  # DEPRECATED - no longer extracted, will be removed
    learning_approach: Optional[str] = None  # procedural, conceptual, factual, analytical


@dataclass
class AnalysisResult:
    """Result of exercise analysis."""
    is_valid_exercise: bool
    is_fragment: bool
    should_merge_with_previous: bool
    difficulty: Optional[str]
    confidence: float
    knowledge_items: Optional[List['KnowledgeItemInfo']] = None

    @staticmethod
    def _normalize_name(name: Optional[str]) -> Optional[str]:
        """Normalize knowledge item name to snake_case ID."""
        if not name:
            return None
        normalized = name.lower()
        normalized = re.sub(r'[^\w\s-]', '', normalized)
        normalized = re.sub(r'[\s-]+', '_', normalized)
        return normalized


class ExerciseAnalyzer:
    """Analyzes exercises using LLM to discover topics and core loops."""

    def __init__(self, llm_manager: Optional[LLMManager] = None, language: str = "en",
                 monolingual: bool = False, procedure_cache: Optional['ProcedureCache'] = None):
        """Initialize analyzer.

        Args:
            llm_manager: LLM manager instance
            language: Output language for analysis (any ISO 639-1 code, e.g., "en", "de", "zh")
            monolingual: Enable strictly monolingual mode (all procedures in single language)
            procedure_cache: Optional ProcedureCache instance for pattern caching (Option 3 optimization)
        """
        self.llm = llm_manager or LLMManager()
        self.language = language
        self.monolingual = monolingual
        self.primary_language = None  # Will be detected from course metadata/exercises

        # Option 3: Procedure Pattern Caching
        self.procedure_cache = procedure_cache
        self.cache_stats = {'hits': 0, 'misses': 0}

        # Initialize translation detector for monolingual mode
        self.translation_detector = None
        if self.monolingual:
            try:
                from core.translation_detector import TranslationDetector
                self.translation_detector = TranslationDetector(llm_manager=self.llm)
                print("[INFO] Translation detector initialized for monolingual mode")
            except Exception as e:
                print(f"[WARNING] Failed to initialize TranslationDetector for monolingual mode: {e}")
                print("  Monolingual mode will be disabled")
                self.monolingual = False  # Disable if can't initialize

        # SemanticMatcher removed - using string-based similarity
        self.semantic_matcher = None
        self.use_semantic = False

    def _language_instruction(self, action: str = "Respond") -> str:
        """Generate dynamic language instruction for any language."""
        return f"{action} in {self.language.upper()} language."

    def _lang_instruction(self) -> str:
        """Generate language instruction phrase for any language."""
        return f"in {self.language.upper()} language"

    def _language_name(self) -> str:
        """Get full language name for prompts."""
        return self.language.upper()

    def analyze_exercise(self, exercise_text: str, course_name: str,
                        exercise_context: Optional[str] = None,
                        is_sub_question: bool = False) -> AnalysisResult:
        """Analyze a single exercise.

        Args:
            exercise_text: Exercise text
            course_name: Course name for context
            exercise_context: Optional context (parent context for subs, exercise summary for standalone)
            is_sub_question: Whether this is a sub-question (affects prompt wording)

        Returns:
            AnalysisResult with classification
        """
        # Build prompt
        prompt = self._build_analysis_prompt(
            exercise_text, course_name, exercise_context, is_sub_question
        )

        # Call LLM
        response = self.llm.generate(
            prompt=prompt,
            model=self.llm.primary_model,
            temperature=0.3,  # Lower temp for more consistent analysis
            json_mode=True
        )

        if not response.success:
            # Log error and return default
            print(f"[ERROR] LLM failed for exercise: {response.error}")
            print(f"  Text preview: {exercise_text[:100]}...")
            return self._default_analysis_result()

        # Parse JSON response
        data = self.llm.parse_json_response(response)
        if not data:
            return self._default_analysis_result()

        # Parse knowledge_item (ONE per exercise)
        knowledge_items = []
        if "knowledge_item" in data and data["knowledge_item"]:
            item_data = data["knowledge_item"]
            knowledge_items.append(KnowledgeItemInfo(
                name=item_data.get("name", "unknown"),
                knowledge_type=None,  # DEPRECATED - no longer extracted
                learning_approach=item_data.get("learning_approach"),
            ))

        return AnalysisResult(
            is_valid_exercise=data.get("is_valid_exercise", True),
            is_fragment=data.get("is_fragment", False),
            should_merge_with_previous=data.get("should_merge_with_previous", False),
            difficulty=data.get("difficulty"),
            confidence=data.get("confidence", 0.5),
            knowledge_items=knowledge_items if knowledge_items else None,
        )

    def _build_analysis_prompt(self, exercise_text: str, course_name: str,
                               exercise_context: Optional[str] = None,
                               is_sub_question: bool = False) -> str:
        """Build prompt for exercise analysis.

        Args:
            exercise_text: Exercise text
            course_name: Course name
            exercise_context: Optional context (parent context for subs, exercise summary for standalone)
            is_sub_question: Whether this is a sub-question (affects prompt wording)

        Returns:
            Prompt string
        """
        base_prompt = f"""You are analyzing exam exercises for the course: {course_name}.

Extract the knowledge item (core skill/concept) being tested.

EXERCISE TEXT:
```
{exercise_text[:2000]}
```
"""

        if exercise_context:
            if is_sub_question:
                base_prompt += f"""
PARENT CONTEXT (background only):
```
{exercise_context[:1000]}
```

This is background context. Name the knowledge item based on what THIS SUB-QUESTION specifically asks, not the parent context.
"""
            else:
                base_prompt += f"""
EXERCISE SUMMARY:
```
{exercise_context[:1000]}
```

This summarizes the exercise. Use it to understand the context, but name the knowledge item based on the core skill being tested.
"""

        # Build learning approach options from constants
        learning_approaches_str = "|".join(LEARNING_APPROACHES)

        # Add language instruction (supports any ISO 639-1 language)
        base_prompt += f"""
IMPORTANT: {self._language_instruction("Respond")} All names must be in {self._language_name()} language.

Respond in JSON format:
{{
  "difficulty": "easy|medium|hard",
  "confidence": 0.0-1.0,
  "knowledge_item": {{
    "name": "snake_case_name",  // e.g., "base_conversion_binary", "fsm_design"
    "learning_approach": "{learning_approaches_str}"
  }}
}}

KNOWLEDGE ITEM (the ONE core skill being tested):
- Ask: "If a student fails this exercise, what specific skill are they missing?"
- Ask: "What would this exercise be called in a study guide?"
- Name the CONCEPT being tested, not the task performed
- Think: "What textbook chapter covers this?"
- Name should make sense outside this exercise context
- snake_case, e.g., "matrix_multiplication", "contract_formation_elements", "differential_diagnosis"
- If multiple concepts, pick the primary one

LEARNING APPROACH:
- procedural = APPLY steps/calculate/solve/design
- conceptual = EXPLAIN/compare/reason why
- factual = RECALL facts/definitions/formulas
- analytical = ANALYZE/evaluate/critique/prove

CONTEXT EXCLUSION:
- Extract ONLY course concepts, NOT word problem scenarios
- Test: "Does this have a formal definition/procedure in the course, or just scenery?"

Respond ONLY with valid JSON.
"""

        return base_prompt

    def _normalize_knowledge_item_id(self, knowledge_item_name: Optional[str]) -> Optional[str]:
        """Normalize core loop name to ID.

        Args:
            knowledge_item_name: Human-readable name

        Returns:
            Normalized ID (lowercase, underscores)
        """
        if not knowledge_item_name:
            return None

        # Convert to lowercase, replace spaces with underscores
        knowledge_item_id = knowledge_item_name.lower()
        knowledge_item_id = re.sub(r'[^\w\s-]', '', knowledge_item_id)
        knowledge_item_id = re.sub(r'[\s-]+', '_', knowledge_item_id)

        return knowledge_item_id

    def _default_analysis_result(self) -> AnalysisResult:
        """Return default analysis result on error."""
        return AnalysisResult(
            is_valid_exercise=True,
            is_fragment=False,
            should_merge_with_previous=False,
            difficulty=None,
            confidence=0.0,
            knowledge_items=None,
        )

    def _build_result_from_cache(self, cache_hit: 'CacheHit', exercise_text: str) -> AnalysisResult:
        """Build AnalysisResult from cache hit (Option 3: Procedure Pattern Caching).

        Args:
            cache_hit: Cache hit result containing cached knowledge items
            exercise_text: Original exercise text

        Returns:
            AnalysisResult built from cached data
        """
        # Convert cached data to KnowledgeItemInfo
        knowledge_items = []
        if cache_hit.knowledge_items:
            for ki in cache_hit.knowledge_items:
                knowledge_items.append(KnowledgeItemInfo(
                    name=ki.get('name', 'unknown'),
                    knowledge_type=None,  # DEPRECATED - no longer extracted
                    learning_approach=ki.get('learning_approach'),
                ))

        return AnalysisResult(
            is_valid_exercise=True,
            is_fragment=False,
            should_merge_with_previous=False,
            difficulty=cache_hit.difficulty,
            confidence=cache_hit.confidence,
            knowledge_items=knowledge_items if knowledge_items else None,
        )

    def _detect_primary_language(self, exercises: List[Dict[str, Any]], course_name: str) -> str:
        """Detect the primary language of the course.

        Args:
            exercises: List of exercise dicts
            course_name: Course name for additional context

        Returns:
            Primary language code (e.g., "english", "italian")
        """
        if not self.translation_detector:
            # Fallback to analysis language if no detector (supports any language)
            return self.language

        # Sample first few exercises to detect language
        sample_size = min(5, len(exercises))
        language_counts = {}

        for exercise in exercises[:sample_size]:
            text = exercise.get('text', '')
            if not text:
                continue

            detected_lang = self.translation_detector.detect_language(text)
            if detected_lang and detected_lang != "unknown":
                language_counts[detected_lang] = language_counts.get(detected_lang, 0) + 1

        # Get most common language
        if language_counts:
            primary_lang = max(language_counts, key=language_counts.get)
            print(f"[INFO] Detected primary course language: {primary_lang} (from {sample_size} exercises)")
            return primary_lang
        else:
            # Fallback to analysis language (supports any language)
            print(f"[INFO] Could not detect language, using fallback: {self.language}")
            return self.language

    def merge_exercises(self, exercises: List[Dict[str, Any]], skip_analyzed: bool = False) -> List[Dict[str, Any]]:
        """Analyze exercises (no fragment merging - Smart Split handles that).

        Args:
            exercises: List of exercise dicts from database
            skip_analyzed: If True, skip exercises already marked as analyzed

        Returns:
            List of exercises with analysis results
        """
        if not exercises:
            return []

        results = []

        for exercise in exercises:
            # Skip already analyzed exercises if requested
            if skip_analyzed and exercise.get('analyzed'):
                print(f"[DEBUG] Skipping already analyzed exercise: {exercise['id'][:40]}...")
                continue

            # Analyze exercise
            analysis = self.analyze_exercise(
                exercise["text"],
                exercise.get("course_name", "Unknown Course"),
            )

            results.append({
                **exercise,
                "merged_from": [exercise["id"]],
                "analysis": analysis
            })

        return results

    def _analyze_exercise_with_retry(self, exercise_text: str, course_name: str,
                                     max_retries: int = 2) -> AnalysisResult:
        """Analyze exercise with retry logic for failed API calls.

        Args:
            exercise_text: Exercise text
            course_name: Course name for context
            max_retries: Maximum number of retries on failure

        Returns:
            AnalysisResult with classification
        """
        # Check procedure cache first (Option 3: Performance Optimization)
        if self.procedure_cache and Config.PROCEDURE_CACHE_ENABLED:
            cache_hit = self.procedure_cache.lookup(exercise_text, course_code=course_name)
            if cache_hit and cache_hit.confidence >= Config.PROCEDURE_CACHE_MIN_CONFIDENCE:
                self.cache_stats['hits'] += 1
                # Build result from cache
                return self._build_result_from_cache(cache_hit, exercise_text)
            else:
                # No cache hit or low confidence - track as miss
                self.cache_stats['misses'] += 1

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                result = self.analyze_exercise(exercise_text, course_name)
                # Check if analysis was successful
                if result.confidence > 0.0 or result.topic is not None:
                    # Add to procedure cache for future use (Option 3)
                    if self.procedure_cache and Config.PROCEDURE_CACHE_ENABLED and result.procedures:
                        self.procedure_cache.add(
                            exercise_text=exercise_text,
                            topic=result.topic or '',
                            difficulty=result.difficulty or 'medium',
                            variations=result.variations or [],
                            procedures=[p.__dict__ if hasattr(p, '__dict__') else p for p in result.procedures],
                            confidence=result.confidence,
                            course_code=course_name
                        )
                    return result
                # If we got default result, retry
                if attempt < max_retries:
                    print(f"  Retry {attempt + 1}/{max_retries} for exercise...")
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
                    continue
                return result
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    print(f"  Error on attempt {attempt + 1}: {str(e)}, retrying...")
                    time.sleep(1 * (attempt + 1))
                    continue

        # All retries failed, return default
        print(f"  All retries failed: {last_error}")
        return self._default_analysis_result()

    async def _analyze_exercise_with_retry_async(self, exercise_text: str, course_name: str,
                                                  max_retries: int = 2) -> AnalysisResult:
        """Analyze exercise asynchronously with retry logic for failed API calls.

        Args:
            exercise_text: Exercise text
            course_name: Course name for context
            max_retries: Maximum number of retries on failure

        Returns:
            AnalysisResult with classification
        """
        # Check procedure cache first (Option 3: Performance Optimization)
        # Note: Cache lookup is synchronous, but fast (in-memory)
        if self.procedure_cache and Config.PROCEDURE_CACHE_ENABLED:
            cache_hit = self.procedure_cache.lookup(exercise_text, course_code=course_name)
            if cache_hit and cache_hit.confidence >= Config.PROCEDURE_CACHE_MIN_CONFIDENCE:
                self.cache_stats['hits'] += 1
                # Build result from cache
                return self._build_result_from_cache(cache_hit, exercise_text)
            else:
                # No cache hit or low confidence - track as miss
                self.cache_stats['misses'] += 1

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                # Build prompt
                prompt = self._build_analysis_prompt(
                    exercise_text, course_name
                )

                # Call LLM asynchronously
                response = await self.llm.generate_async(
                    prompt=prompt,
                    model=self.llm.primary_model,
                    temperature=0.3,
                    json_mode=True
                )

                if not response.success:
                    # Log error and retry or return default
                    if attempt < max_retries:
                        print(f"  Retry {attempt + 1}/{max_retries} for exercise (error: {response.error})...")
                        await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                        continue
                    print(f"[ERROR] LLM failed for exercise: {response.error}")
                    print(f"  Text preview: {exercise_text[:100]}...")
                    return self._default_analysis_result()

                # Parse JSON response
                data = self.llm.parse_json_response(response)
                if not data:
                    if attempt < max_retries:
                        print(f"  Retry {attempt + 1}/{max_retries} for exercise (invalid JSON)...")
                        await asyncio.sleep(1 * (attempt + 1))
                        continue
                    return self._default_analysis_result()

                # Parse procedures (new format) or fallback to old format
                procedures = []
                if "procedures" in data and data["procedures"]:
                    # New format: multiple procedures
                    for proc_data in data["procedures"]:
                        procedures.append(ProcedureInfo(
                            name=proc_data.get("name", "Unknown Procedure"),
                            type=proc_data.get("type", "other"),
                            steps=proc_data.get("steps", []),
                            point_number=proc_data.get("point_number"),
                            transformation=proc_data.get("transformation")
                        ))
                elif "knowledge_item_name" in data and data["knowledge_item_name"]:
                    # Old format: single procedure - convert to new format
                    procedures.append(ProcedureInfo(
                        name=data["knowledge_item_name"],
                        type="other",
                        steps=data.get("procedure", []),
                        point_number=None,
                        transformation=None
                    ))

                # Normalize procedures to primary language if monolingual mode enabled
                if self.monolingual and procedures:
                    procedures = self._normalize_procedures_to_primary_language(procedures)

                # Phase 9.1: Extract exercise type information
                exercise_type = data.get("exercise_type", "procedural")
                type_confidence = data.get("type_confidence", 0.5)
                proof_keywords = data.get("proof_keywords", [])
                theory_metadata = data.get("theory_metadata")

                # Phase 9.2: Extract theory categorization information
                theory_category = data.get("theory_category")
                theorem_name = data.get("theorem_name")
                concept_id = data.get("concept_id")
                prerequisite_concepts = data.get("prerequisite_concepts")

                # Concept variation support
                parent_concept_name = data.get("parent_concept_name")
                variation_parameter = data.get("variation_parameter")

                # Build result
                result = AnalysisResult(
                    is_valid_exercise=data.get("is_valid_exercise", True),
                    is_fragment=data.get("is_fragment", False),
                    should_merge_with_previous=data.get("should_merge_with_previous", False),
                    topic=data.get("topic"),
                    difficulty=data.get("difficulty"),
                    variations=data.get("variations", []),
                    confidence=data.get("confidence", 0.5),
                    procedures=procedures,
                    exercise_type=exercise_type,
                    type_confidence=type_confidence,
                    proof_keywords=proof_keywords if proof_keywords else None,
                    theory_metadata=theory_metadata,
                    theory_category=theory_category,
                    theorem_name=theorem_name,
                    concept_id=concept_id,
                    prerequisite_concepts=prerequisite_concepts,
                    parent_concept_name=parent_concept_name,
                    variation_parameter=variation_parameter
                )

                # Check if analysis was successful
                if result.confidence > 0.0 or result.topic is not None:
                    # Add to procedure cache for future use (Option 3)
                    if self.procedure_cache and Config.PROCEDURE_CACHE_ENABLED and result.procedures:
                        self.procedure_cache.add(
                            exercise_text=exercise_text,
                            topic=result.topic or '',
                            difficulty=result.difficulty or 'medium',
                            variations=result.variations or [],
                            procedures=[p.__dict__ if hasattr(p, '__dict__') else p for p in result.procedures],
                            confidence=result.confidence,
                            course_code=course_name
                        )
                    return result

                # If we got default result, retry
                if attempt < max_retries:
                    print(f"  Retry {attempt + 1}/{max_retries} for exercise (low confidence)...")
                    await asyncio.sleep(1 * (attempt + 1))
                    continue

                return result

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    print(f"  Error on attempt {attempt + 1}: {str(e)}, retrying...")
                    await asyncio.sleep(1 * (attempt + 1))
                    continue

        # All retries failed, return default
        print(f"  All retries failed: {last_error}")
        return self._default_analysis_result()

    def merge_exercises_parallel(self, exercises: List[Dict[str, Any]],
                                 batch_size: Optional[int] = None,
                                 show_progress: bool = True,
                                 skip_analyzed: bool = False) -> List[Dict[str, Any]]:
        """Analyze exercises in parallel batches (no fragment merging - Smart Split handles that).

        Args:
            exercises: List of exercise dicts from database
            batch_size: Number of exercises to process in parallel (defaults to Config.BATCH_SIZE)
            show_progress: Show progress bar during processing
            skip_analyzed: If True, skip exercises already marked as analyzed

        Returns:
            List of exercises with analysis results
        """
        if not exercises:
            return []

        batch_size = batch_size or Config.BATCH_SIZE
        total = len(exercises)

        print(f"[INFO] Starting parallel batch analysis of {total} exercises (batch_size={batch_size})...")
        start_time = time.time()

        # Store analysis results indexed by exercise position
        analysis_results = {}

        # Process in batches
        def analyze_single(index: int, exercise: Dict[str, Any]) -> tuple:
            """Analyze a single exercise and return (index, analysis, error)."""
            try:
                # Skip already analyzed if requested
                if skip_analyzed and exercise.get('analyzed'):
                    return (index, None, None)  # Signal to skip

                analysis = self._analyze_exercise_with_retry(
                    exercise["text"],
                    exercise.get("course_name", "Unknown Course"),
                )
                return (index, analysis, None)
            except Exception as e:
                print(f"  [ERROR] Failed to analyze exercise {index}: {str(e)}")
                return (index, self._default_analysis_result(), str(e))

        # Process exercises in batches with ThreadPoolExecutor
        processed = 0
        failed_count = 0
        skipped_count = 0

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = exercises[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))

            if show_progress:
                print(f"  Processing batch {batch_start//batch_size + 1}/{(total + batch_size - 1)//batch_size} (exercises {batch_start+1}-{batch_end}/{total})...")

            # Prepare analysis tasks for this batch
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {}

                for idx, exercise in zip(batch_indices, batch):
                    future = executor.submit(analyze_single, idx, exercise)
                    futures[future] = idx

                # Collect results as they complete
                for future in as_completed(futures):
                    idx, analysis, error = future.result()

                    if analysis is None:  # Skipped exercise
                        skipped_count += 1
                    else:
                        analysis_results[idx] = analysis

                    processed += 1

                    if error:
                        failed_count += 1

                    if show_progress and processed % 5 == 0:
                        elapsed = time.time() - start_time
                        rate = processed / elapsed if elapsed > 0 else 0
                        eta = (total - processed) / rate if rate > 0 else 0
                        print(f"    Progress: {processed}/{total} ({100*processed/total:.1f}%) | {rate:.1f} ex/s | ETA: {eta:.0f}s")

        elapsed_time = time.time() - start_time

        print(f"[INFO] Batch analysis complete in {elapsed_time:.1f}s")
        print(f"  Processed: {processed}/{total} exercises")
        if skipped_count > 0:
            print(f"  Skipped (already analyzed): {skipped_count} exercises")
        print(f"  Failed: {failed_count} exercises")
        print(f"  Rate: {processed/elapsed_time:.1f} exercises/second")

        # Build results list (each exercise is standalone - no fragment merging)
        results = []
        for i, exercise in enumerate(exercises):
            if i not in analysis_results:
                continue

            results.append({
                **exercise,
                "merged_from": [exercise["id"]],
                "analysis": analysis_results[i]
            })

        print(f"[INFO] Analyzed {len(results)} exercises")

        return results

    async def merge_exercises_async(self, exercises: List[Dict[str, Any]],
                                    batch_size: Optional[int] = None,
                                    show_progress: bool = True,
                                    skip_analyzed: bool = False) -> List[Dict[str, Any]]:
        """Analyze exercises using async batch processing (no fragment merging - Smart Split handles that).

        Args:
            exercises: List of exercise dicts from database
            batch_size: Number of exercises to process concurrently (defaults to Config.BATCH_SIZE)
            show_progress: Show progress bar during processing
            skip_analyzed: If True, skip exercises already marked as analyzed

        Returns:
            List of exercises with analysis results
        """
        if not exercises:
            return []

        batch_size = batch_size or Config.BATCH_SIZE
        total = len(exercises)

        print(f"[INFO] Starting async batch analysis of {total} exercises (batch_size={batch_size})...")
        start_time = time.time()

        # Store analysis results indexed by exercise position
        analysis_results = {}

        # Process in batches
        async def analyze_single(index: int, exercise: Dict[str, Any]) -> tuple:
            """Analyze a single exercise and return (index, analysis, error)."""
            try:
                # Skip already analyzed if requested
                if skip_analyzed and exercise.get('analyzed'):
                    return (index, None, None)  # Signal to skip

                analysis = await self._analyze_exercise_with_retry_async(
                    exercise["text"],
                    exercise.get("course_name", "Unknown Course"),
                )
                return (index, analysis, None)
            except Exception as e:
                print(f"  [ERROR] Failed to analyze exercise {index}: {str(e)}")
                return (index, self._default_analysis_result(), str(e))

        # Process exercises in batches with asyncio.gather()
        processed = 0
        failed_count = 0
        skipped_count = 0

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = exercises[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))

            if show_progress:
                print(f"  Processing batch {batch_start//batch_size + 1}/{(total + batch_size - 1)//batch_size} (exercises {batch_start+1}-{batch_end}/{total})...")

            # Prepare analysis tasks for this batch
            tasks = []
            for idx, exercise in zip(batch_indices, batch):
                task = analyze_single(idx, exercise)
                tasks.append(task)

            # Run all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result in results:
                if isinstance(result, Exception):
                    # Task raised an exception
                    print(f"  [ERROR] Task failed with exception: {result}")
                    failed_count += 1
                    processed += 1
                    continue

                idx, analysis, error = result

                if analysis is None:  # Skipped exercise
                    skipped_count += 1
                else:
                    analysis_results[idx] = analysis

                processed += 1

                if error:
                    failed_count += 1

                if show_progress and processed % 5 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    eta = (total - processed) / rate if rate > 0 else 0
                    print(f"    Progress: {processed}/{total} ({100*processed/total:.1f}%) | {rate:.1f} ex/s | ETA: {eta:.0f}s")

        elapsed_time = time.time() - start_time

        print(f"[INFO] Async batch analysis complete in {elapsed_time:.1f}s")
        print(f"  Processed: {processed}/{total} exercises")
        if skipped_count > 0:
            print(f"  Skipped (already analyzed): {skipped_count} exercises")
        print(f"  Failed: {failed_count} exercises")
        print(f"  Rate: {processed/elapsed_time:.1f} exercises/second")

        # Build results list (each exercise is standalone - no fragment merging)
        results_list = []
        for i, exercise in enumerate(exercises):
            if i not in analysis_results:
                continue

            results_list.append({
                **exercise,
                "merged_from": [exercise["id"]],
                "analysis": analysis_results[i]
            })

        print(f"[INFO] Analyzed {len(results_list)} exercises")

        return results_list

    def discover_knowledge_items(self, course_code: str,
                            batch_size: int = 10,
                            skip_analyzed: bool = False,
                            use_parallel: bool = True) -> Dict[str, Any]:
        """Discover core loops for a course.

        Args:
            course_code: Course code
            batch_size: Number of exercises to analyze at once
            skip_analyzed: If True, skip already analyzed exercises
            use_parallel: If True, use parallel batch processing (default: True)

        Returns:
            Dict with core loops discovered
        """
        with Database() as db:
            # Get all exercises for course
            exercises = db.get_exercises_by_course(course_code)

            if not exercises:
                return {"knowledge_items": {}}

            # Detect primary language if monolingual mode enabled
            if self.monolingual and not self.primary_language:
                course = db.get_course(course_code)
                course_name = course['name'] if course else course_code
                self.primary_language = self._detect_primary_language(exercises, course_name)
                print(f"[MONOLINGUAL MODE] Primary language set to: {self.primary_language}")
                print(f"  All procedures will be normalized to {self.primary_language}\n")

            # Merge fragments first - use parallel or sequential mode
            if use_parallel:
                merged_exercises = self.merge_exercises_parallel(
                    exercises,
                    batch_size=batch_size,
                    skip_analyzed=skip_analyzed
                )
            else:
                merged_exercises = self.merge_exercises(exercises, skip_analyzed=skip_analyzed)

            # Collect all analyses
            knowledge_items = {}
            low_confidence_count = 0

            for merged_ex in merged_exercises:
                analysis = merged_ex.get("analysis")
                if not analysis:
                    continue

                # Skip low-confidence analyses
                if analysis.confidence < Config.MIN_ANALYSIS_CONFIDENCE:
                    low_confidence_count += 1
                    print(f"[INFO] Skipping exercise due to low confidence ({analysis.confidence:.2f} < {Config.MIN_ANALYSIS_CONFIDENCE}): {merged_ex['id'][:40]}...")
                    merged_ex["low_confidence_skipped"] = True
                    continue

                # Process ALL procedures (multi-procedure support)
                if analysis.procedures:
                    if len(analysis.procedures) > 1:
                        print(f"[INFO] Multiple procedures detected ({len(analysis.procedures)}) in exercise {merged_ex['id'][:40]}:")
                        for i, proc in enumerate(analysis.procedures, 1):
                            print(f"  {i}. {proc.name} (type: {proc.type}, point: {proc.point_number})")

                    for procedure_info in analysis.procedures:
                        knowledge_item_id = self._normalize_knowledge_item_id(procedure_info.name)

                        if knowledge_item_id and procedure_info.name:
                            if knowledge_item_id not in knowledge_items:
                                knowledge_items[knowledge_item_id] = {
                                    "id": knowledge_item_id,
                                    "name": procedure_info.name,
                                    "procedure": procedure_info.steps or [],
                                    "type": procedure_info.type,
                                    "transformation": procedure_info.transformation,
                                    "exercise_count": 0,
                                    "exercises": []
                                }
                            knowledge_items[knowledge_item_id]["exercise_count"] += 1
                            if merged_ex["id"] not in knowledge_items[knowledge_item_id]["exercises"]:
                                knowledge_items[knowledge_item_id]["exercises"].append(merged_ex["id"])

            # Deduplicate against existing database entries
            knowledge_items = self._deduplicate_knowledge_items_with_database(knowledge_items, course_code, db)

            # Log summary statistics
            accepted_count = len(merged_exercises) - low_confidence_count
            if low_confidence_count > 0:
                print(f"\n[SUMMARY] Confidence Filtering Results:")
                print(f"  Total merged exercises: {len(merged_exercises)}")
                print(f"  Accepted (>= {Config.MIN_ANALYSIS_CONFIDENCE} confidence): {accepted_count}")
                print(f"  Skipped (low confidence): {low_confidence_count}")
                print(f"  Skip rate: {(low_confidence_count / len(merged_exercises) * 100):.1f}%\n")

            return {
                "knowledge_items": knowledge_items,
                "merged_exercises": merged_exercises,
                "original_count": len(exercises),
                "merged_count": len(merged_exercises),
                "low_confidence_skipped": low_confidence_count,
                "accepted_count": accepted_count
            }

    # Backwards compatibility alias
    def discover_topics_and_knowledge_items(self, *args, **kwargs) -> Dict[str, Any]:
        """DEPRECATED: Use discover_knowledge_items() instead. Topics have been removed."""
        result = self.discover_knowledge_items(*args, **kwargs)
        result["topics"] = {}  # Empty for backwards compatibility
        return result

    async def discover_knowledge_items_async(self, course_code: str,
                                        batch_size: int = 10,
                                        skip_analyzed: bool = False) -> Dict[str, Any]:
        """Discover core loops for a course using async processing.

        Args:
            course_code: Course code
            batch_size: Number of exercises to analyze at once
            skip_analyzed: If True, skip already analyzed exercises

        Returns:
            Dict with core loops discovered
        """
        with Database() as db:
            # Get all exercises for course
            exercises = db.get_exercises_by_course(course_code)

            if not exercises:
                return {"knowledge_items": {}}

            # Detect primary language if monolingual mode enabled
            if self.monolingual and not self.primary_language:
                course = db.get_course(course_code)
                course_name = course['name'] if course else course_code
                self.primary_language = self._detect_primary_language(exercises, course_name)
                print(f"[MONOLINGUAL MODE] Primary language set to: {self.primary_language}")
                print(f"  All procedures will be normalized to {self.primary_language}\n")

            # Merge fragments using async processing
            merged_exercises = await self.merge_exercises_async(
                exercises,
                batch_size=batch_size,
                skip_analyzed=skip_analyzed
            )

            # Collect all analyses
            knowledge_items = {}
            low_confidence_count = 0

            for merged_ex in merged_exercises:
                analysis = merged_ex.get("analysis")
                if not analysis:
                    continue

                # Skip low-confidence analyses
                if analysis.confidence < Config.MIN_ANALYSIS_CONFIDENCE:
                    low_confidence_count += 1
                    print(f"[INFO] Skipping exercise due to low confidence ({analysis.confidence:.2f} < {Config.MIN_ANALYSIS_CONFIDENCE}): {merged_ex['id'][:40]}...")
                    merged_ex["low_confidence_skipped"] = True
                    continue

                # Process ALL procedures
                if analysis.procedures:
                    if len(analysis.procedures) > 1:
                        print(f"[INFO] Multiple procedures detected ({len(analysis.procedures)}) in exercise {merged_ex['id'][:40]}:")
                        for i, proc in enumerate(analysis.procedures, 1):
                            print(f"  {i}. {proc.name} (type: {proc.type}, point: {proc.point_number})")

                    for procedure_info in analysis.procedures:
                        knowledge_item_id = self._normalize_knowledge_item_id(procedure_info.name)

                        if knowledge_item_id and procedure_info.name:
                            if knowledge_item_id not in knowledge_items:
                                knowledge_items[knowledge_item_id] = {
                                    "id": knowledge_item_id,
                                    "name": procedure_info.name,
                                    "procedure": procedure_info.steps or [],
                                    "type": procedure_info.type,
                                    "transformation": procedure_info.transformation,
                                    "exercise_count": 0,
                                    "exercises": []
                                }
                            knowledge_items[knowledge_item_id]["exercise_count"] += 1
                            if merged_ex["id"] not in knowledge_items[knowledge_item_id]["exercises"]:
                                knowledge_items[knowledge_item_id]["exercises"].append(merged_ex["id"])

            # Deduplicate against existing database entries
            knowledge_items = self._deduplicate_knowledge_items_with_database(knowledge_items, course_code, db)

            # Log summary statistics
            accepted_count = len(merged_exercises) - low_confidence_count
            if low_confidence_count > 0:
                print(f"\n[SUMMARY] Confidence Filtering Results:")
                print(f"  Total merged exercises: {len(merged_exercises)}")
                print(f"  Accepted (>= {Config.MIN_ANALYSIS_CONFIDENCE} confidence): {accepted_count}")
                print(f"  Skipped (low confidence): {low_confidence_count}")
                print(f"  Skip rate: {(low_confidence_count / len(merged_exercises) * 100):.1f}%\n")

            return {
                "knowledge_items": knowledge_items,
                "merged_exercises": merged_exercises,
                "original_count": len(exercises),
                "merged_count": len(merged_exercises),
                "low_confidence_skipped": low_confidence_count,
                "accepted_count": accepted_count
            }

    # Backwards compatibility alias
    async def discover_topics_and_knowledge_items_async(self, *args, **kwargs) -> Dict[str, Any]:
        """DEPRECATED: Use discover_knowledge_items_async() instead. Topics have been removed."""
        result = await self.discover_knowledge_items_async(*args, **kwargs)
        result["topics"] = {}  # Empty for backwards compatibility
        return result

    def _similarity(self, str1: str, str2: str) -> Tuple[float, str]:
        """Calculate similarity between two strings (0.0 to 1.0).

        Args:
            str1: First string
            str2: Second string

        Returns:
            Tuple of (similarity score, reason)
            - similarity: 0.0 = completely different, 1.0 = identical
            - reason: "semantic_similarity", "translation", "string_similarity", etc.
        """
        if self.use_semantic and self.semantic_matcher:
            # Use semantic similarity
            result = self.semantic_matcher.should_merge(str1, str2, threshold=0.0)
            return result.similarity_score, result.reason
        else:
            # Fallback to string similarity
            similarity = SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
            return similarity, "string_similarity"

    def _deduplicate_knowledge_items(self, knowledge_items: Dict[str, Any]) -> Dict[str, Any]:
        """Deduplicate similar core loops using semantic similarity.

        Args:
            knowledge_items: Dictionary of core loops

        Returns:
            Tuple of (deduplicated core loops dictionary, ID mapping dict)
        """
        if len(knowledge_items) <= 1:
            return knowledge_items

        threshold = Config.SEMANTIC_SIMILARITY_THRESHOLD if self.use_semantic else Config.KNOWLEDGE_ITEM_SIMILARITY_THRESHOLD
        loop_ids = list(knowledge_items.keys())
        merged_loops = {}
        skip_loops = set()

        # Track mapping from old IDs to canonical IDs
        self.knowledge_item_id_mapping = {}

        for i, loop1_id in enumerate(loop_ids):
            if loop1_id in skip_loops:
                continue

            loop1 = knowledge_items[loop1_id]
            canonical_id = loop1_id
            canonical_data = loop1.copy()

            # Map canonical ID to itself
            self.knowledge_item_id_mapping[canonical_id] = canonical_id

            # Check for similar core loops (compare names, not IDs)
            for loop2_id in loop_ids[i+1:]:
                if loop2_id in skip_loops:
                    continue

                loop2 = knowledge_items[loop2_id]

                # Use semantic matching if available
                if self.use_semantic and self.semantic_matcher:
                    result = self.semantic_matcher.should_merge(
                        loop1["name"], loop2["name"], threshold
                    )
                    if result.should_merge:
                        print(f"[DEDUP] Core loop '{loop1['name']}' → '{loop2['name']}' (similarity: {result.similarity_score:.2f}, reason: {result.reason})")
                        # Merge loop2 into canonical
                        canonical_data["exercise_count"] += loop2["exercise_count"]
                        canonical_data["exercises"] = list(set(canonical_data["exercises"]) | set(loop2["exercises"]))
                        # Merge procedures (prefer longer/more detailed one)
                        if len(loop2.get("procedure", [])) > len(canonical_data.get("procedure", [])):
                            canonical_data["procedure"] = loop2["procedure"]
                        skip_loops.add(loop2_id)
                        # Map merged ID to canonical ID
                        self.knowledge_item_id_mapping[loop2_id] = canonical_id
                    elif Config.SEMANTIC_LOG_NEAR_MISSES and result.similarity_score >= 0.80:
                        print(f"[SKIP] Core loop '{loop1['name']}' ≠ '{loop2['name']}' (similarity: {result.similarity_score:.2f}, reason: {result.reason})")
                else:
                    # Fallback to string similarity
                    similarity, reason = self._similarity(loop1["name"], loop2["name"])
                    if similarity >= threshold:
                        print(f"[DEBUG] Merging similar core loops: '{loop1['name']}' ≈ '{loop2['name']}' (similarity: {similarity:.2f})")
                        # Merge loop2 into canonical
                        canonical_data["exercise_count"] += loop2["exercise_count"]
                        canonical_data["exercises"] = list(set(canonical_data["exercises"]) | set(loop2["exercises"]))
                        # Merge procedures (prefer longer/more detailed one)
                        if len(loop2.get("procedure", [])) > len(canonical_data.get("procedure", [])):
                            canonical_data["procedure"] = loop2["procedure"]
                        skip_loops.add(loop2_id)
                        # Map merged ID to canonical ID
                        self.knowledge_item_id_mapping[loop2_id] = canonical_id

            merged_loops[canonical_id] = canonical_data

        return merged_loops

    def _deduplicate_knowledge_items_with_database(self, knowledge_items: Dict[str, Any],
                                              course_code: str,
                                              db) -> Dict[str, Any]:
        """Deduplicate core loops against existing database entries, then within batch.

        Args:
            knowledge_items: Dictionary of new core loops from current analysis
            course_code: Course code
            db: Database instance

        Returns:
            Deduplicated core loops dictionary with mappings to existing DB loops
        """
        threshold = Config.SEMANTIC_SIMILARITY_THRESHOLD if self.use_semantic else Config.KNOWLEDGE_ITEM_SIMILARITY_THRESHOLD

        # Load existing core loops from database
        existing_loops = db.get_knowledge_items_by_course(course_code)
        existing_loop_map = {loop['id']: loop for loop in existing_loops}

        # Track mappings from new loop IDs to canonical (db or batch) IDs
        loop_id_mapping = {}
        deduplicated_loops = {}

        for new_loop_id, new_loop_data in knowledge_items.items():
            matched_existing = None
            best_similarity = 0.0
            best_reason = ""

            # Check against existing database loops first
            for existing_loop in existing_loops:
                if self.use_semantic and self.semantic_matcher:
                    result = self.semantic_matcher.should_merge(
                        new_loop_data['name'], existing_loop['name'], threshold
                    )
                    if result.should_merge and result.similarity_score > best_similarity:
                        best_similarity = result.similarity_score
                        matched_existing = existing_loop['id']
                        best_reason = result.reason
                else:
                    similarity, reason = self._similarity(new_loop_data['name'], existing_loop['name'])
                    if similarity >= threshold and similarity > best_similarity:
                        best_similarity = similarity
                        matched_existing = existing_loop['id']
                        best_reason = reason

            if matched_existing:
                # Reuse existing loop
                print(f"[DEDUP] Core loop '{new_loop_data['name']}' → existing '{existing_loop_map[matched_existing]['name']}' (similarity: {best_similarity:.2f}, reason: {best_reason})")
                loop_id_mapping[new_loop_id] = matched_existing
                # Don't add to deduplicated_loops, we'll use DB entry
            else:
                # New loop, add to batch
                deduplicated_loops[new_loop_id] = new_loop_data
                loop_id_mapping[new_loop_id] = new_loop_id

        # Now deduplicate within the new batch
        deduplicated_loops = self._deduplicate_knowledge_items(deduplicated_loops)

        # Update loop mapping with any batch deduplication
        if hasattr(self, 'knowledge_item_id_mapping'):
            for old_id, canonical_id in self.knowledge_item_id_mapping.items():
                if old_id in loop_id_mapping:
                    loop_id_mapping[old_id] = canonical_id

        # Store the mapping for later use
        self.knowledge_item_id_mapping = loop_id_mapping

        return deduplicated_loops

    # ========================================================================
    # Learning Material Analysis Methods (Phase 10)
    # ========================================================================

    def analyze_learning_material(self, material_text: str, course_name: str) -> AnalysisResult:
        """Analyze a learning material to detect topics.

        Similar to analyze_exercise() but for theory/worked examples.
        Returns topic names that this material relates to.

        Args:
            material_text: Material content (theory or worked example)
            course_name: Course name for context

        Returns:
            AnalysisResult with topic and metadata
        """
        # Build prompt for material analysis
        prompt = self._build_material_analysis_prompt(material_text, course_name)

        # Call LLM
        response = self.llm.generate(
            prompt=prompt,
            model=self.llm.primary_model,
            temperature=0.3,
            json_mode=True
        )

        if not response.success:
            print(f"[ERROR] LLM failed for material: {response.error}")
            print(f"  Text preview: {material_text[:100]}...")
            return self._default_analysis_result()

        # Parse JSON response
        data = self.llm.parse_json_response(response)
        if not data:
            return self._default_analysis_result()

        # Extract topics (material may relate to multiple topics)
        topics = data.get("topics", [])
        primary_topic = topics[0] if topics else None

        # Extract procedures if any (for worked examples)
        procedures = []
        if "procedures" in data and data["procedures"]:
            for proc_data in data["procedures"]:
                procedures.append(ProcedureInfo(
                    name=proc_data.get("name", "Unknown Procedure"),
                    type=proc_data.get("type", "other"),
                    steps=proc_data.get("steps", []),
                    point_number=proc_data.get("point_number"),
                    transformation=proc_data.get("transformation")
                ))

        # Return analysis result
        return AnalysisResult(
            is_valid_exercise=True,  # Materials are always valid
            is_fragment=False,
            should_merge_with_previous=False,
            topic=primary_topic,
            difficulty=data.get("difficulty"),
            variations=topics[1:] if len(topics) > 1 else [],  # Additional topics
            confidence=data.get("confidence", 0.5),
            procedures=procedures,
            exercise_type=data.get("material_type", "theory"),
            type_confidence=data.get("type_confidence", 0.8)
        )

    def _build_material_analysis_prompt(self, material_text: str, course_name: str) -> str:
        """Build prompt for learning material analysis.

        Args:
            material_text: Material content
            course_name: Course name

        Returns:
            Prompt string
        """
        prompt = f"""You are analyzing learning materials (theory or worked examples) for the course: {course_name}.

Your task is to analyze this material and determine:
1. What topic(s) does it cover?
2. What concepts or procedures does it explain?
3. Is it theory (definitions, explanations) or a worked example (step-by-step solution)?

MATERIAL TEXT:
```
{material_text[:3000]}
```

IMPORTANT: {self._language_instruction("Respond")} All topic names must be in {self._language_name()} language.

Respond in JSON format with:
{{
  "topics": ["primary topic", "secondary topic", ...],  // 1-3 specific topics this material covers
  "material_type": "theory|worked_example|reference",  // Type of material
  "difficulty": "easy|medium|hard",  // Complexity level
  "confidence": 0.0-1.0,  // Your confidence in this analysis
  "procedures": [  // ONLY for worked examples: procedures demonstrated
    {{
      "name": "procedure name",
      "type": "design|transformation|verification|minimization|analysis|other",
      "steps": ["step 1", "step 2", ...],
      "point_number": null,
      "transformation": null
    }}
  ],
  "key_concepts": ["concept1", "concept2", ...],  // Main concepts covered
  "type_confidence": 0.0-1.0  // Confidence in material type
}}

TOPIC NAMING RULES (CRITICAL):
- NEVER use the course name "{course_name}" as the topic - it's too generic!
- Topics MUST be SPECIFIC subtopics within the course
- Be as specific as possible - narrow topics are better than broad ones
- For worked examples, identify the procedure/algorithm being demonstrated
- Materials may relate to multiple topics - list the most relevant ones (1-3)

MATERIAL TYPE CLASSIFICATION:
- **theory**: Definitions, theorems, explanations, conceptual content
  * Contains definitions, properties, explanations without computations
  * No step-by-step computations

- **worked_example**: Step-by-step solution showing how to solve a problem
  * Shows the execution of a procedure/algorithm with steps
  * Should identify the procedure and extract steps

- **reference**: Tables, formulas, reference material without explanations
  * Summary tables, formula sheets, quick reference content

Respond ONLY with valid JSON, no other text.
"""
        return prompt

    def link_materials_to_topics(self, course_code: str):
        """Analyze learning materials and link them to topics.

        For each unlinked learning material:
        1. Analyze content to detect topics
        2. Match detected topics to existing course topics (by name/semantic similarity)
        3. Create links via db.link_material_to_topic()

        Args:
            course_code: Course code
        """
        with Database() as db:
            # Get all materials for this course
            materials = db.get_learning_materials_by_course(course_code)

            if not materials:
                print(f"[INFO] No learning materials found for course {course_code}")
                return

            # Get existing topics for this course
            course_topics = db.get_topics_by_course(course_code)
            topic_map = {t['name']: t for t in course_topics}

            if not course_topics:
                print(f"[WARNING] No topics found for course {course_code}. Run 'analyze' first.")
                return

            # Get course name for context
            course = db.get_course(course_code)
            course_name = course['name'] if course else course_code

            print(f"[INFO] Linking {len(materials)} materials to {len(course_topics)} topics...")

            linked_count = 0
            skipped_count = 0

            for material in materials:
                # Check if already linked
                existing_topics = db.get_topics_for_material(material['id'])
                if existing_topics:
                    print(f"[DEBUG] Material {material['id'][:40]} already linked to {len(existing_topics)} topic(s), skipping")
                    skipped_count += 1
                    continue

                # Analyze material to detect topics
                print(f"[DEBUG] Analyzing material: {material['id'][:40]}... (type: {material['material_type']})")
                analysis = self.analyze_learning_material(
                    material['content'],
                    course_name
                )

                if analysis.confidence < Config.MIN_ANALYSIS_CONFIDENCE:
                    print(f"[WARNING] Low confidence analysis ({analysis.confidence:.2f}), skipping material")
                    continue

                # Collect all detected topics (primary + variations)
                detected_topics = []
                if analysis.topic:
                    detected_topics.append(analysis.topic)
                if analysis.variations:
                    detected_topics.extend(analysis.variations)

                if not detected_topics:
                    print(f"[WARNING] No topics detected for material {material['id'][:40]}")
                    continue

                # Match each detected topic to existing course topics
                for detected_topic in detected_topics:
                    matched_topic_id = self._match_topic_to_existing(
                        detected_topic,
                        course_topics,
                        db
                    )

                    if matched_topic_id:
                        db.link_material_to_topic(material['id'], matched_topic_id)
                        matched_topic = next(t for t in course_topics if t['id'] == matched_topic_id)
                        print(f"[INFO] Linked material to topic: '{matched_topic['name']}'")
                        linked_count += 1
                    else:
                        print(f"[WARNING] Could not match detected topic '{detected_topic}' to existing topics")

            print(f"\n[SUMMARY] Material linking complete:")
            print(f"  Linked: {linked_count} material-topic links created")
            print(f"  Skipped (already linked): {skipped_count} materials")

    def _match_topic_to_existing(self, detected_topic: str, course_topics: List[Dict[str, Any]],
                                  db) -> Optional[int]:
        """Match a detected topic name to an existing course topic.

        Uses semantic similarity to find the best match.

        Args:
            detected_topic: Topic name detected from material
            course_topics: List of existing course topics
            db: Database instance

        Returns:
            Topic ID if match found, None otherwise
        """
        threshold = Config.SEMANTIC_SIMILARITY_THRESHOLD if self.use_semantic else Config.KNOWLEDGE_ITEM_SIMILARITY_THRESHOLD

        best_match_id = None
        best_similarity = 0.0
        best_reason = ""

        for topic in course_topics:
            # Try semantic matching if available
            if self.use_semantic and self.semantic_matcher:
                result = self.semantic_matcher.should_merge(
                    detected_topic, topic['name'], threshold
                )
                if result.should_merge and result.similarity_score > best_similarity:
                    best_similarity = result.similarity_score
                    best_match_id = topic['id']
                    best_reason = result.reason
            else:
                # Fallback to string similarity
                similarity, reason = self._similarity(detected_topic, topic['name'])
                if similarity >= threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = topic['id']
                    best_reason = reason

        if best_match_id:
            matched_topic = next(t for t in course_topics if t['id'] == best_match_id)
            print(f"[MATCH] '{detected_topic}' → '{matched_topic['name']}' (similarity: {best_similarity:.2f}, reason: {best_reason})")

        return best_match_id

    def link_worked_examples_to_exercises(self, course_code: str, max_links_per_example: int = 5):
        """Link worked examples to similar practice exercises.

        For each worked_example material:
        1. Get its topics
        2. Find exercises with same topics
        3. Use semantic similarity to find most related exercises
        4. Create links via db.link_material_to_exercise(type='worked_example')

        Args:
            course_code: Course code
            max_links_per_example: Maximum number of exercises to link per example
        """
        with Database() as db:
            # Get all worked example materials
            worked_examples = db.get_learning_materials_by_course(
                course_code,
                material_type='worked_example'
            )

            if not worked_examples:
                print(f"[INFO] No worked examples found for course {course_code}")
                return

            print(f"[INFO] Linking {len(worked_examples)} worked examples to exercises...")

            total_links = 0

            for example in worked_examples:
                # Get topics for this worked example
                example_topics = db.get_topics_for_material(example['id'])

                if not example_topics:
                    print(f"[WARNING] Worked example {example['id'][:40]} has no topics, skipping")
                    continue

                print(f"[DEBUG] Processing example {example['id'][:40]} with {len(example_topics)} topic(s)")

                # Find exercises with same topics
                candidate_exercises = []
                for topic in example_topics:
                    # Get core loops for this topic
                    knowledge_items = db.get_knowledge_items_by_topic(topic['id'])

                    # Get exercises for each core loop
                    for knowledge_item in knowledge_items:
                        exercises = db.get_exercises_by_knowledge_item(knowledge_item['id'])
                        candidate_exercises.extend(exercises)

                # Remove duplicates
                candidate_exercises = {ex['id']: ex for ex in candidate_exercises}.values()
                candidate_exercises = list(candidate_exercises)

                if not candidate_exercises:
                    print(f"[WARNING] No candidate exercises found for topics: {[t['name'] for t in example_topics]}")
                    continue

                print(f"[DEBUG] Found {len(candidate_exercises)} candidate exercises")

                # Rank exercises by similarity to worked example
                similarities = []
                for exercise in candidate_exercises:
                    similarity = self._calculate_text_similarity(
                        example['content'],
                        exercise['text']
                    )
                    similarities.append((exercise['id'], similarity))

                # Sort by similarity (highest first) and take top N
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_matches = similarities[:max_links_per_example]

                # Create links for top matches
                for exercise_id, similarity in top_matches:
                    if similarity >= Config.WORKED_EXAMPLE_EXERCISE_SIMILARITY_THRESHOLD:
                        db.link_material_to_exercise(
                            example['id'],
                            exercise_id,
                            link_type='worked_example'
                        )
                        total_links += 1
                        print(f"[LINK] Example → Exercise {exercise_id[:40]} (similarity: {similarity:.2f})")

            print(f"\n[SUMMARY] Worked example linking complete:")
            print(f"  Created {total_links} worked_example links")

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings.

        Uses semantic embeddings if available, otherwise falls back to string matching.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0.0 to 1.0)
        """
        if self.use_semantic and self.semantic_matcher:
            # Use semantic matcher's embedding-based similarity
            result = self.semantic_matcher.should_merge(text1, text2, threshold=0.0)
            return result.similarity_score
        else:
            # Fallback to string similarity
            similarity, _ = self._similarity(text1, text2)
            return similarity

    # ========================================================================
    # Topic Splitting Methods (Phase 6)
    # ========================================================================

    def detect_generic_topics(self, course_code: str, db: 'Database') -> List[Dict[str, Any]]:
        """Detect generic topics that should be split.

        Args:
            course_code: Course code to check
            db: Database instance

        Returns:
            List of dicts with topic info that should be split:
            [{"id": topic_id, "name": topic_name, "knowledge_item_count": N}, ...]
        """
        generic_topics = []

        # Get all topics for this course
        topics = db.get_topics_by_course(course_code)

        # Get course info for name comparison
        course = db.get_course(course_code)
        course_name = course['name'] if course else ""

        for topic in topics:
            # Get core loops for this topic
            knowledge_items = db.get_knowledge_items_by_topic(topic['id'])
            loop_count = len(knowledge_items)

            # Detection criteria
            is_generic = False
            reason = []

            # Criterion 1: Topic has too many core loops
            if loop_count >= Config.GENERIC_TOPIC_THRESHOLD:
                is_generic = True
                reason.append(f"{loop_count} core loops (threshold: {Config.GENERIC_TOPIC_THRESHOLD})")

            # Criterion 2: Topic name matches or is very similar to course name
            if course_name and self._is_topic_name_generic(topic['name'], course_name):
                is_generic = True
                reason.append(f"topic name '{topic['name']}' is generic/matches course")

            if is_generic:
                generic_topics.append({
                    "id": topic['id'],
                    "name": topic['name'],
                    "knowledge_item_count": loop_count,
                    "knowledge_items": [cl['id'] for cl in knowledge_items],
                    "reason": "; ".join(reason)
                })

        return generic_topics

    def _is_topic_name_generic(self, topic_name: str, course_name: str) -> bool:
        """Check if topic name is too generic compared to course name."""
        # Normalize both names for comparison
        topic_norm = topic_name.lower().strip()
        course_norm = course_name.lower().strip()

        # Check if topic is exactly the course name
        if topic_norm == course_norm:
            return True

        # Check if topic is main words from course name
        # e.g., course "Topic A B" contains topic "Topic A"
        course_words = set(course_norm.split())
        topic_words = set(topic_norm.split())

        # If topic is subset of course words and has < 3 unique words, it's generic
        if topic_words.issubset(course_words) and len(topic_words) < 3:
            return True

        return False

    def cluster_knowledge_items_for_topic(self, topic_id: int, topic_name: str,
                                     knowledge_items: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """Cluster core loops into semantic groups using LLM.

        Args:
            topic_id: ID of generic topic
            topic_name: Name of generic topic
            knowledge_items: List of core loop dicts (with 'id' and 'name' keys)

        Returns:
            List of clusters or None if clustering fails:
            [
                {
                    "topic_name": "Specific Topic Name",
                    "knowledge_item_ids": ["loop1", "loop2", ...]
                },
                ...
            ]
        """
        try:
            print(f"[INFO] Clustering {len(knowledge_items)} core loops for topic '{topic_name}'...")

            # Build core loop list for prompt
            knowledge_item_list = "\n".join([
                f"{i+1}. {cl['name']} (ID: {cl['id']})"
                for i, cl in enumerate(knowledge_items)
            ])

            # Build clustering prompt
            prompt = f"""You are analyzing core loops (procedural problem-solving patterns) from the topic "{topic_name}".

These {len(knowledge_items)} core loops are currently grouped together but are too diverse.
Cluster them into {Config.TOPIC_CLUSTER_MIN}-{Config.TOPIC_CLUSTER_MAX} specific subtopics based on semantic similarity.

Core loops to cluster:
{knowledge_item_list}

Requirements:
- Create {Config.TOPIC_CLUSTER_MIN}-{Config.TOPIC_CLUSTER_MAX} clusters
- Each core loop must appear in exactly ONE cluster
- Give each cluster a specific, descriptive topic name {self._lang_instruction()}
- Topic names should reflect the mathematical/algorithmic concept, NOT be generic
- Group by semantic similarity (what concepts/techniques are being practiced)

Return ONLY valid JSON in this format:
{{
  "clusters": [
    {{
      "topic_name": "Specific Topic Name Here",
      "knowledge_item_ids": ["loop_id_1", "loop_id_2", ...]
    }},
    ...
  ]
}}

No markdown code blocks, just JSON."""

            # Call LLM
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.5,  # Some creativity for grouping
                json_mode=True
            )

            if not response.success:
                print(f"[ERROR] LLM clustering failed: {response.error}")
                return None

            # Parse response
            data = self.llm.parse_json_response(response)
            if not data or 'clusters' not in data:
                print(f"[ERROR] Invalid clustering response format")
                return None

            clusters = data['clusters']

            # Validate clustering
            all_assigned_ids = set()
            for cluster in clusters:
                if 'topic_name' not in cluster or 'knowledge_item_ids' not in cluster:
                    print(f"[ERROR] Invalid cluster format: {cluster}")
                    return None
                all_assigned_ids.update(cluster['knowledge_item_ids'])

            original_ids = set(cl['id'] for cl in knowledge_items)

            # Check if all core loops were assigned
            if all_assigned_ids != original_ids:
                missing = original_ids - all_assigned_ids
                extra = all_assigned_ids - original_ids
                print(f"[WARNING] Clustering validation failed:")
                if missing:
                    print(f"  Missing IDs: {missing}")
                if extra:
                    print(f"  Extra IDs: {extra}")
                return None

            # Check cluster count
            if not (Config.TOPIC_CLUSTER_MIN <= len(clusters) <= Config.TOPIC_CLUSTER_MAX):
                print(f"[WARNING] Cluster count {len(clusters)} outside range {Config.TOPIC_CLUSTER_MIN}-{Config.TOPIC_CLUSTER_MAX}")

            print(f"[INFO] Successfully created {len(clusters)} clusters")
            for i, cluster in enumerate(clusters, 1):
                print(f"  {i}. {cluster['topic_name']}: {len(cluster['knowledge_item_ids'])} core loops")

            return clusters

        except Exception as e:
            print(f"[ERROR] Clustering failed: {e}")
            return None
