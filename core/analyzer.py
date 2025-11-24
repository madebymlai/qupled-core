"""
AI-powered exercise analyzer for Examina.
Handles exercise merging, topic discovery, and core loop identification.
"""

import json
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed

from models.llm_manager import LLMManager, LLMResponse
from storage.database import Database
from config import Config

# Try to import semantic matcher, fallback to string similarity if not available
try:
    from core.semantic_matcher import SemanticMatcher
    SEMANTIC_MATCHING_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] SemanticMatcher not available: {e}")
    print("  Falling back to string-based similarity matching")
    SEMANTIC_MATCHING_AVAILABLE = False


@dataclass
class ProcedureInfo:
    """Information about a single procedure/algorithm in an exercise."""
    name: str
    type: str  # design, transformation, verification, minimization, analysis, other
    steps: List[str]
    point_number: Optional[int] = None
    transformation: Optional[Dict[str, str]] = None  # {"source_format": "X", "target_format": "Y"}


@dataclass
class AnalysisResult:
    """Result of exercise analysis."""
    is_valid_exercise: bool
    is_fragment: bool
    should_merge_with_previous: bool
    topic: Optional[str]
    difficulty: Optional[str]
    variations: Optional[List[str]]
    confidence: float
    procedures: List[ProcedureInfo]  # NEW: Multiple procedures support

    # Backward compatibility fields (derived from first procedure)
    @property
    def core_loop_id(self) -> Optional[str]:
        """Primary core loop ID (first procedure)."""
        if self.procedures:
            return self._normalize_core_loop_id(self.procedures[0].name)
        return None

    @property
    def core_loop_name(self) -> Optional[str]:
        """Primary core loop name (first procedure)."""
        if self.procedures:
            return self.procedures[0].name
        return None

    @property
    def procedure(self) -> Optional[List[str]]:
        """Primary procedure steps (first procedure)."""
        if self.procedures:
            return self.procedures[0].steps
        return None

    @staticmethod
    def _normalize_core_loop_id(core_loop_name: Optional[str]) -> Optional[str]:
        """Normalize core loop name to ID."""
        if not core_loop_name:
            return None
        core_loop_id = core_loop_name.lower()
        core_loop_id = re.sub(r'[^\w\s-]', '', core_loop_id)
        core_loop_id = re.sub(r'[\s-]+', '_', core_loop_id)
        return core_loop_id


class ExerciseAnalyzer:
    """Analyzes exercises using LLM to discover topics and core loops."""

    def __init__(self, llm_manager: Optional[LLMManager] = None, language: str = "en"):
        """Initialize analyzer.

        Args:
            llm_manager: LLM manager instance
            language: Output language for analysis ("en" or "it")
        """
        self.llm = llm_manager or LLMManager()
        self.language = language

        # Initialize semantic matcher if available
        if SEMANTIC_MATCHING_AVAILABLE and Config.SEMANTIC_SIMILARITY_ENABLED:
            try:
                self.semantic_matcher = SemanticMatcher()
                self.use_semantic = self.semantic_matcher.enabled
                if self.use_semantic:
                    print("[INFO] Semantic similarity matching enabled")
                else:
                    print("[INFO] Semantic matcher loaded but model unavailable, using string similarity")
            except Exception as e:
                print(f"[WARNING] Failed to initialize SemanticMatcher: {e}")
                self.semantic_matcher = None
                self.use_semantic = False
        else:
            self.semantic_matcher = None
            self.use_semantic = False
            if not SEMANTIC_MATCHING_AVAILABLE:
                print("[INFO] Semantic matching not available, using string similarity")
            else:
                print("[INFO] Semantic matching disabled in config, using string similarity")

    def analyze_exercise(self, exercise_text: str, course_name: str,
                        previous_exercise: Optional[str] = None) -> AnalysisResult:
        """Analyze a single exercise.

        Args:
            exercise_text: Exercise text
            course_name: Course name for context
            previous_exercise: Previous exercise text (for merge detection)

        Returns:
            AnalysisResult with classification
        """
        # Build prompt
        prompt = self._build_analysis_prompt(
            exercise_text, course_name, previous_exercise
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
        elif "core_loop_name" in data and data["core_loop_name"]:
            # Old format: single procedure - convert to new format
            procedures.append(ProcedureInfo(
                name=data["core_loop_name"],
                type="other",  # Unknown type in old format
                steps=data.get("procedure", []),
                point_number=None,
                transformation=None
            ))

        # Extract fields
        return AnalysisResult(
            is_valid_exercise=data.get("is_valid_exercise", True),
            is_fragment=data.get("is_fragment", False),
            should_merge_with_previous=data.get("should_merge_with_previous", False),
            topic=data.get("topic"),
            difficulty=data.get("difficulty"),
            variations=data.get("variations", []),
            confidence=data.get("confidence", 0.5),
            procedures=procedures
        )

    def _build_analysis_prompt(self, exercise_text: str, course_name: str,
                               previous_exercise: Optional[str]) -> str:
        """Build prompt for exercise analysis.

        Args:
            exercise_text: Exercise text
            course_name: Course name
            previous_exercise: Previous exercise (for merge detection)

        Returns:
            Prompt string
        """
        base_prompt = f"""You are analyzing exam exercises for the course: {course_name}.

Your task is to analyze this text and determine:
1. Is it a valid, complete exercise? Or just exam instructions/headers?
2. Is it a fragment that should be merged with other parts?
3. What topic does it cover?
4. What is the core solving procedure (core loop)?

EXERCISE TEXT:
```
{exercise_text[:2000]}
```
"""

        if previous_exercise:
            base_prompt += f"""
PREVIOUS EXERCISE:
```
{previous_exercise[:1000]}
```

Does this exercise appear to be a continuation or sub-part of the previous one?
"""

        # Add language instruction
        language_instruction = {
            "it": "IMPORTANTE: Rispondi in ITALIANO. Tutti i nomi di topic, procedure e step devono essere in italiano.",
            "en": "IMPORTANT: Respond in ENGLISH. All topic names, procedures and steps must be in English."
        }

        base_prompt += f"""
{language_instruction.get(self.language, language_instruction["en"])}

Respond in JSON format with:
{{
  "is_valid_exercise": true/false,  // false if it's just exam instructions or headers
  "is_fragment": true/false,  // true if incomplete or part of larger exercise
  "should_merge_with_previous": true/false,  // true if continuation of previous
  "topic": "SPECIFIC topic name",  // MUST be specific, NOT generic course name!
  "difficulty": "easy|medium|hard",
  "variations": ["variation1", ...],  // specific variants used
  "confidence": 0.0-1.0,  // your confidence in this analysis
  "procedures": [  // ALL distinct procedures/algorithms required (NEW: can be multiple!)
    {{
      "name": "procedure name",  // e.g., "Mealy Machine Design", "SOP to POS Conversion"
      "type": "design|transformation|verification|minimization|analysis|other",
      "steps": ["step 1", "step 2", ...],  // solving steps for this procedure
      "point_number": 1,  // which numbered point (1, 2, 3, etc.) - null if not applicable
      "transformation": {{  // ONLY if type=transformation
        "source_format": "format name",  // e.g., "Mealy Machine", "SOP"
        "target_format": "format name"   // e.g., "Moore Machine", "POS"
      }}
    }}
  ]
}}

IMPORTANT ANALYSIS GUIDELINES:
- If text contains only exam rules (like "NON si può usare la calcolatrice"), mark as NOT valid exercise
- If text is clearly a sub-question (starts with "1.", "2.") right after numbered list, it's a fragment
- Core loop/procedure is the ALGORITHM/PROCEDURE to solve, not just the topic

TOPIC NAMING RULES (CRITICAL):
- NEVER use the course name "{course_name}" as the topic - it's too generic!
- Topics MUST be SPECIFIC subtopics within the course (e.g., for Linear Algebra: "Autovalori e Diagonalizzazione", "Sottospazi Vettoriali", "Applicazioni Lineari")
- Topics should cluster related procedures together (aim for 3-8 core loops per topic)
- Be as specific as possible - narrow topics are better than broad ones
- If unsure, prefer more specific over more general

MULTI-PROCEDURE DETECTION:
- If exercise has numbered points (1., 2., 3.), analyze EACH point separately
- Each distinct procedure should have its own entry in "procedures" array
- Set "point_number" to indicate which numbered point it belongs to
- If exercise requires multiple procedures (e.g., "design AND verify"), list ALL of them
- For transformations/conversions (Mealy→Moore, SOP→POS, etc.), set type="transformation" and fill "transformation" object

BACKWARD COMPATIBILITY:
- Even if exercise has only ONE procedure, still return it in "procedures" array
- Extract actual solving steps if you can identify them
- The first procedure in the array is considered the PRIMARY procedure

Respond ONLY with valid JSON, no other text.
"""

        return base_prompt

    def _normalize_core_loop_id(self, core_loop_name: Optional[str]) -> Optional[str]:
        """Normalize core loop name to ID.

        Args:
            core_loop_name: Human-readable name

        Returns:
            Normalized ID (lowercase, underscores)
        """
        if not core_loop_name:
            return None

        # Convert to lowercase, replace spaces with underscores
        core_loop_id = core_loop_name.lower()
        core_loop_id = re.sub(r'[^\w\s-]', '', core_loop_id)
        core_loop_id = re.sub(r'[\s-]+', '_', core_loop_id)

        return core_loop_id

    def _default_analysis_result(self) -> AnalysisResult:
        """Return default analysis result on error."""
        return AnalysisResult(
            is_valid_exercise=True,
            is_fragment=False,
            should_merge_with_previous=False,
            topic=None,
            difficulty=None,
            variations=None,
            confidence=0.0,
            procedures=[]  # Empty procedures list
        )

    def merge_exercises(self, exercises: List[Dict[str, Any]], skip_analyzed: bool = False) -> List[Dict[str, Any]]:
        """Merge exercise fragments into complete exercises.

        Args:
            exercises: List of exercise dicts from database
            skip_analyzed: If True, skip exercises already marked as analyzed

        Returns:
            List of merged exercises
        """
        if not exercises:
            return []

        merged = []
        current_merge = None

        for i, exercise in enumerate(exercises):
            # Skip already analyzed exercises if requested
            if skip_analyzed and exercise.get('analyzed'):
                # If we have a current merge, save it before skipping
                if current_merge:
                    merged.append(current_merge)
                    current_merge = None
                print(f"[DEBUG] Skipping already analyzed exercise: {exercise['id'][:40]}...")
                continue

            # Analyze exercise
            previous_text = current_merge["text"] if current_merge else None
            if i > 0 and not current_merge:
                previous_text = exercises[i-1].get("text")

            analysis = self.analyze_exercise(
                exercise["text"],
                "Computer Architecture",  # TODO: get from exercise
                previous_text
            )

            # Skip invalid exercises (exam instructions, etc.)
            if not analysis.is_valid_exercise:
                print(f"[DEBUG] Skipping invalid exercise: {exercise['id'][:20]}... ({exercise['text'][:60]}...)")
                continue

            # Should merge with previous?
            if analysis.should_merge_with_previous and current_merge:
                # Merge into current
                current_merge["text"] += "\n\n" + exercise["text"]
                current_merge["merged_from"].append(exercise["id"])
                if exercise.get("image_paths"):
                    if not current_merge.get("image_paths"):
                        current_merge["image_paths"] = []
                    current_merge["image_paths"].extend(exercise["image_paths"])
            else:
                # Save previous merge if exists
                if current_merge:
                    merged.append(current_merge)

                # Start new exercise (or merge)
                current_merge = {
                    **exercise,
                    "merged_from": [exercise["id"]],
                    "analysis": analysis
                }

        # Don't forget last one
        if current_merge:
            merged.append(current_merge)

        return merged

    def _analyze_exercise_with_retry(self, exercise_text: str, course_name: str,
                                     previous_exercise: Optional[str] = None,
                                     max_retries: int = 2) -> AnalysisResult:
        """Analyze exercise with retry logic for failed API calls.

        Args:
            exercise_text: Exercise text
            course_name: Course name for context
            previous_exercise: Previous exercise text (for merge detection)
            max_retries: Maximum number of retries on failure

        Returns:
            AnalysisResult with classification
        """
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                result = self.analyze_exercise(exercise_text, course_name, previous_exercise)
                # Check if analysis was successful
                if result.confidence > 0.0 or result.topic is not None:
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

    def merge_exercises_parallel(self, exercises: List[Dict[str, Any]],
                                 batch_size: Optional[int] = None,
                                 show_progress: bool = True,
                                 skip_analyzed: bool = False) -> List[Dict[str, Any]]:
        """Merge exercise fragments using parallel batch processing.

        This method analyzes exercises in parallel batches to improve performance.
        Each batch is processed concurrently, with retry logic for failed exercises.

        Args:
            exercises: List of exercise dicts from database
            batch_size: Number of exercises to process in parallel (defaults to Config.BATCH_SIZE)
            show_progress: Show progress bar during processing
            skip_analyzed: If True, skip exercises already marked as analyzed

        Returns:
            List of merged exercises
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
        def analyze_single(index: int, exercise: Dict[str, Any], prev_text: Optional[str]) -> tuple:
            """Analyze a single exercise and return (index, analysis, error)."""
            try:
                # Skip already analyzed if requested
                if skip_analyzed and exercise.get('analyzed'):
                    return (index, None, None)  # Signal to skip

                analysis = self._analyze_exercise_with_retry(
                    exercise["text"],
                    "Computer Architecture",  # TODO: get from exercise
                    prev_text
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

                for i, (idx, exercise) in enumerate(zip(batch_indices, batch)):
                    # Determine previous exercise text for merge detection
                    prev_text = None
                    if idx > 0:
                        prev_text = exercises[idx - 1].get("text")

                    future = executor.submit(analyze_single, idx, exercise, prev_text)
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

        # Now merge exercises sequentially based on analysis results
        print(f"[INFO] Merging exercise fragments...")
        merged = []
        current_merge = None

        for i, exercise in enumerate(exercises):
            # Check if skipped
            if i not in analysis_results:
                if current_merge:
                    merged.append(current_merge)
                    current_merge = None
                continue

            analysis = analysis_results[i]

            # Skip invalid exercises
            if not analysis.is_valid_exercise:
                print(f"[DEBUG] Skipping invalid exercise: {exercise['id'][:20]}... ({exercise['text'][:60]}...)")
                continue

            # Should merge with previous?
            if analysis.should_merge_with_previous and current_merge:
                # Merge into current
                current_merge["text"] += "\n\n" + exercise["text"]
                current_merge["merged_from"].append(exercise["id"])
                if exercise.get("image_paths"):
                    if not current_merge.get("image_paths"):
                        current_merge["image_paths"] = []
                    current_merge["image_paths"].extend(exercise["image_paths"])
            else:
                # Save previous merge if exists
                if current_merge:
                    merged.append(current_merge)

                # Start new exercise
                current_merge = {
                    **exercise,
                    "merged_from": [exercise["id"]],
                    "analysis": analysis
                }

        # Don't forget last one
        if current_merge:
            merged.append(current_merge)

        print(f"[INFO] Merged {len(exercises)} fragments → {len(merged)} complete exercises")

        return merged

    def discover_topics_and_core_loops(self, course_code: str,
                                      batch_size: int = 10,
                                      skip_analyzed: bool = False,
                                      use_parallel: bool = True) -> Dict[str, Any]:
        """Discover topics and core loops for a course.

        Args:
            course_code: Course code
            batch_size: Number of exercises to analyze at once
            skip_analyzed: If True, skip already analyzed exercises
            use_parallel: If True, use parallel batch processing (default: True)

        Returns:
            Dict with topics and core loops discovered
        """
        with Database() as db:
            # Get all exercises for course
            exercises = db.get_exercises_by_course(course_code)

            if not exercises:
                return {"topics": {}, "core_loops": {}}

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
            topics = {}
            core_loops = {}
            low_confidence_count = 0

            for merged_ex in merged_exercises:
                analysis = merged_ex.get("analysis")
                if not analysis:
                    continue

                # Skip low-confidence analyses
                if analysis.confidence < Config.MIN_ANALYSIS_CONFIDENCE:
                    low_confidence_count += 1
                    print(f"[INFO] Skipping exercise due to low confidence ({analysis.confidence:.2f} < {Config.MIN_ANALYSIS_CONFIDENCE}): {merged_ex['id'][:40]}...")
                    # Mark exercise as skipped in metadata
                    merged_ex["low_confidence_skipped"] = True
                    continue

                # Track topic
                if analysis.topic:
                    if analysis.topic not in topics:
                        topics[analysis.topic] = {
                            "name": analysis.topic,
                            "exercise_count": 0,
                            "core_loops": set()
                        }
                    topics[analysis.topic]["exercise_count"] += 1

                # Process ALL procedures (new multi-procedure support)
                if analysis.procedures:
                    # Log if multiple procedures detected
                    if len(analysis.procedures) > 1:
                        print(f"[INFO] Multiple procedures detected ({len(analysis.procedures)}) in exercise {merged_ex['id'][:40]}:")
                        for i, proc in enumerate(analysis.procedures, 1):
                            print(f"  {i}. {proc.name} (type: {proc.type}, point: {proc.point_number})")

                    # Process each procedure
                    for procedure_info in analysis.procedures:
                        core_loop_id = self._normalize_core_loop_id(procedure_info.name)

                        # Track core loop under topic
                        if analysis.topic and core_loop_id:
                            topics[analysis.topic]["core_loops"].add(core_loop_id)

                        # Track core loop
                        if core_loop_id and procedure_info.name:
                            if core_loop_id not in core_loops:
                                core_loops[core_loop_id] = {
                                    "id": core_loop_id,
                                    "name": procedure_info.name,
                                    "topic": analysis.topic,
                                    "procedure": procedure_info.steps or [],
                                    "type": procedure_info.type,
                                    "transformation": procedure_info.transformation,
                                    "exercise_count": 0,
                                    "exercises": []
                                }
                            core_loops[core_loop_id]["exercise_count"] += 1
                            if merged_ex["id"] not in core_loops[core_loop_id]["exercises"]:
                                core_loops[core_loop_id]["exercises"].append(merged_ex["id"])

            # Convert sets to lists for JSON serialization
            for topic_data in topics.values():
                topic_data["core_loops"] = list(topic_data["core_loops"])

            # Deduplicate against existing database entries first, then within batch
            topics = self._deduplicate_topics_with_database(topics, course_code, db)
            core_loops = self._deduplicate_core_loops_with_database(core_loops, course_code, db)

            # Log summary statistics
            accepted_count = len(merged_exercises) - low_confidence_count
            if low_confidence_count > 0:
                print(f"\n[SUMMARY] Confidence Filtering Results:")
                print(f"  Total merged exercises: {len(merged_exercises)}")
                print(f"  Accepted (>= {Config.MIN_ANALYSIS_CONFIDENCE} confidence): {accepted_count}")
                print(f"  Skipped (low confidence): {low_confidence_count}")
                print(f"  Skip rate: {(low_confidence_count / len(merged_exercises) * 100):.1f}%\n")

            return {
                "topics": topics,
                "core_loops": core_loops,
                "merged_exercises": merged_exercises,
                "original_count": len(exercises),
                "merged_count": len(merged_exercises),
                "low_confidence_skipped": low_confidence_count,
                "accepted_count": accepted_count
            }

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

    def _deduplicate_topics(self, topics: Dict[str, Any]) -> Dict[str, Any]:
        """Deduplicate similar topics using semantic similarity.

        Args:
            topics: Dictionary of topics

        Returns:
            Deduplicated topics dictionary
        """
        if len(topics) <= 1:
            return topics

        threshold = Config.SEMANTIC_SIMILARITY_THRESHOLD if self.use_semantic else Config.CORE_LOOP_SIMILARITY_THRESHOLD
        topic_names = list(topics.keys())
        merged_topics = {}
        skip_topics = set()

        # Track mapping from old topic names to canonical names
        self.topic_name_mapping = {}

        for i, topic1 in enumerate(topic_names):
            if topic1 in skip_topics:
                continue

            # Start with this topic
            canonical_topic = topic1
            canonical_data = topics[topic1].copy()

            # Map canonical topic to itself
            self.topic_name_mapping[canonical_topic] = canonical_topic

            # Check for similar topics
            for topic2 in topic_names[i+1:]:
                if topic2 in skip_topics:
                    continue

                # Use semantic matching if available
                if self.use_semantic and self.semantic_matcher:
                    result = self.semantic_matcher.should_merge(topic1, topic2, threshold)
                    if result.should_merge:
                        print(f"[DEDUP] Topic '{topic1}' → '{topic2}' (similarity: {result.similarity_score:.2f}, reason: {result.reason})")
                        # Merge topic2 into canonical
                        canonical_data["exercise_count"] += topics[topic2]["exercise_count"]
                        canonical_data["core_loops"] = list(set(canonical_data["core_loops"]) | set(topics[topic2]["core_loops"]))
                        skip_topics.add(topic2)
                        # Map merged topic to canonical
                        self.topic_name_mapping[topic2] = canonical_topic
                    elif Config.SEMANTIC_LOG_NEAR_MISSES and result.similarity_score >= 0.80:
                        print(f"[SKIP] Topic '{topic1}' ≠ '{topic2}' (similarity: {result.similarity_score:.2f}, reason: {result.reason})")
                else:
                    # Fallback to string similarity
                    similarity, reason = self._similarity(topic1, topic2)
                    if similarity >= threshold:
                        print(f"[DEBUG] Merging similar topics: '{topic1}' ≈ '{topic2}' (similarity: {similarity:.2f})")
                        # Merge topic2 into canonical
                        canonical_data["exercise_count"] += topics[topic2]["exercise_count"]
                        canonical_data["core_loops"] = list(set(canonical_data["core_loops"]) | set(topics[topic2]["core_loops"]))
                        skip_topics.add(topic2)
                        # Map merged topic to canonical
                        self.topic_name_mapping[topic2] = canonical_topic

            merged_topics[canonical_topic] = canonical_data

        return merged_topics

    def _deduplicate_core_loops(self, core_loops: Dict[str, Any]) -> Dict[str, Any]:
        """Deduplicate similar core loops using semantic similarity.

        Args:
            core_loops: Dictionary of core loops

        Returns:
            Tuple of (deduplicated core loops dictionary, ID mapping dict)
        """
        if len(core_loops) <= 1:
            return core_loops

        threshold = Config.SEMANTIC_SIMILARITY_THRESHOLD if self.use_semantic else Config.CORE_LOOP_SIMILARITY_THRESHOLD
        loop_ids = list(core_loops.keys())
        merged_loops = {}
        skip_loops = set()

        # Track mapping from old IDs to canonical IDs
        self.core_loop_id_mapping = {}

        for i, loop1_id in enumerate(loop_ids):
            if loop1_id in skip_loops:
                continue

            loop1 = core_loops[loop1_id]
            canonical_id = loop1_id
            canonical_data = loop1.copy()

            # Map canonical ID to itself
            self.core_loop_id_mapping[canonical_id] = canonical_id

            # Check for similar core loops (compare names, not IDs)
            for loop2_id in loop_ids[i+1:]:
                if loop2_id in skip_loops:
                    continue

                loop2 = core_loops[loop2_id]

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
                        self.core_loop_id_mapping[loop2_id] = canonical_id
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
                        self.core_loop_id_mapping[loop2_id] = canonical_id

            merged_loops[canonical_id] = canonical_data

        return merged_loops

    def _deduplicate_topics_with_database(self, topics: Dict[str, Any],
                                          course_code: str,
                                          db) -> Dict[str, Any]:
        """Deduplicate topics against existing database entries, then within batch.

        Args:
            topics: Dictionary of new topics from current analysis
            course_code: Course code
            db: Database instance

        Returns:
            Deduplicated topics dictionary with mappings to existing DB topics
        """
        threshold = Config.SEMANTIC_SIMILARITY_THRESHOLD if self.use_semantic else Config.CORE_LOOP_SIMILARITY_THRESHOLD

        # Load existing topics from database
        existing_topics = db.get_topics_by_course(course_code)
        existing_topic_map = {t['name']: t for t in existing_topics}

        # Track mappings from new topic names to canonical (db or batch) names
        topic_mapping = {}
        deduplicated_topics = {}

        for new_topic_name, new_topic_data in topics.items():
            matched_existing = None
            best_similarity = 0.0
            best_reason = ""

            # Check against existing database topics first
            for existing_name in existing_topic_map.keys():
                if self.use_semantic and self.semantic_matcher:
                    result = self.semantic_matcher.should_merge(
                        new_topic_name, existing_name, threshold
                    )
                    if result.should_merge and result.similarity_score > best_similarity:
                        best_similarity = result.similarity_score
                        matched_existing = existing_name
                        best_reason = result.reason
                else:
                    similarity, reason = self._similarity(new_topic_name, existing_name)
                    if similarity >= threshold and similarity > best_similarity:
                        best_similarity = similarity
                        matched_existing = existing_name
                        best_reason = reason

            if matched_existing:
                # Reuse existing topic
                print(f"[DEDUP] Topic '{new_topic_name}' → existing '{matched_existing}' (similarity: {best_similarity:.2f}, reason: {best_reason})")
                topic_mapping[new_topic_name] = matched_existing
                # Don't add to deduplicated_topics, we'll use DB entry
            else:
                # New topic, add to batch
                deduplicated_topics[new_topic_name] = new_topic_data
                topic_mapping[new_topic_name] = new_topic_name

        # Now deduplicate within the new batch
        deduplicated_topics = self._deduplicate_topics(deduplicated_topics)

        # Update topic mapping with any batch deduplication
        if hasattr(self, 'topic_name_mapping'):
            for old_name, canonical_name in self.topic_name_mapping.items():
                if old_name in topic_mapping:
                    topic_mapping[old_name] = canonical_name

        # Store the mapping for later use
        self.topic_name_mapping = topic_mapping

        return deduplicated_topics

    def _deduplicate_core_loops_with_database(self, core_loops: Dict[str, Any],
                                              course_code: str,
                                              db) -> Dict[str, Any]:
        """Deduplicate core loops against existing database entries, then within batch.

        Args:
            core_loops: Dictionary of new core loops from current analysis
            course_code: Course code
            db: Database instance

        Returns:
            Deduplicated core loops dictionary with mappings to existing DB loops
        """
        threshold = Config.SEMANTIC_SIMILARITY_THRESHOLD if self.use_semantic else Config.CORE_LOOP_SIMILARITY_THRESHOLD

        # Load existing core loops from database
        existing_loops = db.get_core_loops_by_course(course_code)
        existing_loop_map = {loop['id']: loop for loop in existing_loops}

        # Track mappings from new loop IDs to canonical (db or batch) IDs
        loop_id_mapping = {}
        deduplicated_loops = {}

        for new_loop_id, new_loop_data in core_loops.items():
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
        deduplicated_loops = self._deduplicate_core_loops(deduplicated_loops)

        # Update loop mapping with any batch deduplication
        if hasattr(self, 'core_loop_id_mapping'):
            for old_id, canonical_id in self.core_loop_id_mapping.items():
                if old_id in loop_id_mapping:
                    loop_id_mapping[old_id] = canonical_id

        # Store the mapping for later use
        self.core_loop_id_mapping = loop_id_mapping

        return deduplicated_loops

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
            [{"id": topic_id, "name": topic_name, "core_loop_count": N}, ...]
        """
        generic_topics = []

        # Get all topics for this course
        topics = db.get_topics_by_course(course_code)

        # Get course info for name comparison
        course = db.get_course(course_code)
        course_name = course['name'] if course else ""

        for topic in topics:
            # Get core loops for this topic
            core_loops = db.get_core_loops_by_topic(topic['id'])
            loop_count = len(core_loops)

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
                    "core_loop_count": loop_count,
                    "core_loops": [cl['id'] for cl in core_loops],
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
        # e.g., "Algebra Lineare" contains "Algebra"
        course_words = set(course_norm.split())
        topic_words = set(topic_norm.split())

        # If topic is subset of course words and has < 3 unique words, it's generic
        if topic_words.issubset(course_words) and len(topic_words) < 3:
            return True

        return False

    def cluster_core_loops_for_topic(self, topic_id: int, topic_name: str,
                                     core_loops: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """Cluster core loops into semantic groups using LLM.

        Args:
            topic_id: ID of generic topic
            topic_name: Name of generic topic
            core_loops: List of core loop dicts (with 'id' and 'name' keys)

        Returns:
            List of clusters or None if clustering fails:
            [
                {
                    "topic_name": "Specific Topic Name",
                    "core_loop_ids": ["loop1", "loop2", ...]
                },
                ...
            ]
        """
        try:
            print(f"[INFO] Clustering {len(core_loops)} core loops for topic '{topic_name}'...")

            # Build core loop list for prompt
            core_loop_list = "\n".join([
                f"{i+1}. {cl['name']} (ID: {cl['id']})"
                for i, cl in enumerate(core_loops)
            ])

            # Build clustering prompt
            lang_instruction = "in Italian" if self.language == "it" else "in English"
            prompt = f"""You are analyzing core loops (procedural problem-solving patterns) from the topic "{topic_name}".

These {len(core_loops)} core loops are currently grouped together but are too diverse.
Cluster them into {Config.TOPIC_CLUSTER_MIN}-{Config.TOPIC_CLUSTER_MAX} specific subtopics based on semantic similarity.

Core loops to cluster:
{core_loop_list}

Requirements:
- Create {Config.TOPIC_CLUSTER_MIN}-{Config.TOPIC_CLUSTER_MAX} clusters
- Each core loop must appear in exactly ONE cluster
- Give each cluster a specific, descriptive topic name {lang_instruction}
- Topic names should reflect the mathematical/algorithmic concept, NOT be generic
- Group by semantic similarity (what concepts/techniques are being practiced)

Return ONLY valid JSON in this format:
{{
  "clusters": [
    {{
      "topic_name": "Specific Topic Name Here",
      "core_loop_ids": ["loop_id_1", "loop_id_2", ...]
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
                if 'topic_name' not in cluster or 'core_loop_ids' not in cluster:
                    print(f"[ERROR] Invalid cluster format: {cluster}")
                    return None
                all_assigned_ids.update(cluster['core_loop_ids'])

            original_ids = set(cl['id'] for cl in core_loops)

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
                print(f"  {i}. {cluster['topic_name']}: {len(cluster['core_loop_ids'])} core loops")

            return clusters

        except Exception as e:
            print(f"[ERROR] Clustering failed: {e}")
            return None
