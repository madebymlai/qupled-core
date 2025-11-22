"""
AI-powered exercise analyzer for Examina.
Handles exercise merging, topic discovery, and core loop identification.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from models.llm_manager import LLMManager, LLMResponse
from storage.database import Database
from config import Config


@dataclass
class AnalysisResult:
    """Result of exercise analysis."""
    is_valid_exercise: bool
    is_fragment: bool
    should_merge_with_previous: bool
    topic: Optional[str]
    core_loop_id: Optional[str]
    core_loop_name: Optional[str]
    procedure: Optional[List[str]]
    difficulty: Optional[str]
    variations: Optional[List[str]]
    confidence: float


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
            return AnalysisResult(
                is_valid_exercise=True,  # Assume valid if we can't analyze
                is_fragment=False,
                should_merge_with_previous=False,
                topic=None,
                core_loop_id=None,
                core_loop_name=None,
                procedure=None,
                difficulty=None,
                variations=None,
                confidence=0.0
            )

        # Parse JSON response
        data = self.llm.parse_json_response(response)
        if not data:
            return self._default_analysis_result()

        # Extract fields
        return AnalysisResult(
            is_valid_exercise=data.get("is_valid_exercise", True),
            is_fragment=data.get("is_fragment", False),
            should_merge_with_previous=data.get("should_merge_with_previous", False),
            topic=data.get("topic"),
            core_loop_id=self._normalize_core_loop_id(data.get("core_loop_name")),
            core_loop_name=data.get("core_loop_name"),
            procedure=data.get("procedure", []),
            difficulty=data.get("difficulty"),
            variations=data.get("variations", []),
            confidence=data.get("confidence", 0.5)
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
  "topic": "topic name",  // e.g., "Sequential Circuits", "Boolean Algebra"
  "core_loop_name": "procedure name",  // e.g., "Mealy Machine Design", "Karnaugh Maps"
  "procedure": ["step 1", "step 2", ...],  // solving steps
  "difficulty": "easy|medium|hard",
  "variations": ["variation1", ...],  // specific variants used
  "confidence": 0.0-1.0  // your confidence in this analysis
}}

IMPORTANT:
- If text contains only exam rules (like "NON si può usare la calcolatrice"), mark as NOT valid exercise
- If text is clearly a sub-question (starts with "1.", "2.") right after numbered list, it's a fragment
- Core loop is the ALGORITHM/PROCEDURE to solve, not just the topic
- Extract actual solving steps if you can identify them

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
            core_loop_id=None,
            core_loop_name=None,
            procedure=None,
            difficulty=None,
            variations=None,
            confidence=0.0
        )

    def merge_exercises(self, exercises: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge exercise fragments into complete exercises.

        Args:
            exercises: List of exercise dicts from database

        Returns:
            List of merged exercises
        """
        if not exercises:
            return []

        merged = []
        current_merge = None

        for i, exercise in enumerate(exercises):
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

    def discover_topics_and_core_loops(self, course_code: str,
                                      batch_size: int = 10) -> Dict[str, Any]:
        """Discover topics and core loops for a course.

        Args:
            course_code: Course code
            batch_size: Number of exercises to analyze at once

        Returns:
            Dict with topics and core loops discovered
        """
        with Database() as db:
            # Get all exercises for course
            exercises = db.get_exercises_by_course(course_code)

            if not exercises:
                return {"topics": {}, "core_loops": {}}

            # Merge fragments first
            merged_exercises = self.merge_exercises(exercises)

            # Collect all analyses
            topics = {}
            core_loops = {}

            for merged_ex in merged_exercises:
                analysis = merged_ex.get("analysis")
                if not analysis:
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

                    # Track core loop under topic
                    if analysis.core_loop_id:
                        topics[analysis.topic]["core_loops"].add(analysis.core_loop_id)

                # Track core loop
                if analysis.core_loop_id and analysis.core_loop_name:
                    if analysis.core_loop_id not in core_loops:
                        core_loops[analysis.core_loop_id] = {
                            "id": analysis.core_loop_id,
                            "name": analysis.core_loop_name,
                            "topic": analysis.topic,
                            "procedure": analysis.procedure or [],
                            "exercise_count": 0,
                            "exercises": []
                        }
                    core_loops[analysis.core_loop_id]["exercise_count"] += 1
                    core_loops[analysis.core_loop_id]["exercises"].append(merged_ex["id"])

            # Convert sets to lists for JSON serialization
            for topic_data in topics.values():
                topic_data["core_loops"] = list(topic_data["core_loops"])

            return {
                "topics": topics,
                "core_loops": core_loops,
                "merged_exercises": merged_exercises,
                "original_count": len(exercises),
                "merged_count": len(merged_exercises)
            }
