"""
AI-powered exercise analyzer for Examina.
Extracts knowledge items from exercises for spaced repetition learning.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from models.llm_manager import LLMManager

logger = logging.getLogger(__name__)

# Learning approaches - how to best teach this knowledge
LEARNING_APPROACHES = {
    "procedural": "Problem requiring step-by-step solution. Student must show work.",
    "conceptual": "Question about principles. Student must explain reasoning.",
    "factual": "Question testing recall. Student must state specific information.",
    "analytical": "Scenario-based question. Student must analyze and conclude.",
}


@dataclass
class KnowledgeItemInfo:
    """Unified knowledge item extracted from exercise analysis."""

    name: str  # snake_case identifier
    learning_approach: Optional[str] = None  # procedural, conceptual, factual, analytical


@dataclass
class AnalysisResult:
    """Result of exercise analysis."""

    is_valid_exercise: bool
    is_fragment: bool
    should_merge_with_previous: bool
    difficulty: Optional[str]
    confidence: float
    knowledge_items: Optional[List["KnowledgeItemInfo"]] = None

    @staticmethod
    def _normalize_name(name: Optional[str]) -> Optional[str]:
        """Normalize knowledge item name to snake_case ID."""
        if not name:
            return None
        normalized = name.lower()
        normalized = re.sub(r"[^\w\s-]", "", normalized)
        normalized = re.sub(r"[\s-]+", "_", normalized)
        return normalized


class ExerciseAnalyzer:
    """Analyzes exercises using LLM to extract knowledge items."""

    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        language: str = "en",
        **kwargs,  # Accept but ignore legacy params (monolingual, procedure_cache, use_cache, etc.)
    ):
        """Initialize analyzer.

        Args:
            llm_manager: LLM manager instance
            language: Output language for analysis (ISO 639-1 code)
        """
        self.llm = llm_manager or LLMManager()
        self.language = language

    def _language_instruction(self, action: str = "Respond") -> str:
        """Generate dynamic language instruction for any language."""
        return f"{action} in {self.language.upper()} language."

    def _language_name(self) -> str:
        """Get full language name for prompts."""
        return self.language.upper()

    def analyze_exercise(
        self,
        exercise_text: str,
        course_name: str,
        exercise_context: Optional[str] = None,
        is_sub_question: bool = False,
    ) -> AnalysisResult:
        """Analyze a single exercise.

        Args:
            exercise_text: Exercise text
            course_name: Course name for context
            exercise_context: Optional context (parent context for subs, exercise summary for standalone)
            is_sub_question: Whether this is a sub-question (affects prompt wording)

        Returns:
            AnalysisResult with classification
        """
        prompt = self._build_analysis_prompt(
            exercise_text, course_name, exercise_context, is_sub_question
        )

        response = self.llm.generate(
            prompt=prompt,
            model=self.llm.primary_model,
            temperature=0.3,
            json_mode=True,
        )

        if not response.success:
            print(f"[ERROR] LLM failed for exercise: {response.error}")
            print(f"  Text preview: {exercise_text[:100]}...")
            return self._default_analysis_result()

        data = self.llm.parse_json_response(response)
        if not data:
            return self._default_analysis_result()

        knowledge_items = []
        if "knowledge_item" in data and data["knowledge_item"]:
            item_data = data["knowledge_item"]
            knowledge_items.append(
                KnowledgeItemInfo(
                    name=item_data.get("name", "unknown"),
                    learning_approach=item_data.get("learning_approach"),
                )
            )

        return AnalysisResult(
            is_valid_exercise=data.get("is_valid_exercise", True),
            is_fragment=data.get("is_fragment", False),
            should_merge_with_previous=data.get("should_merge_with_previous", False),
            difficulty=data.get("difficulty"),
            confidence=data.get("confidence", 0.5),
            knowledge_items=knowledge_items if knowledge_items else None,
        )

    def _build_analysis_prompt(
        self,
        exercise_text: str,
        course_name: str,
        exercise_context: Optional[str] = None,
        is_sub_question: bool = False,
    ) -> str:
        """Build prompt for exercise analysis."""
        base_prompt = f"""You are analyzing exam exercises for the course: {course_name}.

Extract the **knowledge item** (core skill/concept) being tested.

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

        learning_approaches_str = "|".join(LEARNING_APPROACHES.keys())
        learning_approaches_desc = "\n".join(
            f"- **{k}** = {v}" for k, v in LEARNING_APPROACHES.items()
        )

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

**KNOWLEDGE ITEM** (the **ONE** core skill being tested):
- Ask: "If a student fails this exercise, what **specific skill** are they missing?"
- Ask: "What would this exercise be called in a study guide?"
- Name the **CONCEPT** being tested, **not the task performed**
- Think: "What textbook chapter covers this?"
- Name should make sense outside this exercise context
- **snake_case**, e.g., "matrix_multiplication"
- If **multiple concepts**, pick the **primary one**

**LEARNING APPROACH**:
{learning_approaches_desc}

**CONTEXT EXCLUSION**:
- Extract **ONLY** course concepts, **NOT** word problem scenarios
- Test: "Does this have a formal definition/procedure in the course, or just scenery?"

Respond ONLY with valid JSON.
"""

        return base_prompt

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


def generate_item_description(
    exercises: list[dict],
    llm: LLMManager,
) -> str:
    """
    Generate a chapter subtitle from exercises using R1.

    Uses textbook editor mindset for concise, clear descriptions (~100 chars).

    Args:
        exercises: List of exercise dicts with keys: text, is_sub, context
        llm: LLMManager instance

    Returns:
        Chapter subtitle string (falls back to first exercise text on error)
    """
    if not exercises:
        return ""

    exercises_text = []
    for ex in exercises[:6]:
        context = ex.get("context", "")
        text = ex.get("text", "")
        if ex.get("is_sub") and context:
            exercises_text.append(f"Context: {context}\nQuestion: {text}")
        else:
            exercises_text.append(context or text)

    exercises_text = [t for t in exercises_text if t]
    if not exercises_text:
        return ""

    system = "You are a textbook editor."

    prompt = f"""Describe in English the skill/concept being tested (**start with a verb**, no colons):

{chr(10).join(f"- {t}" for t in exercises_text)}

Return JSON: {{"description": "..."}}"""

    try:
        response = llm.generate(prompt=prompt, model="deepseek-reasoner", system=system)
        if response and response.text:
            result = json.loads(response.text)
            return result.get("description", exercises_text[0][:100])
        return exercises_text[0][:100]
    except Exception as e:
        logger.warning(f"Description generation failed: {e}")
        return exercises_text[0][:100] if exercises_text else ""
