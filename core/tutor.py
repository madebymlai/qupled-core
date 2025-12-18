"""
Interactive AI tutor for Examina.
Provides learning features for KnowledgeItems.
"""

import re
import random
from typing import List, Dict, Any, Optional

from models.llm_manager import LLMManager
from config import Config


def get_language_name(code: str) -> str:
    """Get language instruction string for LLM prompts.

    Uses explicit phrasing that LLMs understand without hardcoded mapping.
    LLMs are trained on ISO 639-1 codes and understand them in context.
    """
    # LLMs understand "the language with code X" unambiguously
    # This avoids confusing cases like "in it" being parsed as English "it"
    return f"the language with ISO 639-1 code '{code}'"


# Philosophy: "The Smartest Kid in the Library" - warm, calm, insider knowledge
# LaTeX formatting: Use $...$ for inline math, $$...$$ for display/block math

# Shared LaTeX instruction for all prompts
LATEX_INSTRUCTION = "Use LaTeX: $...$ inline, $$...$$ display."

# Section types per learning_approach
SECTIONS_BY_APPROACH = {
    "factual": ["overview", "fact", "context", "memory_aid"],
    "conceptual": ["overview", "definition", "exam_patterns", "common_mistakes"],
    "procedural": ["overview", "when_to_use", "steps", "worked_example", "watch_out"],
    "analytical": ["overview", "approach", "worked_example", "scoring_tips"],
}

# Max tokens per section type (safety caps - R1 auto-regulates length)
SECTION_MAX_TOKENS = {
    "overview": 500,
    "fact": 500,
    "definition": 800,
    "when_to_use": 800,
    "context": 800,
    "steps": 1500,
    "approach": 1500,
    "worked_example": 2500,
    "watch_out": 800,
    "common_mistakes": 800,
    "memory_aid": 800,
    "scoring_tips": 800,
    "exam_patterns": 800,
}

# Prompt version for cache invalidation - bump when prompts change
SECTION_PROMPT_VERSION = 2

# Section-by-section prompts for waterfall learn mode (R1 optimized)
# Simple prompts - system prompt handles role, let R1 think
SECTION_PROMPTS = {
    "procedural": {
        "overview": f"""~80 words. What problem does this procedure solve? When would you use it in an exam?

Bold **key terms** you'd highlight for your friend.
{LATEX_INSTRUCTION}""",
        "when_to_use": f"""~150 words. What clues in exam questions signal this procedure? What keywords or patterns should trigger "use this method"?

Bold **key terms** you'd highlight for your friend.
{LATEX_INSTRUCTION}""",
        "steps": f"""~500 words. Teach the steps thoroughly. Explain the reasoning.

Bold **key terms** you'd highlight for your friend.
{LATEX_INSTRUCTION}""",
        "worked_example": f"""~600 words. Solve the exercise step by step. Show your work and reasoning.

Bold **key terms** you'd highlight for your friend.
{LATEX_INSTRUCTION}""",
        "watch_out": f"""~200 words. Common traps professors set in exams and mistakes students make. How to avoid them.

Bold **key terms** you'd highlight for your friend.
{LATEX_INSTRUCTION}""",
    },
    "conceptual": {
        "overview": f"""~50 words. What is this concept about?

Bold **key terms** you'd highlight for your friend.
{LATEX_INSTRUCTION}""",
        "definition": f"""~300 words. Define the concept formally and explain the intuition in plain language.

Bold **key terms** you'd highlight for your friend.
{LATEX_INSTRUCTION}""",
        "exam_patterns": f"""~200 words. How do professors test this concept? What question formats appear in exams?

Bold **key terms** you'd highlight for your friend.
{LATEX_INSTRUCTION}""",
        "common_mistakes": f"""~200 words. Common traps professors set in exams and mistakes students make. How to avoid them.

Bold **key terms** you'd highlight for your friend.
{LATEX_INSTRUCTION}""",
    },
    "factual": {
        "overview": f"""~50 words. Introduce this topic. Why does it matter for the exam?

Bold **key terms** you'd highlight for your friend.
{LATEX_INSTRUCTION}""",
        "fact": f"""~150 words. State the fact clearly and explain why it's true.

Bold **key terms** you'd highlight for your friend.
{LATEX_INSTRUCTION}""",
        "context": f"""~200 words. When and where does this appear in exams? What does it connect to?

Bold **key terms** you'd highlight for your friend.
{LATEX_INSTRUCTION}""",
        "memory_aid": f"""~150 words. A mnemonic or memory trick to remember this.

Bold **key terms** you'd highlight for your friend.
{LATEX_INSTRUCTION}""",
    },
    "analytical": {
        "overview": f"""~100 words. What type of problem is this? What makes it challenging?

Bold **key terms** you'd highlight for your friend.
{LATEX_INSTRUCTION}""",
        "approach": f"""~300 words. How to think about this type of problem. What framework or strategy to use.

Bold **key terms** you'd highlight for your friend.
{LATEX_INSTRUCTION}""",
        "worked_example": f"""~600 words. Solve the exercise showing your full reasoning process.

Bold **key terms** you'd highlight for your friend.
{LATEX_INSTRUCTION}""",
        "scoring_tips": f"""~150 words. How to maximize your score. What graders look for.

Bold **key terms** you'd highlight for your friend.
{LATEX_INSTRUCTION}""",
    },
}

# Map which sections need context from previous sections
SECTION_CONTEXT_DEPENDENCIES = {
    "procedural": {
        "worked_example": "steps",  # worked example needs steps content
        "watch_out": "steps",  # watch out references steps
    },
    "analytical": {
        "worked_example": "approach",  # worked example needs approach content
    },
    # conceptual and factual don't need context passing
}


class Tutor:
    """AI tutor for learning core loops and practicing exercises."""

    def __init__(self, llm_manager: Optional[LLMManager] = None, language: str = "en"):
        """Initialize tutor.

        Args:
            llm_manager: LLM manager instance
            language: Output language (any ISO 639-1 code, e.g., "en", "de", "zh")
        """
        self.llm = llm_manager or LLMManager(provider=Config.LLM_PROVIDER)
        self.language = language

    @property
    def llm_manager(self) -> LLMManager:
        """Alias for llm for backward compatibility with cloud."""
        return self.llm

    def _language_instruction(self, action: str = "Respond") -> str:
        """Generate dynamic language instruction for any language.

        Args:
            action: The action verb (e.g., "Respond", "Create", "Explain")

        Returns:
            Language instruction string that works for any ISO 639-1 code
        """
        # LLM understands any ISO 639-1 language code
        return f"{action} in {self.language.upper()} language."

    def learn_section(
        self,
        knowledge_item: Dict[str, Any],
        section_name: str,
        section_index: int,
        exercises: List[Dict[str, Any]],
        previous_section_content: Optional[str] = None,
        notes: Optional[List[str]] = None,
        parent_exercise_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a single section for waterfall learn mode.

        Each section is generated independently with a focused prompt.

        Args:
            knowledge_item: KnowledgeItem dict with id, name, learning_approach, content
            section_name: Name of section to generate (e.g., "overview", "steps", "worked_example")
            section_index: Index of this section (0-based)
            exercises: List of linked exercise dicts for examples
            previous_section_content: Optional content from a previous section (for context dependencies)
            notes: Optional list of user's note content strings (PRO users)
            parent_exercise_context: Optional parent exercise text for sub-questions

        Returns:
            Dict with section content and metadata
        """

        # Get learning_approach (default to conceptual)
        learning_approach = knowledge_item.get("learning_approach", "conceptual").lower()
        if learning_approach not in SECTION_PROMPTS:
            learning_approach = "conceptual"

        # Get section prompts for this approach
        approach_prompts = SECTION_PROMPTS.get(learning_approach, {})

        # Get the specific section prompt
        section_prompt = approach_prompts.get(section_name)
        if not section_prompt:
            return {
                "content": f"Unknown section: {section_name}",
                "section_name": section_name,
                "section_index": section_index,
                "error": True,
            }

        # Get total sections for this approach
        sections_list = list(approach_prompts.keys())
        total_sections = len(sections_list)

        # Select example exercise for worked example section
        example_exercise = None
        if "example" in section_name.lower() and exercises:
            example_exercise = self._select_example_exercise(exercises)

        # Build the prompt
        prompt = self._build_section_prompt(
            knowledge_item=knowledge_item,
            section_prompt=section_prompt,
            section_name=section_name,
            example_exercise=example_exercise,
            previous_section_content=previous_section_content,
            notes=notes,
            parent_exercise_context=parent_exercise_context,
        )

        # System prompt: role only (R1 best practice)
        system = "You are the smartest student in the library helping a friend before their exam."

        # Call LLM with R1 for better teaching quality
        # Note: no max_tokens - deepseek-reasoner self-regulates output length
        response = self.llm.generate(
            prompt=prompt,
            system=system,
            model="deepseek-reasoner",
        )

        if not response.success:
            return {
                "content": f"Could not generate section: {response.error}",
                "section_name": section_name,
                "section_index": section_index,
                "total_sections": total_sections,
                "learning_approach": learning_approach,
                "error": True,
            }

        return {
            "content": response.text,
            "section_name": section_name,
            "section_index": section_index,
            "total_sections": total_sections,
            "is_last": section_index == total_sections - 1,
            "learning_approach": learning_approach,
            "error": False,
        }

    def _build_section_prompt(
        self,
        knowledge_item: Dict[str, Any],
        section_prompt: str,
        section_name: str,
        example_exercise: Optional[Dict[str, Any]],
        previous_section_content: Optional[str],
        notes: Optional[List[str]],
        parent_exercise_context: Optional[str],
    ) -> str:
        """Build LLM prompt for a single section."""
        import json

        # Build language instruction
        if self.language and self.language.lower() != "en":
            lang_name = get_language_name(self.language)
            language_instruction = f"IMPORTANT: You MUST respond entirely in {lang_name}. Do not respond in English.\n\n"
        else:
            language_instruction = "Respond in English.\n\n"

        # Start with language instruction and section prompt
        prompt_parts = [
            language_instruction + section_prompt,
            "",
            f"Topic: {knowledge_item.get('name', 'Unknown')}",
        ]

        # Add description if available
        description = knowledge_item.get("description")
        if description:
            prompt_parts.append(f"Description: {description}")

        # Add previous section content if this section depends on it
        if previous_section_content:
            prompt_parts.append("")
            prompt_parts.append("CONTEXT FROM PREVIOUS SECTION:")
            prompt_parts.append("The student has already read this content:")
            prompt_parts.append("---")
            prompt_parts.append(previous_section_content)
            prompt_parts.append("---")
            prompt_parts.append("Reference this when relevant (e.g., step numbers, key concepts).")

        # Add example exercise for worked example sections
        if example_exercise:
            prompt_parts.append("")
            prompt_parts.append("EXAM EXERCISE:")

            # Check if sub-question with parent context
            exercise_context = example_exercise.get("exercise_context")
            if exercise_context:
                prompt_parts.append("Full exercise context:")
                prompt_parts.append(exercise_context)
                prompt_parts.append("")
                prompt_parts.append("Sub-question to solve:")

            prompt_parts.append(example_exercise.get("text", example_exercise.get("content", "")))

            # Add image context if exercise has associated visual content
            image_context = example_exercise.get("image_context")
            if image_context:
                prompt_parts.append("")
                prompt_parts.append("IMAGE DESCRIPTION:")
                prompt_parts.append(image_context)

            # Add solution if available (for reference)
            solution = example_exercise.get("solution")
            if solution:
                prompt_parts.append("")
                prompt_parts.append("Solution:")
                prompt_parts.append(solution)

        # Add user's notes (PRO feature)
        if notes:
            prompt_parts.append("")
            prompt_parts.append("Student's notes on this topic:")
            for note in notes[:3]:
                note_text = note[:1500] if len(note) > 1500 else note
                prompt_parts.append(note_text)
            prompt_parts.append("")
            prompt_parts.append("Incorporate relevant parts if they help.")

        return "\n".join(prompt_parts)

    def get_sections_for_approach(self, learning_approach: str) -> List[str]:
        """Get list of section names for a learning approach.

        Args:
            learning_approach: The learning approach (procedural, conceptual, factual, analytical)

        Returns:
            List of section names in order
        """
        approach = learning_approach.lower()
        if approach not in SECTION_PROMPTS:
            approach = "conceptual"
        return list(SECTION_PROMPTS[approach].keys())

    def get_section_context_dependency(
        self, learning_approach: str, section_name: str
    ) -> Optional[str]:
        """Check if a section needs content from a previous section.

        Args:
            learning_approach: The learning approach
            section_name: The section to check

        Returns:
            Name of section to get context from, or None
        """
        approach = learning_approach.lower()
        dependencies = SECTION_CONTEXT_DEPENDENCIES.get(approach, {})
        return dependencies.get(section_name)

    def _select_example_exercise(self, exercises: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select best exercise for worked example.

        Prioritizes: exam > exercise_sheet > homework
        """
        if not exercises:
            return None

        priority = {"exam": 1, "exercise_sheet": 2, "homework": 3}
        sorted_ex = sorted(exercises, key=lambda e: priority.get(e.get("source_type", ""), 99))

        # Get top tier (all with same best source_type)
        best_type = sorted_ex[0].get("source_type")
        top_tier = [e for e in sorted_ex if e.get("source_type") == best_type]

        # Random pick within top tier for variety
        return random.choice(top_tier)
