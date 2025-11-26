"""
Study Strategies & Metacognitive Module for Examina (LLM-Powered).
Teaches students HOW to learn effectively, not just WHAT to learn.

Option C: Full LLM-based strategy generation with caching.
No hardcoded strategies - fully scalable to any subject.
"""

from typing import Dict, List, Optional, Any
import json
import hashlib
from pathlib import Path


class StudyStrategyManager:
    """
    Manages study strategies and metacognitive guidance for different core loops.

    Uses LLM to automatically generate high-quality study strategies for ANY
    core loop, with caching for performance. No hardcoded content.
    """

    def __init__(self, language: str = "en"):
        """
        Initialize the study strategy manager.

        Args:
            language: Output language (any ISO 639-1 code, e.g., "en", "de", "zh")
        """
        self.language = language
        self.llm_manager = None

        # Initialize LLM manager
        try:
            from models.llm_manager import LLMManager
            from config import Config
            self.llm_manager = LLMManager(provider=Config.LLM_PROVIDER)
        except Exception as e:
            print(f"[ERROR] Could not initialize LLM manager: {e}")
            raise RuntimeError("LLM manager required for strategy generation")

    def _lang_instruction(self) -> str:
        """Generate language instruction phrase for any language."""
        return f"in {self.language.upper()} language"

    def get_strategy_for_core_loop(self, core_loop_name: str, difficulty: str = "medium") -> Optional[Dict]:
        """
        Get study strategy for a specific core loop and difficulty.

        Automatically generates using LLM if not cached.

        Args:
            core_loop_name: Name of the core loop
            difficulty: Difficulty level (easy, medium, hard)

        Returns:
            Complete strategy dictionary or None if generation fails
        """
        return self._generate_strategy_with_llm(core_loop_name, difficulty)

    def get_problem_solving_framework(self, core_loop_name: str) -> Optional[Dict]:
        """Get step-by-step problem-solving approach."""
        strategy = self.get_strategy_for_core_loop(core_loop_name)
        return strategy.get("framework") if strategy else None

    def get_learning_tips(self, topic_name: str, difficulty: str = "medium") -> List[str]:
        """Get learning tips for a topic."""
        strategy = self.get_strategy_for_core_loop(topic_name, difficulty)
        return strategy.get("learning_tips", []) if strategy else []

    def get_self_assessment_prompts(self, core_loop_name: str) -> List[str]:
        """Get questions for self-assessment."""
        strategy = self.get_strategy_for_core_loop(core_loop_name)
        return strategy.get("self_assessment", []) if strategy else []

    def get_retrieval_practice(self, core_loop_name: str) -> Optional[Dict]:
        """Get active recall techniques."""
        strategy = self.get_strategy_for_core_loop(core_loop_name)
        return strategy.get("retrieval_practice") if strategy else None

    def get_common_mistakes(self, core_loop_name: str) -> List[str]:
        """Get common mistakes for a core loop."""
        strategy = self.get_strategy_for_core_loop(core_loop_name)
        return strategy.get("common_mistakes", []) if strategy else []

    # ========================================================================
    # LLM Generation & Caching Methods
    # ========================================================================

    def _generate_strategy_with_llm(self, core_loop_name: str, difficulty: str) -> Optional[Dict]:
        """
        Generate study strategy using LLM with caching.

        Args:
            core_loop_name: Name of the core loop
            difficulty: Difficulty level

        Returns:
            Generated strategy dictionary or None if generation fails
        """
        # Check cache first
        cached = self._load_cached_strategy(core_loop_name, difficulty)
        if cached:
            return cached

        # Generate with LLM
        try:
            print(f"[INFO] Generating study strategy for '{core_loop_name}' ({difficulty})...")

            prompt = self._build_strategy_generation_prompt(core_loop_name, difficulty)
            response = self.llm_manager.generate(prompt, temperature=0.7)

            # Check if LLM call succeeded
            if not response.success:
                print(f"[ERROR] LLM generation failed: {response.error}")
                return None

            # Parse response into strategy structure
            strategy = self._parse_llm_strategy(response.text, core_loop_name)

            if strategy:
                # Cache for future use
                self._cache_strategy(core_loop_name, difficulty, strategy)
                print(f"[INFO] Strategy generated and cached")
                return strategy
            else:
                print(f"[WARNING] Failed to parse LLM strategy")
                return None

        except Exception as e:
            print(f"[ERROR] Strategy generation failed: {e}")
            return None

    def _load_cached_strategy(self, core_loop_name: str, difficulty: str) -> Optional[Dict]:
        """Load cached strategy from file."""
        try:
            from config import Config

            if not Config.STUDY_STRATEGY_CACHE_ENABLED:
                return None

            cache_key = self._get_cache_key(core_loop_name, difficulty)
            cache_file = Config.STUDY_STRATEGY_CACHE_DIR / f"{cache_key}.json"

            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)

        except Exception as e:
            print(f"[WARNING] Failed to load cached strategy: {e}")

        return None

    def _cache_strategy(self, core_loop_name: str, difficulty: str, strategy: Dict):
        """Cache generated strategy to file."""
        try:
            from config import Config

            if not Config.STUDY_STRATEGY_CACHE_ENABLED:
                return

            Config.ensure_dirs()  # Ensure cache directory exists

            cache_key = self._get_cache_key(core_loop_name, difficulty)
            cache_file = Config.STUDY_STRATEGY_CACHE_DIR / f"{cache_key}.json"

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(strategy, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"[WARNING] Failed to cache strategy: {e}")

    def _get_cache_key(self, core_loop_name: str, difficulty: str) -> str:
        """Generate cache key for a strategy."""
        # Use hash to handle long/complex names
        content = f"{core_loop_name}_{difficulty}_{self.language}"
        return hashlib.md5(content.encode()).hexdigest()

    def _build_strategy_generation_prompt(self, core_loop_name: str, difficulty: str) -> str:
        """Build prompt for LLM to generate study strategy."""
        return f"""Generate a comprehensive study strategy for learning "{core_loop_name}" at {difficulty} difficulty level.

Output a JSON structure with these exact fields:

{{
  "framework": {{
    "approach": "Overall problem-solving strategy description",
    "steps": [
      {{
        "step": 1,
        "action": "What to do",
        "why": "Why this step is necessary",
        "how": "Concrete instructions on how to execute",
        "reasoning": "Underlying principle or theory",
        "validation": "How to verify this step is correct",
        "common_mistakes": ["Mistake 1", "Mistake 2"]
      }}
      // Include 5-7 steps
    ]
  }},
  "learning_tips": [
    "Actionable tip 1",
    "Actionable tip 2",
    // Include 5-7 tips
  ],
  "self_assessment": [
    "Metacognitive question 1?",
    "Metacognitive question 2?",
    // Include 5 questions
  ],
  "retrieval_practice": {{
    "technique": "Name of active recall technique",
    "exercises": [
      "Practice exercise 1",
      "Practice exercise 2"
      // Include 3-5 exercises
    ]
  }},
  "common_mistakes": [
    "Common error 1 with explanation",
    "Common error 2 with explanation"
    // Include 5-7 mistakes
  ],
  "time_estimate": "Realistic time estimate per exercise (e.g., '15-20 minutes')"
}}

Requirements:
- All text {self._lang_instruction()}
- Focus on HOW to learn and think about the problem, not just WHAT
- Include metacognitive strategies (self-questioning, monitoring understanding)
- Make tips actionable and specific to {core_loop_name}
- Explain the WHY behind each step, not just the mechanics

Return ONLY valid JSON, no markdown code blocks or extra text."""

    def _parse_llm_strategy(self, llm_response: str, core_loop_name: str) -> Optional[Dict]:
        """
        Parse LLM JSON response into strategy dictionary.

        Args:
            llm_response: Raw LLM response text
            core_loop_name: Name of core loop (for error messages)

        Returns:
            Parsed strategy dictionary or None if parsing fails
        """
        try:
            # Extract JSON from response (might have extra text)
            response = llm_response.strip()

            # Remove markdown code blocks if present
            if "```" in response:
                # Find content between ``` markers
                lines = response.split("\n")
                start_idx = 0
                end_idx = len(lines)

                for i, line in enumerate(lines):
                    if line.strip().startswith("```"):
                        if start_idx == 0:
                            start_idx = i + 1
                        else:
                            end_idx = i
                            break

                response = "\n".join(lines[start_idx:end_idx])

            # Parse JSON
            strategy = json.loads(response)

            # Validate required fields
            required_fields = ["framework", "learning_tips", "self_assessment",
                             "retrieval_practice", "common_mistakes"]

            for field in required_fields:
                if field not in strategy:
                    print(f"[WARNING] Missing required field '{field}' in LLM strategy")
                    return None

            return strategy

        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse LLM JSON for '{core_loop_name}': {e}")
            print(f"[DEBUG] Response preview: {llm_response[:300]}...")
            return None
        except Exception as e:
            print(f"[ERROR] Unexpected error parsing strategy: {e}")
            return None

    def format_strategy_output(self, strategy: Dict, core_loop_name: str) -> str:
        """
        Format strategy dictionary as markdown for display.

        Args:
            strategy: Strategy dictionary
            core_loop_name: Name of core loop

        Returns:
            Formatted markdown string
        """
        lines = []

        # Header
        lines.append(f"# Study Strategy: {core_loop_name}\n")

        # Framework
        if strategy.get("framework"):
            fw = strategy["framework"]
            lines.append("## Problem-Solving Framework\n")
            lines.append(f"**Approach:** {fw.get('approach', 'N/A')}\n")

            if fw.get("steps"):
                lines.append("### Steps:\n")
                for step in fw["steps"]:
                    lines.append(f"#### Step {step.get('step', '?')}: {step.get('action', 'N/A')}")
                    lines.append(f"- **Why:** {step.get('why', 'N/A')}")
                    lines.append(f"- **How:** {step.get('how', 'N/A')}")
                    if step.get("reasoning"):
                        lines.append(f"- **Reasoning:** {step['reasoning']}")
                    if step.get("validation"):
                        lines.append(f"- **Validation:** {step['validation']}")
                    if step.get("common_mistakes"):
                        lines.append("- **Common Mistakes:**")
                        for mistake in step["common_mistakes"]:
                            lines.append(f"  - {mistake}")
                    lines.append("")

        # Learning Tips
        if strategy.get("learning_tips"):
            lines.append("## Learning Tips\n")
            for tip in strategy["learning_tips"]:
                lines.append(f"- {tip}")
            lines.append("")

        # Self-Assessment
        if strategy.get("self_assessment"):
            lines.append("## Self-Assessment Questions\n")
            for question in strategy["self_assessment"]:
                lines.append(f"- {question}")
            lines.append("")

        # Common Mistakes
        if strategy.get("common_mistakes"):
            lines.append("## Common Mistakes\n")
            for mistake in strategy["common_mistakes"]:
                lines.append(f"- {mistake}")
            lines.append("")

        # Retrieval Practice
        if strategy.get("retrieval_practice"):
            rp = strategy["retrieval_practice"]
            lines.append("## Retrieval Practice\n")
            lines.append(f"**Technique:** {rp.get('technique', 'N/A')}\n")
            if rp.get("exercises"):
                lines.append("**Exercises:**")
                for ex in rp["exercises"]:
                    lines.append(f"- {ex}")
            lines.append("")

        # Time Estimate
        if strategy.get("time_estimate"):
            lines.append(f"**Estimated Time:** {strategy['time_estimate']}\n")

        return "\n".join(lines)
