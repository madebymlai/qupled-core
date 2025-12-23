"""
Generic Solution Separator for PDFs with Questions + Answers

This module uses LLM-based detection to separate questions from solutions
in PDF files where both appear together. Works for ANY:
- Language (Italian, English, etc.)
- Subject (CS, Math, Physics, etc.)
- Format (inline, appendix, interleaved)
- Structure (numbered, paragraphs, sections)

NO HARDCODING - fully adaptive to content.

TODO: Work in progress - not yet integrated into the main pipeline.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from models.llm_manager import LLMManager


@dataclass
class QuestionAnswerPair:
    """Separated question and answer."""

    question: str
    answer: Optional[str]
    confidence: float  # 0.0-1.0, how confident we are about the separation
    separation_method: str  # "llm", "pattern", "none"


class SolutionSeparator:
    """
    Separates questions from answers in exercise text using LLM-based detection.

    This is a GENERIC solution that works for:
    - Any language (no hardcoded Italian/English patterns)
    - Any subject (CS, Math, Physics, Chemistry, etc.)
    - Any format (inline answers, appendix, Q then A, interleaved)
    - Any structure (numbered, bullet points, paragraphs)
    """

    def __init__(self, llm_manager: Optional[LLMManager] = None):
        """Initialize solution separator.

        Args:
            llm_manager: LLM manager for text analysis (defaults to Anthropic)
        """
        self.llm = llm_manager or LLMManager(provider="anthropic")

    def has_solution(self, text: str) -> bool:
        """
        Detect if text contains both question AND solution (not just question).

        Uses LLM to analyze text structure and content.

        Args:
            text: Exercise text to analyze

        Returns:
            True if text appears to contain both Q and A
        """
        # Quick length heuristic: very short text unlikely to have both
        if len(text) < 100:
            return False

        prompt = f"""Analyze this exercise text and determine if it contains BOTH a question AND its answer/solution.

TEXT:
{text[:2000]}
{"...(truncated)" if len(text) > 2000 else ""}

Answer with ONLY "yes" or "no".

Guidelines:
- "yes" if text has: question + detailed explanation/solution/answer
- "no" if text has: only question, only theory, only code, only solution
- Look for: explanatory text, step-by-step reasoning, definitions, examples

Answer:"""

        response = self.llm.generate(
            prompt=prompt,
            system="You are an expert at analyzing educational content structure. Be accurate and concise.",
            temperature=0.0,
            max_tokens=10,
        )

        if response.success:
            answer = response.text.strip().lower()
            return answer.startswith("yes")

        # Conservative fallback: assume no solution if LLM fails
        return False

    def separate(self, text: str) -> QuestionAnswerPair:
        """
        Separate question from answer in exercise text.

        Uses LLM to identify the boundary between question and answer.
        This is GENERIC and works for any format/language/subject.

        Args:
            text: Full exercise text (question + answer)

        Returns:
            QuestionAnswerPair with separated content
        """
        # First check if there's actually a solution to separate
        if not self.has_solution(text):
            return QuestionAnswerPair(
                question=text, answer=None, confidence=0.9, separation_method="none"
            )

        # Use LLM to find the separation point
        prompt = f"""Separate this exercise text into QUESTION and ANSWER parts.

TEXT:
{text}

Task: Identify where the question ends and the answer/solution begins.

Respond in JSON format:
{{
  "question": "the question text only",
  "answer": "the answer/solution text only",
  "confidence": 0.95
}}

Guidelines:
- Question: The problem statement, what's being asked
- Answer: Explanations, definitions, step-by-step solution, reasoning
- Confidence: 0.0-1.0, how clear the boundary is
- If uncertain, err on the side of including more in the question
- Preserve all original text (don't summarize)
"""

        response = self.llm.generate(
            prompt=prompt,
            system="You are an expert at analyzing educational content. Be precise and preserve all original text.",
            temperature=0.0,
            max_tokens=4000,
            json_mode=True,
        )

        if not response.success:
            # Fallback: return as-is
            return QuestionAnswerPair(
                question=text, answer=None, confidence=0.0, separation_method="none"
            )

        # Parse JSON response
        try:
            import json

            # Strip markdown code fences if present
            response_text = response.text.strip()
            if response_text.startswith("```"):
                # Remove opening fence (```json or ```)
                lines = response_text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                # Remove closing fence (```)
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                response_text = "\n".join(lines)

            result = json.loads(response_text)

            question = result.get("question", "").strip()
            answer = result.get("answer", "").strip()
            confidence = float(result.get("confidence", 0.5))

            # Validation: question + answer should cover most of original text
            combined_length = len(question) + len(answer)
            original_length = len(text)
            coverage_ratio = combined_length / original_length if original_length > 0 else 0

            # If coverage is poor, return original
            if coverage_ratio < 0.7:
                return QuestionAnswerPair(
                    question=text, answer=None, confidence=0.0, separation_method="none"
                )

            return QuestionAnswerPair(
                question=question, answer=answer, confidence=confidence, separation_method="llm"
            )

        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback: return as-is
            return QuestionAnswerPair(
                question=text, answer=None, confidence=0.0, separation_method="none"
            )

    def batch_separate(self, exercises: List[Dict]) -> List[QuestionAnswerPair]:
        """
        Separate questions from answers for multiple exercises.

        Args:
            exercises: List of exercise dicts with 'text' field

        Returns:
            List of QuestionAnswerPair objects
        """
        results = []
        for ex in exercises:
            text = ex.get("text", "")
            result = self.separate(text)
            results.append(result)

        return results
