"""
LLM-powered exercise splitting for Examina.
Uses LLM to accurately identify exercise boundaries in PDFs.
"""

import json
import logging
import hashlib
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from core.pdf_processor import PDFContent, PDFPage
from core.exercise_splitter import Exercise, ExerciseSplitter
from models.llm_manager import LLMManager

logger = logging.getLogger(__name__)


# System prompt for exercise detection
SYSTEM_PROMPT = """You are an expert at analyzing academic documents in ANY language. Your task is to identify individual exercises/problems in exam papers, homework sheets, and exercise collections.

IMPORTANT RULES:
1. An exercise is a COMPLETE problem that a student needs to solve
2. Sub-questions (a, b, c or i, ii, iii or 1.1, 1.2) belong to their PARENT exercise - do NOT split them
3. Instructions, headers, and administrative text are NOT exercises
4. Exercise markers vary by language:
   - Italian: Esercizio, Problema, Domanda, Quesito
   - English: Exercise, Problem, Question, Task
   - German: Aufgabe, Übung, Frage
   - French: Exercice, Problème, Question
   - Spanish: Ejercicio, Problema, Pregunta
   - Or just numbers: 1., 2., 3. or I, II, III or (a), (b), (c)

You must return ONLY valid JSON, no explanations."""


USER_PROMPT_TEMPLATE = """Analyze this academic document and identify each distinct exercise/problem.

For each exercise, provide:
- exercise_number: The identifier (e.g., "1", "2", "I", "A", "Esercizio 1")
- start_text: The first 80 characters of the exercise (exact match from document)
- end_text: The last 80 characters of the exercise (exact match from document)

DOCUMENT:
---
{text}
---

Return a JSON object with this exact structure:
{{
  "exercises": [
    {{"exercise_number": "1", "start_text": "...", "end_text": "..."}},
    {{"exercise_number": "2", "start_text": "...", "end_text": "..."}}
  ],
  "total_count": 2,
  "notes": "any observations about the document structure"
}}

If no exercises are found, return: {{"exercises": [], "total_count": 0, "notes": "reason"}}"""


@dataclass
class ExerciseBoundary:
    """Detected exercise boundary from LLM."""
    exercise_number: str
    start_text: str
    end_text: str
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None


class LLMExerciseSplitter:
    """LLM-powered exercise boundary detection."""

    def __init__(self, llm_manager: Optional[LLMManager] = None, provider: str = "deepseek"):
        """Initialize LLM exercise splitter.

        Args:
            llm_manager: Existing LLMManager instance (optional)
            provider: LLM provider to use if creating new manager
        """
        self.llm = llm_manager or LLMManager(provider=provider)
        self.regex_fallback = ExerciseSplitter()
        self.exercise_counter = 0

    def split_pdf_content(self, pdf_content: PDFContent, course_code: str) -> List[Exercise]:
        """Split PDF content into individual exercises using LLM.

        Args:
            pdf_content: Extracted PDF content
            course_code: Course code for ID generation

        Returns:
            List of extracted exercises
        """
        self.exercise_counter = 0

        # Combine all pages into single text with page markers
        full_text, page_map = self._prepare_text_with_markers(pdf_content)

        if not full_text.strip():
            logger.warning("Empty PDF content, no exercises to extract")
            return []

        try:
            # Use LLM to detect exercise boundaries
            boundaries = self._detect_boundaries_with_llm(full_text)

            if not boundaries:
                logger.info("LLM found no exercises, falling back to regex")
                return self.regex_fallback.split_pdf_content(pdf_content, course_code)

            # Extract exercises based on boundaries
            exercises = self._extract_exercises(
                full_text, boundaries, page_map, pdf_content, course_code
            )

            logger.info(f"LLM extracted {len(exercises)} exercises from PDF")
            return exercises

        except Exception as e:
            logger.warning(f"LLM exercise splitting failed: {e}, falling back to regex")
            return self.regex_fallback.split_pdf_content(pdf_content, course_code)

    def _prepare_text_with_markers(self, pdf_content: PDFContent) -> tuple[str, Dict[int, int]]:
        """Combine all pages into single text with position tracking.

        Args:
            pdf_content: PDF content with pages

        Returns:
            Tuple of (combined text, page_map mapping char positions to page numbers)
        """
        text_parts = []
        page_map = {}  # char_position -> page_number
        current_pos = 0

        for page in pdf_content.pages:
            page_text = page.text.strip()
            if page_text:
                # Track where this page starts
                page_map[current_pos] = page.page_number

                # Add page marker for context (LLM can see page breaks)
                if text_parts:
                    text_parts.append(f"\n\n--- Page {page.page_number} ---\n\n")
                    current_pos += len(f"\n\n--- Page {page.page_number} ---\n\n")

                text_parts.append(page_text)
                current_pos += len(page_text)

        return "".join(text_parts), page_map

    def _detect_boundaries_with_llm(self, text: str) -> List[ExerciseBoundary]:
        """Use LLM to detect exercise boundaries in text.

        Args:
            text: Full document text

        Returns:
            List of detected exercise boundaries
        """
        # Truncate very long texts to avoid token limits
        max_chars = 50000  # ~12,500 tokens
        if len(text) > max_chars:
            logger.warning(f"Text too long ({len(text)} chars), truncating to {max_chars}")
            text = text[:max_chars]

        # Build prompt
        prompt = USER_PROMPT_TEMPLATE.format(text=text)

        # Call LLM
        logger.info("Calling LLM for exercise boundary detection...")
        response = self.llm.generate(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            temperature=0.1,  # Low temperature for consistent output
            json_mode=True,
            max_tokens=4096
        )

        if not response.success:
            logger.error(f"LLM call failed: {response.error}")
            raise RuntimeError(f"LLM call failed: {response.error}")

        # Parse JSON response
        result = self._parse_llm_response(response.text, text)

        return result

    def _parse_llm_response(self, response_text: str, original_text: str) -> List[ExerciseBoundary]:
        """Parse LLM JSON response into exercise boundaries.

        Args:
            response_text: Raw LLM response text
            original_text: Original document text for position finding

        Returns:
            List of ExerciseBoundary objects with positions
        """
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            logger.warning("Failed to parse JSON directly, trying extraction")
            try:
                # Look for JSON object in response
                import re
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found in response")
            except Exception as e:
                logger.error(f"Failed to parse LLM response: {e}")
                raise

        exercises_data = data.get("exercises", [])
        if not exercises_data:
            logger.info(f"LLM found no exercises. Notes: {data.get('notes', 'none')}")
            return []

        logger.info(f"LLM detected {data.get('total_count', len(exercises_data))} exercises")

        boundaries = []
        for ex in exercises_data:
            boundary = ExerciseBoundary(
                exercise_number=str(ex.get("exercise_number", "")),
                start_text=ex.get("start_text", ""),
                end_text=ex.get("end_text", "")
            )

            # Find actual positions in original text
            if boundary.start_text:
                # Clean up the start text for matching
                start_clean = boundary.start_text.strip()
                # Try exact match first
                start_pos = original_text.find(start_clean)
                if start_pos == -1:
                    # Try partial match (first 40 chars)
                    partial = start_clean[:40] if len(start_clean) > 40 else start_clean
                    start_pos = original_text.find(partial)

                boundary.start_pos = start_pos if start_pos >= 0 else None

            if boundary.end_text:
                end_clean = boundary.end_text.strip()
                # Search from start_pos to narrow down
                search_start = boundary.start_pos if boundary.start_pos else 0
                end_pos = original_text.find(end_clean, search_start)
                if end_pos == -1:
                    # Try partial match
                    partial = end_clean[-40:] if len(end_clean) > 40 else end_clean
                    end_pos = original_text.find(partial, search_start)

                if end_pos >= 0:
                    boundary.end_pos = end_pos + len(end_clean)
                else:
                    boundary.end_pos = None

            # Only include if we found at least start position
            if boundary.start_pos is not None:
                boundaries.append(boundary)
            else:
                logger.warning(f"Could not find position for exercise {boundary.exercise_number}")

        # Sort by start position
        boundaries.sort(key=lambda b: b.start_pos or 0)

        return boundaries

    def _extract_exercises(
        self,
        full_text: str,
        boundaries: List[ExerciseBoundary],
        page_map: Dict[int, int],
        pdf_content: PDFContent,
        course_code: str
    ) -> List[Exercise]:
        """Extract exercise objects from detected boundaries.

        Args:
            full_text: Combined document text
            boundaries: Detected exercise boundaries
            page_map: Mapping of positions to page numbers
            pdf_content: Original PDF content
            course_code: Course code

        Returns:
            List of Exercise objects
        """
        exercises = []

        for i, boundary in enumerate(boundaries):
            # Determine text range
            start_pos = boundary.start_pos or 0

            # End is either the specified end or start of next exercise
            if boundary.end_pos:
                end_pos = boundary.end_pos
            elif i + 1 < len(boundaries) and boundaries[i + 1].start_pos:
                end_pos = boundaries[i + 1].start_pos
            else:
                end_pos = len(full_text)

            # Extract text
            exercise_text = full_text[start_pos:end_pos].strip()

            # Clean up page markers from text
            exercise_text = self._clean_text(exercise_text)

            if not exercise_text or len(exercise_text) < 20:
                logger.warning(f"Skipping empty/short exercise {boundary.exercise_number}")
                continue

            # Determine page number
            page_number = self._get_page_number(start_pos, page_map)

            # Get images from that page
            page_images = []
            for page in pdf_content.pages:
                if page.page_number == page_number and page.images:
                    page_images = page.images
                    break

            # Create exercise
            exercise = self._create_exercise(
                text=exercise_text,
                page_number=page_number,
                exercise_number=boundary.exercise_number,
                images=page_images,
                has_latex=any(p.has_latex for p in pdf_content.pages if p.page_number == page_number),
                source_pdf=pdf_content.file_path.name,
                course_code=course_code
            )

            exercises.append(exercise)

        return exercises

    def _get_page_number(self, char_pos: int, page_map: Dict[int, int]) -> int:
        """Determine page number for a character position.

        Args:
            char_pos: Character position in combined text
            page_map: Mapping of start positions to page numbers

        Returns:
            Page number (1-indexed)
        """
        page_num = 1
        for pos, pnum in sorted(page_map.items()):
            if pos <= char_pos:
                page_num = pnum
            else:
                break
        return page_num

    def _clean_text(self, text: str) -> str:
        """Clean exercise text.

        Args:
            text: Raw exercise text

        Returns:
            Cleaned text
        """
        import re

        # Remove page markers we added
        text = re.sub(r'\n*--- Page \d+ ---\n*', '\n', text)

        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        # Remove page numbers
        text = re.sub(r'(?:^|\n)Pagina\s+\d+(?:\n|$)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'(?:^|\n)Page\s+\d+(?:\n|$)', '', text, flags=re.IGNORECASE)

        return text.strip()

    def _create_exercise(
        self,
        text: str,
        page_number: int,
        exercise_number: Optional[str],
        images: List[bytes],
        has_latex: bool,
        source_pdf: str,
        course_code: str
    ) -> Exercise:
        """Create an Exercise object.

        Args:
            text: Exercise text
            page_number: Page number
            exercise_number: Exercise number/label
            images: Image data
            has_latex: Whether LaTeX was detected
            source_pdf: Source PDF filename
            course_code: Course code

        Returns:
            Exercise object
        """
        self.exercise_counter += 1

        # Generate unique ID
        components = f"{course_code}_{source_pdf}_{page_number}_{exercise_number or 'none'}_{self.exercise_counter}"
        hash_obj = hashlib.md5(components.encode())
        short_hash = hash_obj.hexdigest()[:12]

        course_abbrev = course_code.lower().replace('b', '').replace('0', '')[:6]
        exercise_id = f"{course_abbrev}_{self.exercise_counter:04d}_{short_hash}"

        return Exercise(
            id=exercise_id,
            text=text,
            page_number=page_number,
            exercise_number=exercise_number,
            has_images=len(images) > 0,
            image_data=images,
            has_latex=has_latex,
            latex_content=None,
            source_pdf=source_pdf
        )


# Convenience function for one-off splitting
def split_pdf_with_llm(pdf_content: PDFContent, course_code: str, provider: str = "deepseek") -> List[Exercise]:
    """Split PDF content using LLM.

    Args:
        pdf_content: PDF content to split
        course_code: Course code
        provider: LLM provider

    Returns:
        List of exercises
    """
    splitter = LLMExerciseSplitter(provider=provider)
    return splitter.split_pdf_content(pdf_content, course_code)
