"""
Exercise splitting for Examina.
Splits PDF content into individual exercises based on patterns.
"""

import re
import json
import hashlib
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

from core.pdf_processor import PDFContent, PDFPage

if TYPE_CHECKING:
    from models.llm_manager import LLMManager

logger = logging.getLogger(__name__)


def _strip_inline_flags(pattern: str) -> Tuple[str, int]:
    """Strip inline regex flags and return (clean_pattern, flags).

    LLM may return patterns with inline flags like (?i), (?m), (?s).
    We strip these and return appropriate re flags instead.
    """
    flags = 0
    # Match inline flags at start or anywhere in pattern
    inline_flag_pattern = r'\(\?([imslux]+)\)'

    def replace_flags(match):
        nonlocal flags
        for char in match.group(1):
            if char == 'i':
                flags |= re.IGNORECASE
            elif char == 'm':
                flags |= re.MULTILINE
            elif char == 's':
                flags |= re.DOTALL
        return ''

    clean_pattern = re.sub(inline_flag_pattern, replace_flags, pattern)
    return clean_pattern, flags


def _is_roman_numeral(s: str) -> bool:
    """Check if string is a valid roman numeral (i, ii, iii, iv, v, etc.)."""
    if not s:
        return False
    return bool(re.match(r'^[ivxlcdm]+$', s.lower()))


def _roman_to_int(s: str) -> int:
    """Convert roman numeral string to integer.

    Examples: i=1, ii=2, iii=3, iv=4, v=5, vi=6, ix=9, x=10
    """
    roman_map = {'i': 1, 'v': 5, 'x': 10, 'l': 50, 'c': 100, 'd': 500, 'm': 1000}
    s = s.lower()
    result = 0
    prev = 0
    for c in reversed(s):
        curr = roman_map.get(c, 0)
        if curr < prev:
            result -= curr
        else:
            result += curr
        prev = curr
    return result


@dataclass
class Exercise:
    """Represents a single exercise extracted from a PDF."""
    id: str
    text: str
    page_number: int
    exercise_number: Optional[str]
    has_images: bool
    image_data: List[bytes]
    has_latex: bool
    latex_content: Optional[str]
    source_pdf: str
    # Solution fields (populated via LLM-provided solution_pattern)
    solution: Optional[str] = None
    solution_page: Optional[int] = None
    # Sub-question support (added for unified knowledge model)
    parent_exercise_number: Optional[str] = None  # "2" if this is "2a"
    sub_question_marker: Optional[str] = None     # "a", "b", "c", "i", "ii", etc.
    is_sub_question: bool = False
    # Exercise context (LLM-generated: parent context for subs, exercise summary for standalone)
    exercise_context: Optional[str] = None

    def get_preview_text(self, max_length: int = 100) -> str:
        """Get a clean preview of the exercise text for display.

        Uses structural patterns (language-agnostic) to remove exercise markers,
        form fields, and other non-content text.

        Args:
            max_length: Maximum length of the preview text

        Returns:
            Clean preview text suitable for display
        """
        # Split into lines and find first meaningful content line
        lines = self.text.strip().split('\n')

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip lines with form fields (underscores, dots as blanks)
            if '_____' in line or '.....' in line or '___' in line:
                continue

            # Skip very short lines (likely headers or labels)
            if len(line) < 20:
                continue

            # Skip lines that are mostly uppercase and short (likely headers)
            if len(line) < 60 and line.upper() == line:
                continue

            # Skip lines that start with "word + number" pattern (exercise markers)
            # This catches "Esercizio 1", "Exercise 2", "Aufgabe 3", etc.
            if re.match(r'^[A-Za-z\u00C0-\u024F]+\s+\d+\s*$', line):
                continue

            # Found a good line - clean it up
            # Remove leading "word + number" if followed by more content
            cleaned = re.sub(r'^[A-Za-z\u00C0-\u024F]+\s+\d+\s*', '', line).strip()
            if not cleaned or len(cleaned) < 15:
                cleaned = line  # Use original if cleaning removed too much

            # Remove leading number patterns like "1.", "1)"
            cleaned = re.sub(r'^\d+[\.\)\:]\s*', '', cleaned).strip()

            # Get first sentence or truncate
            if '.' in cleaned[:120] and cleaned.index('.') > 20:
                preview = cleaned[:cleaned.index('.') + 1]
            else:
                preview = cleaned[:max_length]

            # Truncate if needed
            if len(preview) > max_length:
                preview = preview[:max_length].rsplit(' ', 1)[0] + "..."
            elif len(cleaned) > len(preview):
                preview = preview.rstrip('.') + "..."

            return preview

        # Fallback - no good lines found
        # Just return first N chars of raw text, cleaned of obvious junk
        text = self.text.strip()
        text = re.sub(r'[_]{3,}', '', text)  # Remove underscores
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        preview = text[:max_length].strip()
        if len(text) > max_length:
            preview = preview.rsplit(' ', 1)[0] + "..."
        return preview if preview else f"#{self.exercise_number or '?'}"


class MarkerType(Enum):
    """Type of exercise marker."""
    PARENT = "parent"  # Main exercise marker (e.g., "Exercise 1")
    SUB = "sub"        # Sub-question marker (e.g., "1.", "a)")


@dataclass
class Marker:
    """A detected exercise marker in the document."""
    marker_type: MarkerType
    marker_text: str       # The actual marker text (e.g., "Exercise 1", "a)")
    number: str            # Extracted number/letter ("1", "a", etc.)
    start_position: int    # Character position where marker starts
    question_start: int    # Character position where question text begins


@dataclass
class MarkerPattern:
    """Pattern for exercise markers detected by LLM.

    LLM returns actual regex patterns for language-agnostic detection.

    Note: sub_patterns is no longer returned from pattern detection (Call 1).
    Per-exercise sub_patterns are now determined in Call 3 after parent boundaries
    are known, to avoid false positives from inline conditions like "i) X or ii) Y".
    """
    exercise_pattern: str                   # Regex for exercise markers (e.g., "Esercizio\\s+(\\d+)")
    sub_patterns: Optional[List[str]] = None  # DEPRECATED: Now determined per-exercise in Call 3
    solution_pattern: Optional[str] = None  # Keyword or regex for solutions (e.g., "Soluzione")

    @property
    def sub_pattern(self) -> Optional[str]:
        """Backward compatibility: return first sub_pattern if any."""
        return self.sub_patterns[0] if self.sub_patterns else None


@dataclass
class ExerciseNode:
    """Hierarchical exercise structure for building parent-child relationships."""
    marker: Marker
    context: str              # Setup text (for parents)
    question_text: str        # The actual question
    children: List["ExerciseNode"] = field(default_factory=list)
    parent: Optional["ExerciseNode"] = None


@dataclass
class ExplicitExercise:
    """Explicit exercise markers from LLM detection."""
    number: str
    start_marker: str  # First ~50 chars of exercise
    end_marker: Optional[str] = None  # Last ~50 chars of QUESTION (before junk)


@dataclass
class DetectionResult:
    """Result from LLM exercise detection."""
    pattern: Optional[MarkerPattern] = None  # Pattern-based detection (legacy)
    explicit_markers: Optional[List[str]] = None  # Legacy: simple marker texts
    explicit_exercises: Optional[List[ExplicitExercise]] = None  # Explicit start markers
    has_solutions: bool = False  # Whether document contains solutions


@dataclass
class ExerciseAnalysis:
    """Result from Call 2: per-exercise analysis."""
    end_pos: int  # Character position where exercise ends
    has_sub_questions: bool  # Whether exercise contains sub-questions


def _detect_exercises(
    text_sample: str,
    llm_manager: "LLMManager",
) -> Optional[DetectionResult]:
    """Call 1: Detect parent exercises in document.

    LLM identifies main exercises by returning the unique text that starts each one.

    Args:
        text_sample: First ~30k chars of document
        llm_manager: LLM manager for inference

    Returns:
        DetectionResult with exercise markers and has_solutions flag, None if detection fails
    """
    prompt = """Identify MAIN EXERCISES in this exam document.

Main exercises are TOP-LEVEL sections (like "Exercise 1", "Esercizio 1", "1)", "Problem 1").
NOT main exercises: sub-questions like "1a)", "a)", "i)", "(a)"
IGNORE solution sections - only identify QUESTIONS.

DOCUMENT:
{text}

Return the UNIQUE TEXT that starts each main exercise.
Copy EXACT text verbatim from the document. Include the exercise marker/number.

Output valid JSON:
{{"exercises": ["first exercise start text...", "second exercise start text..."], "has_solutions": true/false}}

If no clear exercise divisions:
{{"exercises": null, "has_solutions": false}}"""

    try:
        llm_response = llm_manager.generate(
            prompt.format(text=text_sample[:30000]),
            temperature=0.0,
            json_mode=True,
        )

        # Check if the response was successful
        if hasattr(llm_response, 'success') and not llm_response.success:
            error_msg = getattr(llm_response, 'error', 'Unknown error')
            logger.warning(f"LLM exercise detection failed: {error_msg}")
            return None

        # Extract text from LLMResponse object
        response = llm_response.text if hasattr(llm_response, 'text') else str(llm_response)

        if not response or not response.strip():
            logger.warning("LLM returned empty response for exercise detection")
            return None

        data = json.loads(response)

        # Parse exercises list (start marker strings)
        exercises = data.get("exercises")
        if not exercises:
            logger.warning("No exercises found in document")
            return None

        # Convert to ExplicitExercise objects
        # Extract number from start of marker text (e.g., "1) DARE..." -> number="1")
        explicit_exercises = []
        for i, marker in enumerate(exercises):
            if not marker:
                continue
            # Try to extract number from marker
            num_match = re.match(r'^[^\d]*(\d+)', marker)
            number = num_match.group(1) if num_match else str(i + 1)
            explicit_exercises.append(ExplicitExercise(
                number=number,
                start_marker=marker,
                end_marker=None,
            ))

        if not explicit_exercises:
            return None

        logger.info(f"Found {len(explicit_exercises)} explicit exercise markers")
        return DetectionResult(
            explicit_exercises=explicit_exercises,
            has_solutions=bool(data.get("has_solutions", False)),
        )

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM exercise detection response: {e}")
        return None
    except Exception as e:
        logger.error(f"LLM exercise detection failed: {e}")
        return None


@dataclass
class ExerciseBoundary:
    """Exercise boundaries found in document."""
    number: str
    start_pos: int
    end_pos: Optional[int]  # None if end_marker not found


# ============================================================================
# CALL 2: EXERCISE ANALYSIS (parallel per exercise)
# ============================================================================


async def _analyze_exercise(
    exercise_num: str,
    exercise_text: str,
    start_pos: int,
    llm_manager: "LLMManager",
) -> Tuple[str, ExerciseAnalysis]:
    """Call 2: Analyze a single exercise (one parallel call).

    Args:
        exercise_num: Exercise number
        exercise_text: Full text of the exercise
        start_pos: Start position in document
        llm_manager: LLM manager

    Returns:
        Tuple of (exercise_num, ExerciseAnalysis)
    """
    prompt = f"""Identify for this exercise:
1. end_marker: LAST 40-60 characters of the ENTIRE exercise question
2. has_sub_questions: true/false

CRITICAL:
- end_marker should be at the END of all question content (INCLUDING any sub-questions), BEFORE any:
  - Form fields or blank lines for answers
  - Solutions or answer sections
  - Page headers/footers
  - Exam instructions
  - Junk text between exercises

- has_sub_questions = true if exercise contains SEPARATE tasks requiring SEPARATE answers
  - Can be marked: a), b), c), 1., 2., 3., i), ii), -, •, etc.
  - Can be unmarked: separate paragraphs asking different things
- has_sub_questions = false if exercise contains only ONE task (with its data/context)

EXERCISE:
\"\"\"
{exercise_text}
\"\"\"

Return JSON:
{{"end_marker": "last 40-60 chars verbatim", "has_sub_questions": true/false}}"""

    try:
        llm_response = await asyncio.to_thread(
            llm_manager.generate, prompt, temperature=0.0
        )
        response_text = llm_response.text if hasattr(llm_response, 'text') else str(llm_response)

        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            data = json.loads(json_match.group())
            end_marker = data.get("end_marker", "")
            has_subs = data.get("has_sub_questions", True)  # Default True = safe fallback

            # Find end position using rfind
            end_pos = start_pos + len(exercise_text)  # Default to rough end
            clean_marker = end_marker.strip().strip("...").strip("…").strip()
            if len(clean_marker) >= 10:
                pos = _fuzzy_rfind(exercise_text, clean_marker)
                if pos >= 0:
                    end_pos = start_pos + pos + len(clean_marker)

            return exercise_num, ExerciseAnalysis(end_pos=end_pos, has_sub_questions=has_subs)

    except Exception as e:
        logger.warning(f"Exercise {exercise_num} analysis failed: {e}")

    # Fallback: assume has subs (safe), use rough end
    return exercise_num, ExerciseAnalysis(
        end_pos=start_pos + len(exercise_text),
        has_sub_questions=True
    )


async def _analyze_boundaries(
    boundaries: List[ExerciseBoundary],
    full_text: str,
    llm_manager: "LLMManager",
) -> Dict[str, ExerciseAnalysis]:
    """Call 2 for Smart Split: Analyze exercise boundaries in parallel.

    Args:
        boundaries: List of exercise boundaries from Call 1
        full_text: Complete document text
        llm_manager: LLM for analysis

    Returns:
        Dict mapping exercise number to ExerciseAnalysis (end_pos, has_sub_questions)
    """
    if not boundaries:
        return {}

    # Pre-compute rough end positions (next exercise or end of text)
    rough_ends: Dict[str, int] = {}
    sorted_boundaries = sorted(boundaries, key=lambda b: b.start_pos)
    for i, boundary in enumerate(sorted_boundaries):
        if i + 1 < len(sorted_boundaries):
            rough_ends[boundary.number] = sorted_boundaries[i + 1].start_pos
        else:
            rough_ends[boundary.number] = len(full_text)

    logger.info(f"Analyzing {len(boundaries)} exercises (parallel)...")

    async def analyze_one(boundary: ExerciseBoundary) -> Tuple[str, ExerciseAnalysis]:
        start = boundary.start_pos
        end = rough_ends.get(boundary.number, len(full_text))
        exercise_text = full_text[start:end].strip()
        return await _analyze_exercise(boundary.number, exercise_text, start, llm_manager)

    tasks = [analyze_one(b) for b in boundaries]
    results = await asyncio.gather(*tasks)
    analysis_dict = dict(results)

    with_subs = sum(1 for a in analysis_dict.values() if a.has_sub_questions)
    logger.info(f"Call 2 complete: {with_subs}/{len(analysis_dict)} exercises have sub-questions")

    return analysis_dict


# =============================================================================
# Call 3: Sub-question Start Markers (parallel per exercise with subs)
# =============================================================================


async def _get_sub_start_markers_for_exercise(
    exercise_num: str,
    exercise_text: str,
    llm_manager: "LLMManager",
) -> Tuple[str, Optional[List[str]]]:
    """Call 3: Get sub-question start markers for one exercise."""
    prompt = f"""Identify sub-questions in this exercise.

Sub-questions are SEPARATE TASKS requiring SEPARATE ANSWERS:
- Can be marked: a), b), c), 1., 2., 3., i), ii), -, •, etc.
- Can be unmarked: separate paragraphs asking different things

Key distinction:
- Sub-question: asks student to DO something (produce answer, calculation, drawing)
- NOT a sub-question: GIVES INFORMATION to student

IMPORTANT: Return the EXACT first 10-15 words as they appear (copy verbatim, including markers).

EXERCISE:
\"\"\"
{exercise_text}
\"\"\"

Return JSON:
{{"sub_questions": ["exact first 10-15 words of sub 1...", "exact first 10-15 words of sub 2..."] or null}}"""

    try:
        def call_llm():
            return llm_manager.generate(prompt, temperature=0.0)

        llm_response = await asyncio.to_thread(call_llm)
        response_text = llm_response.text if hasattr(llm_response, 'text') else str(llm_response)

        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            data = json.loads(json_match.group())
            subs = data.get("sub_questions")
            return exercise_num, subs

    except Exception as e:
        logger.warning(f"Exercise {exercise_num} sub detection failed: {e}")

    return exercise_num, None


async def _get_sub_start_markers_parallel(
    exercises_with_subs: List[Tuple[str, str]],  # List of (exercise_num, exercise_text)
    llm_manager: "LLMManager",
) -> Dict[str, List[str]]:
    """Call 3: Get sub-question start markers in parallel."""
    if not exercises_with_subs:
        return {}

    logger.info(f"Getting sub-question start markers for {len(exercises_with_subs)} exercises in parallel (Call 3)...")

    async def process_one(item: Tuple[str, str]) -> Tuple[str, Optional[List[str]]]:
        ex_num, ex_text = item
        return await _get_sub_start_markers_for_exercise(ex_num, ex_text, llm_manager)

    tasks = [process_one(item) for item in exercises_with_subs]
    results = await asyncio.gather(*tasks)

    # Filter out None results
    results_dict = {num: subs for num, subs in results if subs}

    with_subs = len(results_dict)
    logger.info(f"Call 3 complete: found sub-questions in {with_subs}/{len(exercises_with_subs)} exercises")

    return results_dict


# =============================================================================
# Call 4: Sub-question End Markers (parallel per sub-question)
# =============================================================================


async def _get_sub_end_marker_for_sub(
    sub_id: str,
    sub_text: str,
    llm_manager: "LLMManager",
) -> Tuple[str, Optional[str]]:
    """Call 4: Get end marker for one sub-question."""
    prompt = f"""Identify where this sub-question ENDS.
Return the last 30-50 characters of the actual question (before any trailing junk like page numbers, form fields, next sub-question markers).

SUB-QUESTION:
\"\"\"
{sub_text}
\"\"\"

Return JSON:
{{"end_marker": "last 30-50 chars verbatim"}}

IMPORTANT: end_marker must be EXACT text, used to find where to trim."""

    try:
        def call_llm():
            return llm_manager.generate(prompt, temperature=0.0)

        llm_response = await asyncio.to_thread(call_llm)
        response_text = llm_response.text if hasattr(llm_response, 'text') else str(llm_response)

        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            data = json.loads(json_match.group())
            end_marker = data.get("end_marker")
            return sub_id, end_marker

    except Exception as e:
        logger.warning(f"Sub {sub_id} end marker detection failed: {e}")

    return sub_id, None


async def _get_sub_end_markers_parallel(
    sub_questions: List[Tuple[str, str]],  # List of (sub_id, sub_text)
    llm_manager: "LLMManager",
) -> Dict[str, str]:
    """Call 4: Get sub-question end markers in parallel."""
    if not sub_questions:
        return {}

    logger.info(f"Getting end markers for {len(sub_questions)} sub-questions in parallel (Call 4)...")

    async def process_one(item: Tuple[str, str]) -> Tuple[str, Optional[str]]:
        sub_id, sub_text = item
        return await _get_sub_end_marker_for_sub(sub_id, sub_text, llm_manager)

    tasks = [process_one(item) for item in sub_questions]
    results = await asyncio.gather(*tasks)

    # Filter out None results
    results_dict = {sub_id: marker for sub_id, marker in results if marker}

    logger.info(f"Call 4 complete: found end markers for {len(results_dict)}/{len(sub_questions)} sub-questions")

    return results_dict


# =============================================================================
# Call 5: Context Summaries (parallel per parent with subs)
# =============================================================================


async def _get_context_summary_for_exercise(
    exercise_num: str,
    exercise_text: str,
    has_sub_questions: bool,
    llm_manager: "LLMManager",
) -> Tuple[str, Optional[str]]:
    """Call 5: Get context summary for one exercise (parent or standalone)."""
    if has_sub_questions:
        # Parent exercise: extract shared context for sub-questions
        prompt = f"""Extract the shared context that sub-questions need from this parent exercise.

Good context: data values, parameters, scenario setup, definitions that sub-questions reference.
Return null if sub-questions are independent and don't need shared info.
IMPORTANT: Return context_summary in ENGLISH, even if source is another language.

PARENT EXERCISE:
\"\"\"
{exercise_text}
\"\"\"

Return JSON:
{{"context_summary": "shared context in English" or null}}"""
    else:
        # Standalone exercise: summarize what the exercise asks
        prompt = f"""Summarize this exercise for context.

Focus on:
- The core skill/concept being tested
- Key data values, parameters, or given information
- What the student must do (calculate, explain, design, compare, etc.)

Keep it concise - this summary helps understand what the exercise asks.
IMPORTANT: Return summary in ENGLISH, even if source is another language.

EXERCISE:
\"\"\"
{exercise_text}
\"\"\"

Return JSON:
{{"context_summary": "concise exercise summary in English" or null}}"""

    try:
        def call_llm():
            return llm_manager.generate(prompt, temperature=0.0)

        llm_response = await asyncio.to_thread(call_llm)
        response_text = llm_response.text if hasattr(llm_response, 'text') else str(llm_response)

        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            data = json.loads(json_match.group())
            summary = data.get("context_summary")
            if summary and summary != "null":
                return exercise_num, summary

    except Exception as e:
        logger.warning(f"Exercise {exercise_num} context summary failed: {e}")

    return exercise_num, None


async def _get_context_summaries_parallel(
    exercises_for_context: List[Tuple[str, str, bool]],  # List of (exercise_num, exercise_text, has_sub_questions)
    llm_manager: "LLMManager",
) -> Dict[str, str]:
    """Call 5: Get context summaries in parallel for all exercises."""
    if not exercises_for_context:
        return {}

    parent_count = sum(1 for _, _, has_subs in exercises_for_context if has_subs)
    standalone_count = len(exercises_for_context) - parent_count
    logger.info(f"Getting context summaries for {len(exercises_for_context)} exercises ({parent_count} parents, {standalone_count} standalone) in parallel (Call 5)...")

    async def process_one(item: Tuple[str, str, bool]) -> Tuple[str, Optional[str]]:
        ex_num, exercise_text, has_sub_questions = item
        return await _get_context_summary_for_exercise(ex_num, exercise_text, has_sub_questions, llm_manager)

    tasks = [process_one(item) for item in exercises_for_context]
    results = await asyncio.gather(*tasks)

    # Filter out None results
    summaries_dict = {num: summary for num, summary in results if summary}

    logger.info(f"Call 5 complete: got context summaries for {len(summaries_dict)}/{len(exercises_for_context)} exercises")

    return summaries_dict


def _find_explicit_exercises(
    full_text: str,
    explicit_exercises: List[ExplicitExercise],
) -> List[ExerciseBoundary]:
    """Find exercise boundaries using explicit markers with end positions.

    Args:
        full_text: Complete document text
        explicit_exercises: List of ExplicitExercise with start/end markers

    Returns:
        List of ExerciseBoundary with start and end positions
    """
    boundaries: List[ExerciseBoundary] = []

    for ex in explicit_exercises:
        # Find start position
        start_pos = _fuzzy_find(full_text, ex.start_marker)
        if start_pos < 0:
            logger.warning(f"Start marker not found for exercise {ex.number}: '{ex.start_marker}'")
            continue

        # Find end position (if end_marker provided)
        end_pos = None
        if ex.end_marker:
            # Search for end_marker after start_pos
            end_pos = _fuzzy_find(full_text, ex.end_marker, start_from=start_pos)
            if end_pos >= 0:
                # Include the end_marker text in the exercise
                end_pos += len(ex.end_marker)
            else:
                logger.warning(f"End marker not found for exercise {ex.number}: '{ex.end_marker}'")

        boundaries.append(ExerciseBoundary(
            number=ex.number,
            start_pos=start_pos,
            end_pos=end_pos,
        ))

    # Sort by start position
    boundaries.sort(key=lambda b: b.start_pos)

    return boundaries


def _create_exercises_from_boundaries(
    boundaries: List[ExerciseBoundary],
    full_text: str,
    pdf_content: "PDFContent",
    course_code: str,
    page_lookup: Dict[int, int],
) -> List[Exercise]:
    """Create Exercise objects from explicit boundaries.

    Args:
        boundaries: List of exercise boundaries with start/end positions
        full_text: Complete document text
        pdf_content: PDF content for metadata
        course_code: Course code for ID generation
        page_lookup: Mapping of char positions to page numbers

    Returns:
        List of Exercise objects
    """
    exercises: List[Exercise] = []

    def get_page_number(char_pos: int) -> int:
        """Find page number for a character position."""
        page = 1
        for pos, pg in sorted(page_lookup.items()):
            if pos <= char_pos:
                page = pg
            else:
                break
        return page

    for i, boundary in enumerate(boundaries):
        # Determine end position
        if boundary.end_pos:
            # Use explicit end position
            end_pos = boundary.end_pos
        elif i + 1 < len(boundaries):
            # Fall back to next exercise start
            end_pos = boundaries[i + 1].start_pos
        else:
            # Last exercise - go to end of document
            end_pos = len(full_text)

        # Extract text
        text = full_text[boundary.start_pos:end_pos].strip()

        # Generate exercise ID
        page_num = get_page_number(boundary.start_pos)
        exercise_id = _generate_exercise_id(
            course_code, pdf_content.file_path.name, page_num, i + 1
        )

        exercises.append(Exercise(
            id=exercise_id,
            text=text,
            page_number=page_num,
            exercise_number=boundary.number,
            has_images=False,
            image_data=[],
            has_latex=False,
            latex_content=None,
            source_pdf=pdf_content.file_path.name,
        ))

    return exercises


def _create_exercises_from_explicit_boundaries(
    boundaries: List[ExerciseBoundary],
    explicit_subs: Dict[str, List[str]],
    full_text: str,
    pdf_content: "PDFContent",
    course_code: str,
    page_lookup: Dict[int, int],
) -> List[Exercise]:
    """Create Exercise objects from explicit boundaries with sub-question support.

    Args:
        boundaries: List of exercise boundaries with start/end positions
        explicit_subs: Dict mapping exercise number to list of sub start markers
        full_text: Complete document text
        pdf_content: PDF content for metadata
        course_code: Course code for ID generation
        page_lookup: Mapping of char positions to page numbers

    Returns:
        List of Exercise objects including sub-questions
    """
    exercises: List[Exercise] = []

    def get_page_number(char_pos: int) -> int:
        """Find page number for a character position."""
        page = 1
        for pos, pg in sorted(page_lookup.items()):
            if pos <= char_pos:
                page = pg
            else:
                break
        return page

    for i, boundary in enumerate(boundaries):
        parent_num = boundary.number

        # Determine parent end position
        if boundary.end_pos:
            parent_end = boundary.end_pos
        elif i + 1 < len(boundaries):
            parent_end = boundaries[i + 1].start_pos
        else:
            parent_end = len(full_text)

        parent_text = full_text[boundary.start_pos:parent_end].strip()
        page_num = get_page_number(boundary.start_pos)

        # Check for sub-questions
        subs = explicit_subs.get(parent_num, [])

        if subs:
            # Find all sub positions first
            sub_positions: List[Tuple[int, str]] = []
            for start_marker in subs:
                if not start_marker:
                    continue
                start_pos = _fuzzy_find(parent_text, start_marker)
                if start_pos < 0:
                    logger.warning(f"Could not find sub-question start: {start_marker[:30]}...")
                    continue
                sub_positions.append((start_pos, start_marker))

            # Sort by position
            sub_positions.sort(key=lambda x: x[0])

            # Create exercises with end = next sub start or parent end
            for sub_idx, (start_pos, start_marker) in enumerate(sub_positions):
                sub_num = str(sub_idx + 1)

                # End at next sub or parent end
                if sub_idx + 1 < len(sub_positions):
                    end_pos = sub_positions[sub_idx + 1][0]
                else:
                    end_pos = len(parent_text)

                sub_text = parent_text[start_pos:end_pos].strip()
                if not sub_text:
                    continue

                abs_pos = boundary.start_pos + start_pos
                sub_page = get_page_number(abs_pos)
                ex_num = f"{parent_num}.{sub_num}"

                exercise_id = _generate_exercise_id(
                    course_code, pdf_content.file_path.name, ex_num, abs_pos
                )

                exercises.append(Exercise(
                    id=exercise_id,
                    text=sub_text,
                    page_number=sub_page,
                    exercise_number=ex_num,
                    has_images=False,
                    image_data=[],
                    has_latex=False,
                    latex_content=None,
                    source_pdf=pdf_content.file_path.name,
                    parent_exercise_number=parent_num,
                    sub_question_marker=sub_num,
                    is_sub_question=True,
                ))
        else:
            # No sub-questions - create single exercise
            exercise_id = _generate_exercise_id(
                course_code, pdf_content.file_path.name, parent_num, boundary.start_pos
            )

            exercises.append(Exercise(
                id=exercise_id,
                text=parent_text,
                page_number=page_num,
                exercise_number=parent_num,
                has_images=False,
                image_data=[],
                has_latex=False,
                latex_content=None,
                source_pdf=pdf_content.file_path.name,
            ))

    return exercises


def _normalize_unicode(s: str) -> str:
    """Normalize Unicode for fuzzy matching.

    Handles:
    - Ligatures: ﬃ→ffi, ﬁ→fi, ﬂ→fl
    - Superscripts: ³→3, ²→2, ¹→1
    - Subscripts: ₀→0, ₁→1, ₂→2
    - Smart quotes: ''→', ""→"
    - Accents: é→e (via NFKD decomposition)
    """
    import unicodedata

    # Pre-NFKD: remove chars that NFKD mangles into space + combining
    s = s.replace('\u00b4', '')  # Acute accent

    # NFKD decomposition handles ligatures and compatibility chars
    s = unicodedata.normalize('NFKD', s)

    # Remove combining marks (accents) - keep base chars
    s = ''.join(c for c in s if not unicodedata.combining(c))

    # Additional replacements PDF OCR often produces
    replacements = {
        '³': '3', '²': '2', '¹': '1',
        '₀': '0', '₁': '1', '₂': '2', '₃': '3',
        '→': '->', '←': '<-', '∈': 'in',
        '\u2018': "'", '\u2019': "'",  # Curly single quotes
        '\u201c': '"', '\u201d': '"',  # Curly double quotes
        '`': "'",
        # PDF Private Use Area chars (mathematical delimiters)
        '\uf8eb': '(', '\uf8ed': '(',  # Left brackets
        '\uf8f6': ')', '\uf8f8': ')',  # Right brackets
        # Normalize all brackets to parentheses (LLM uses []{} for matrices/sets, PDF uses special chars → ())
        '[': '(', ']': ')', '{': '(', '}': ')',
    }
    for old, new in replacements.items():
        s = s.replace(old, new)

    return s


def _fuzzy_find(text: str, search_term: str, start_from: int = 0) -> int:
    """Find a term in text with tolerance for OCR errors.

    Handles common OCR issues:
    - Case differences
    - Extra/missing whitespace (including newlines in PDFs)
    - Unicode normalization (ligatures, superscripts, accents)

    Args:
        text: Document text to search
        search_term: Term to find
        start_from: Position to start searching from

    Returns:
        Position of match, or -1 if not found
    """
    # First try exact match
    pos = text.find(search_term, start_from)
    if pos >= 0:
        return pos

    # Try case-insensitive
    text_lower = text.lower()
    search_lower = search_term.lower()
    pos = text_lower.find(search_lower, start_from)
    if pos >= 0:
        return pos

    # Try with normalized whitespace - handles PDF line breaks
    words = search_term.split()
    if words:
        pattern_parts = [re.escape(word) for word in words]
        search_pattern = r'\s+'.join(pattern_parts)
        pattern = re.compile(search_pattern, re.IGNORECASE)
        match = pattern.search(text, start_from)
        if match:
            return match.start()

    # Try with Unicode normalization (ligatures, superscripts, accents)
    text_norm = _normalize_unicode(text)
    search_norm = _normalize_unicode(search_term)
    pos = text_norm.lower().find(search_norm.lower(), start_from)
    if pos >= 0:
        return pos

    # Try normalized + whitespace flexibility (handles "inR" vs "in R")
    words = search_norm.split()
    if words:
        # Allow optional whitespace between any characters
        pattern_parts = [re.escape(word) for word in words]
        search_pattern = r'\s*'.join(pattern_parts)
        pattern = re.compile(search_pattern, re.IGNORECASE)
        match = pattern.search(text_norm, start_from)
        if match:
            return match.start()

    # Last resort: strip all non-alphanumeric chars for prefix match
    # Handles all punctuation differences (commas, brackets, parens, etc.)
    text_alnum = re.sub(r'[^a-z0-9]', '', text_norm.lower())
    search_alnum = re.sub(r'[^a-z0-9]', '', search_norm.lower())
    search_prefix = search_alnum[:50]  # Can use longer prefix since only alphanum

    if search_prefix and len(search_prefix) <= len(text_alnum):
        idx = text_alnum.find(search_prefix)
        if idx >= 0:
            # Map alnum index back to original text position
            alnum_count = 0
            for i, c in enumerate(text_norm):
                if i < start_from:
                    if c.isalnum():
                        alnum_count += 1
                    continue
                if c.isalnum():
                    if alnum_count == idx:
                        return i
                    alnum_count += 1

    return -1


def _fuzzy_rfind(text: str, search_term: str, end_before: int = None) -> int:
    """Find a term in text searching from end, with tolerance for OCR errors.

    Like _fuzzy_find but returns LAST occurrence instead of first.
    Use for end markers where we want the final occurrence.

    Args:
        text: Document text to search
        search_term: Term to find
        end_before: Position to end searching before (searches text[:end_before])

    Returns:
        Position of last match, or -1 if not found
    """
    if end_before is None:
        end_before = len(text)

    search_text = text[:end_before]

    # First try exact match from end
    pos = search_text.rfind(search_term)
    if pos >= 0:
        return pos

    # Try case-insensitive from end
    text_lower = search_text.lower()
    search_lower = search_term.lower()
    pos = text_lower.rfind(search_lower)
    if pos >= 0:
        return pos

    # Try with normalized whitespace - find ALL matches, take last
    words = search_term.split()
    if words:
        pattern_parts = [re.escape(word) for word in words]
        search_pattern = r'\s+'.join(pattern_parts)
        pattern = re.compile(search_pattern, re.IGNORECASE)
        matches = list(pattern.finditer(search_text))
        if matches:
            return matches[-1].start()

    # Try with Unicode normalization
    text_norm = _normalize_unicode(search_text)
    search_norm = _normalize_unicode(search_term)
    pos = text_norm.lower().rfind(search_norm.lower())
    if pos >= 0:
        return pos

    # Try normalized + whitespace flexibility - find all, take last
    words = search_norm.split()
    if words:
        pattern_parts = [re.escape(word) for word in words]
        search_pattern = r'\s*'.join(pattern_parts)
        pattern = re.compile(search_pattern, re.IGNORECASE)
        matches = list(pattern.finditer(text_norm))
        if matches:
            return matches[-1].start()

    # Last resort: alphanumeric suffix match (search from end)
    text_alnum = re.sub(r'[^a-z0-9]', '', text_norm.lower())
    search_alnum = re.sub(r'[^a-z0-9]', '', search_norm.lower())
    search_suffix = search_alnum[-50:] if len(search_alnum) > 50 else search_alnum

    if search_suffix and len(search_suffix) <= len(text_alnum):
        idx = text_alnum.rfind(search_suffix)
        if idx >= 0:
            # Map alnum index back to original text position
            alnum_count = 0
            last_match_pos = -1
            for i, c in enumerate(text_norm):
                if c.isalnum():
                    if alnum_count == idx:
                        last_match_pos = i
                        break
                    alnum_count += 1
            if last_match_pos >= 0:
                return last_match_pos

    return -1


def _fix_decimal_pattern(pattern_str: str) -> str:
    r"""Add negative lookahead after dots in numbered patterns to avoid matching decimals.

    Transforms patterns like:
    - ``(\d+)\.\s*``  ->  ``(\d+)\.(?!\d)\s*``
    - ``(\d+)\.``     ->  ``(\d+)\.(?!\d)``

    Does NOT transform nested numbering patterns like:
    - ``(\d+\.\d+)``  ->  unchanged (matches "1.1", "1.2", etc.)

    This prevents matching "0.3" or "2.5" while still matching "1. " or "2. ".

    Args:
        pattern_str: Original regex pattern

    Returns:
        Fixed pattern with (?!\d) after vulnerable dots
    """
    if not pattern_str:
        return pattern_str

    import re
    # Only fix \d+\. that is followed by \s, ), or end of pattern (not \d)
    # This preserves nested numbering like \d+\.\d+ for "1.1" format
    fixed = re.sub(
        r'(\\d\+\)?\\\.)(\\s|\)|$)',
        r'\1(?!\\d)\2',
        pattern_str
    )
    return fixed


def _generate_exercise_id(
    course_code: str,
    source_pdf: str,
    exercise_number: str,
    char_position: int,
) -> str:
    """Generate a unique exercise ID.

    Args:
        course_code: Course code
        source_pdf: Source PDF filename
        exercise_number: Exercise number (e.g. "1", "1.a", "2.b")
        char_position: Character position in document (guarantees uniqueness)

    Returns:
        Unique exercise ID
    """
    # char_position ensures uniqueness even with duplicate exercise numbers
    components = f"{course_code}_{source_pdf}_{exercise_number}_{char_position}"
    hash_obj = hashlib.md5(components.encode())
    short_hash = hash_obj.hexdigest()[:12]
    course_abbrev = course_code.lower().replace('b', '').replace('0', '')[:6]
    ex_num_clean = exercise_number.replace(".", "_")
    return f"{course_abbrev}_{ex_num_clean}_{short_hash}"


def _split_unstructured(
    pdf_content: "PDFContent",
    course_code: str,
) -> List[Exercise]:
    """Fallback: split document by pages when no markers found.

    Args:
        pdf_content: PDF content
        course_code: Course code

    Returns:
        List of exercises (one per page with substantial content)
    """
    exercises: List[Exercise] = []
    counter = 0

    for page in pdf_content.pages:
        text = page.text.strip()
        if len(text) < 50:
            continue  # Skip empty/header pages

        counter += 1
        exercise_id = _generate_exercise_id(
            course_code,
            pdf_content.file_path.name,
            page.page_number,
            counter,
        )

        exercises.append(Exercise(
            id=exercise_id,
            text=text,
            page_number=page.page_number,
            exercise_number=str(counter),
            has_images=len(page.images) > 0,
            image_data=page.images if page.images else [],
            has_latex=page.has_latex,
            latex_content=page.latex_content,
            source_pdf=pdf_content.file_path.name,
        ))

    return exercises


class ExerciseSplitter:
    """Language-agnostic exercise splitter using dynamic pattern detection."""

    # Structural patterns (language-agnostic fallback)
    STRUCTURAL_PATTERNS = [
        r'(?:^|\n)\s*(\d+)\.\s+',       # "1. " at line start
        r'(?:^|\n)\s*(\d+)\)\s+',       # "1) " at line start
        r'(?:^|\n)\s*\((\d+)\)\s*',     # "(1)" at line start
        r'(?:^|\n)\s*\[(\d+)\]',        # "[1]" at line start
        r'(?:^|\n)\s*([IVXLCDM]+)\.\s', # Roman numerals "I. ", "II. "
    ]

    # Language-agnostic instruction patterns (structural, not language-specific)
    INSTRUCTION_PATTERNS = [
        r'(?:^|\n)\s*[-•]\s+',          # Bullet points (likely instructions)
        r':\s*$',                        # Lines ending with colon (likely headers)
    ]

    def __init__(self):
        """Initialize exercise splitter."""
        self.structural_patterns = [re.compile(p, re.MULTILINE | re.IGNORECASE)
                                   for p in self.STRUCTURAL_PATTERNS]
        self.instruction_patterns = [re.compile(p, re.MULTILINE | re.IGNORECASE)
                                    for p in self.INSTRUCTION_PATTERNS]
        self.exercise_counter = 0
        self._detected_pattern_cache: Dict[str, Optional[re.Pattern]] = {}

    def split_pdf_content(self, pdf_content: PDFContent, course_code: str) -> List[Exercise]:
        """Split PDF content into individual exercises.

        Args:
            pdf_content: Extracted PDF content
            course_code: Course code for ID generation

        Returns:
            List of extracted exercises
        """
        exercises = []
        self.exercise_counter = 0  # Reset counter for each PDF

        # Step 1: Analyze FULL document to detect exercise pattern
        # This is crucial for PDFs where each page has only one exercise marker
        full_text = "\n".join(page.text for page in pdf_content.pages)
        self._document_pattern = self._detect_exercise_pattern(full_text)

        # Process each page using the document-wide pattern
        for page in pdf_content.pages:
            page_exercises = self._split_page(page, pdf_content.file_path.name, course_code)
            exercises.extend(page_exercises)

        # Clean up
        self._document_pattern = None

        return exercises

    def split_pdf_smart(
        self,
        pdf_content: PDFContent,
        course_code: str,
        llm_manager: "LLMManager",
        second_pass_llm: Optional["LLMManager"] = None,
    ) -> List[Exercise]:
        """Split PDF into exercises using LLM-based detection.

        Smart Split Flow (parallel where possible, optimized with has_sub_questions):
        1. _detect_exercises: Parent exercise start markers + has_solutions (batched)
        2. _analyze_boundaries: end_pos + has_sub_questions (parallel per exercise)
        3. _get_sub_start_markers_parallel: Sub-question start markers (parallel, only if has_subs)
        4. _get_sub_end_markers_parallel: Sub-question end markers (parallel, only if has_subs)
        5. _get_context_summaries_parallel: Context summaries (parallel, for ALL exercises)

        Args:
            pdf_content: Extracted PDF content
            course_code: Course code for ID generation
            llm_manager: LLM manager for Call 1 (detection)
            second_pass_llm: Optional LLM for Calls 2-5. If not provided, uses llm_manager.

        Returns:
            List of extracted exercises with context
        """
        # Default second_pass_llm to llm_manager (typically DeepSeek - same quality, 20x cheaper)
        if second_pass_llm is None:
            second_pass_llm = llm_manager

        # Step 1: Concatenate all page text with position tracking
        full_text = ""
        page_lookup: Dict[int, int] = {}  # char_position -> page_number

        for page in pdf_content.pages:
            page_lookup[len(full_text)] = page.page_number
            full_text += page.text + "\n"

        if not full_text.strip():
            return []

        # Step 2: Detect pattern/markers with LLM
        logger.info("Detecting exercise pattern with LLM...")
        detection = _detect_exercises(full_text[:30000], llm_manager)

        if not detection:
            # No detection - try regex fallback, then page-based
            logger.info("No LLM detection, trying regex fallback...")
            regex_pattern = self._detect_exercise_pattern(full_text)

            if regex_pattern:
                # Use the old page-based method with detected pattern
                self._document_pattern = regex_pattern
                exercises = []
                for page in pdf_content.pages:
                    page_exercises = self._split_page(
                        page, pdf_content.file_path.name, course_code
                    )
                    exercises.extend(page_exercises)
                self._document_pattern = None
                return exercises

            # No pattern at all - fall back to page-based splitting
            logger.info("No pattern found, falling back to page-based splitting")
            return _split_unstructured(pdf_content, course_code)

        if detection.explicit_exercises:
            # Smart Split: Explicit exercises with parallel analysis
            logger.info(f"Smart Split: {len(detection.explicit_exercises)} exercises detected")
            boundaries = _find_explicit_exercises(full_text, detection.explicit_exercises)

            if second_pass_llm:
                # Call 2: Analyze exercises (end_pos + has_sub_questions, parallel)
                exercise_analysis = asyncio.run(
                    _analyze_boundaries(boundaries, full_text, second_pass_llm)
                )

                # Update boundaries with accurate end positions from Call 2
                for boundary in boundaries:
                    analysis = exercise_analysis.get(boundary.number)
                    if analysis:
                        boundary.end_pos = analysis.end_pos

                # Filter exercises that have sub-questions
                boundaries_with_subs = [
                    b for b in boundaries
                    if exercise_analysis.get(b.number, ExerciseAnalysis(0, True)).has_sub_questions
                ]
                standalone_count = len(boundaries) - len(boundaries_with_subs)

                if standalone_count > 0:
                    logger.info(f"Skipping Calls 3-4 for {standalone_count} standalone exercises (Call 5 runs for all)")

                # Call 3: Sub-question start markers (parallel, only for exercises with subs)
                if boundaries_with_subs:
                    exercises_for_call3 = [
                        (b.number, full_text[b.start_pos:exercise_analysis[b.number].end_pos])
                        for b in boundaries_with_subs
                    ]
                    explicit_subs = asyncio.run(
                        _get_sub_start_markers_parallel(exercises_for_call3, second_pass_llm)
                    )

                    # Validation: if only 1 sub found, Call 3 likely failed - treat as standalone
                    for ex_num in list(explicit_subs.keys()):
                        if len(explicit_subs[ex_num]) == 1:
                            logger.warning(f"Exercise {ex_num}: expected multiple subs, got 1 - treating as standalone")
                            del explicit_subs[ex_num]

                    # Update boundaries_with_subs to only include exercises that still have subs
                    boundaries_with_subs = [b for b in boundaries_with_subs if b.number in explicit_subs]
                else:
                    explicit_subs = {}
            else:
                explicit_subs = {}
                exercise_analysis = {}
                boundaries_with_subs = []

            exercises = _create_exercises_from_explicit_boundaries(
                boundaries, explicit_subs, full_text, pdf_content, course_code, page_lookup
            )

            # Call 4: Get end markers to trim trailing artifacts (only for exercises with subs)
            if second_pass_llm and explicit_subs:
                sub_exercises = [ex for ex in exercises if ex.is_sub_question]
                if sub_exercises:
                    sub_questions_for_call4 = [
                        (f"{ex.parent_exercise_number}.{ex.sub_question_marker}", ex.text)
                        for ex in sub_exercises
                    ]
                    end_markers = asyncio.run(
                        _get_sub_end_markers_parallel(sub_questions_for_call4, second_pass_llm)
                    )
                    # Apply end markers to trim text
                    for ex in sub_exercises:
                        sub_id = f"{ex.parent_exercise_number}.{ex.sub_question_marker}"
                        end_marker = end_markers.get(sub_id)
                        if end_marker:
                            pos = _fuzzy_rfind(ex.text, end_marker)
                            if pos >= 0:
                                ex.text = ex.text[:pos + len(end_marker)]

            # Call 5: Context summaries (parallel, for ALL exercises)
            if second_pass_llm and exercise_analysis:
                # Build list of all exercises: parents (has_sub_questions=True) + standalone (False)
                boundaries_with_subs_numbers = {b.number for b in boundaries_with_subs}
                exercises_for_call5 = []
                for b in boundaries:
                    if b.number in exercise_analysis:
                        end_pos = exercise_analysis[b.number].end_pos
                        exercise_text = full_text[b.start_pos:end_pos]
                        has_subs = b.number in boundaries_with_subs_numbers
                        exercises_for_call5.append((b.number, exercise_text, has_subs))

                context_summaries = asyncio.run(
                    _get_context_summaries_parallel(exercises_for_call5, second_pass_llm)
                )
                # Apply context summaries to exercises
                for ex in exercises:
                    if ex.is_sub_question and ex.parent_exercise_number:
                        # Sub-questions inherit parent context
                        ctx = context_summaries.get(ex.parent_exercise_number)
                        if ctx:
                            ex.exercise_context = ctx
                    elif not ex.is_sub_question:
                        # Standalone exercises get their own context
                        ctx = context_summaries.get(str(ex.exercise_number))
                        if ctx:
                            ex.exercise_context = ctx

            # Enrich with page data and return
            exercises = self._enrich_with_page_data(exercises, pdf_content)
            logger.info(f"Extracted {len(exercises)} exercises")
            return exercises

        # No explicit exercises detected - fall back to unstructured
        logger.warning("No exercises detected, falling back to unstructured split")
        return _split_unstructured(pdf_content, course_code)

    def _enrich_with_page_data(
        self,
        exercises: List[Exercise],
        pdf_content: PDFContent,
    ) -> List[Exercise]:
        """Enrich exercises with image and latex data from their pages.

        Args:
            exercises: List of exercises (may be missing image/latex data)
            pdf_content: Original PDF content with page data

        Returns:
            Exercises with image and latex data populated
        """
        # Build page lookup
        page_map = {page.page_number: page for page in pdf_content.pages}

        for exercise in exercises:
            page = page_map.get(exercise.page_number)
            if page:
                exercise.has_images = len(page.images) > 0 if page.images else False
                exercise.image_data = page.images if page.images else []
                exercise.has_latex = page.has_latex
                exercise.latex_content = page.latex_content

        return exercises

    def _split_page(self, page: PDFPage, source_pdf: str, course_code: str) -> List[Exercise]:
        """Split a single page into exercises.

        Args:
            page: PDF page content
            source_pdf: Source PDF filename
            course_code: Course code

        Returns:
            List of exercises from this page
        """
        text = page.text
        if not text.strip():
            return []

        # Find all exercise markers FIRST
        markers = self._find_exercise_markers(text)

        if not markers:
            # No markers found on this page
            # Check if this is just an instruction page
            if self._is_instruction_page(text):
                return []  # Skip instruction-only pages

            # If we have a document-wide pattern, pages without markers are likely:
            # - Continuation of previous exercise (don't create new exercise)
            # - Header/instruction pages (already handled above)
            # So skip them to avoid inflating exercise count
            if getattr(self, '_document_pattern', None) is not None:
                return []  # Skip - this is likely continuation text

            # No document pattern AND no page markers - fallback behavior:
            # Treat entire page as single exercise if it has substantial content
            if len(text.strip()) < 50:  # Too short to be a real exercise
                return []

            return [self._create_exercise(
                text=text,
                page_number=page.page_number,
                exercise_number=None,
                images=page.images,
                has_latex=page.has_latex,
                latex_content=page.latex_content,
                source_pdf=source_pdf,
                course_code=course_code
            )]

        # Split text at markers
        exercises = []
        for i, (start_pos, ex_number) in enumerate(markers):
            # Find end position (start of next exercise or end of text)
            if i + 1 < len(markers):
                end_pos = markers[i + 1][0]
            else:
                end_pos = len(text)

            exercise_text = text[start_pos:end_pos].strip()

            if exercise_text:
                # For now, assign all images from the page to each exercise
                # In a more sophisticated version, we could detect which images
                # belong to which exercise based on position
                exercises.append(self._create_exercise(
                    text=exercise_text,
                    page_number=page.page_number,
                    exercise_number=ex_number,
                    images=page.images if page.images else [],
                    has_latex=page.has_latex,
                    latex_content=page.latex_content,
                    source_pdf=source_pdf,
                    course_code=course_code
                ))

        return exercises

    def _detect_exercise_pattern(self, text: str) -> Optional[re.Pattern]:
        """Detect the exercise pattern used in this document dynamically.

        Language-agnostic: Analyzes text to find recurring exercise markers
        instead of hardcoding patterns like "Esercizio", "Exercise", etc.

        Args:
            text: Text to analyze

        Returns:
            Compiled pattern if found, None otherwise
        """
        # Check cache first (use hash of first 1000 chars as key)
        cache_key = str(hash(text[:1000]))
        if cache_key in self._detected_pattern_cache:
            return self._detected_pattern_cache[cache_key]

        # Look for repeated pattern: <word> <number> appearing multiple times
        # E.g., "Esercizio 1", "Esercizio 2" → pattern is "Esercizio"
        # Supports any language including CJK characters
        word_num_pattern = r'\b([A-Za-z\u00C0-\u024F\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]+)\s+(\d+)\b'
        matches = re.findall(word_num_pattern, text, re.IGNORECASE)

        # Count which words appear with multiple different numbers
        word_counts: Dict[str, set] = {}
        for word, num in matches:
            word_lower = word.lower()
            if word_lower not in word_counts:
                word_counts[word_lower] = set()
            word_counts[word_lower].add(num)

        # Find words that appear with 2+ different numbers (likely exercise markers)
        exercise_words = [(w, len(nums)) for w, nums in word_counts.items() if len(nums) >= 2]

        if exercise_words:
            # Use the word that appears with most different numbers
            exercise_words.sort(key=lambda x: x[1], reverse=True)
            word = exercise_words[0][0]
            pattern = re.compile(rf'(?:^|\n)\s*{re.escape(word)}\s+(\d+)', re.IGNORECASE | re.MULTILINE)
            self._detected_pattern_cache[cache_key] = pattern
            return pattern

        self._detected_pattern_cache[cache_key] = None
        return None

    def _find_exercise_markers(self, text: str) -> List[Tuple[int, str]]:
        """Find all exercise markers in text using dynamic detection.

        Strategy (language-agnostic):
        1. Use document-wide pattern if available (detected from full PDF)
        2. Fall back to page-level pattern detection
        3. Fall back to structural patterns (1., 2., etc.) if no word pattern found

        Args:
            text: Text to search

        Returns:
            List of tuples (position, exercise_number)
        """
        markers = []

        # Step 1: Use document-wide pattern if available (set by split_pdf_content)
        # This handles PDFs where each page has only one exercise marker
        detected_pattern = getattr(self, '_document_pattern', None)
        if detected_pattern is None:
            # Fall back to page-level detection
            detected_pattern = self._detect_exercise_pattern(text)

        if detected_pattern:
            for match in detected_pattern.finditer(text):
                position = match.start()
                ex_number = match.group(1) if match.groups() else None
                markers.append((position, ex_number))

        # If dynamic patterns found exercises, use those
        if markers:
            markers = list(set(markers))
            markers.sort(key=lambda x: x[0])
            return markers

        # Step 2: Fall back to structural patterns (1., 2., etc.)
        for pattern in self.structural_patterns:
            # Collect ALL matches first to calculate gaps correctly
            all_matches = [(m.start(), m.group(1) if m.groups() else None)
                          for m in pattern.finditer(text)]

            for i, (position, ex_number) in enumerate(all_matches):
                # Calculate fragment length (distance to next marker or end)
                if i + 1 < len(all_matches):
                    next_marker_pos = all_matches[i + 1][0]
                else:
                    next_marker_pos = len(text)

                fragment_length = next_marker_pos - position
                if fragment_length < 30:  # Minimum 30 chars (allows short Q&A questions)
                    continue

                markers.append((position, ex_number))

        # Remove duplicates and sort by position
        markers = list(set(markers))
        markers.sort(key=lambda x: x[0])

        return markers

    def _is_instruction_page(self, text: str) -> bool:
        """Check if a page contains only instructions (not exercises).

        Language-agnostic: Uses structural patterns instead of language-specific text.

        Args:
            text: Page text

        Returns:
            True if this is an instruction-only page
        """
        # Language-agnostic structural indicators of instruction pages
        # These patterns work across languages
        structural_indicators = [
            r'(?:^|\n)\s*[-•]\s+.{10,}',  # Multiple bullet points
            r':\s*\n',                     # Lines ending with colon then newline
            r'\b\d{4}[-/]\d{2}[-/]\d{2}\b',  # Date patterns (exam dates)
        ]

        # Count structural instruction patterns
        matches = sum(1 for pattern in [re.compile(p, re.MULTILINE) for p in structural_indicators]
                     if len(pattern.findall(text)) >= 2)

        # If page has many instruction-like structures and is short, likely instructions
        if matches >= 2 and len(text.strip()) < 500:
            return True

        # Check if there are NO exercise-like patterns (no repeated word+number)
        # but there ARE multiple bullet points
        detected_pattern = self._detect_exercise_pattern(text)
        bullet_count = len(re.findall(r'(?:^|\n)\s*[-•]\s+', text, re.MULTILINE))

        if detected_pattern is None and bullet_count >= 5:
            return True

        return False

    def _create_exercise(self, text: str, page_number: int,
                        exercise_number: Optional[str],
                        images: List[bytes], has_latex: bool,
                        latex_content: Optional[str], source_pdf: str,
                        course_code: str) -> Exercise:
        """Create an Exercise object.

        Args:
            text: Exercise text
            page_number: Page number
            exercise_number: Exercise number (if detected)
            images: Image data
            has_latex: Whether LaTeX was detected
            latex_content: LaTeX content
            source_pdf: Source PDF filename
            course_code: Course code

        Returns:
            Exercise object
        """
        # Generate unique ID
        exercise_id = self._generate_exercise_id(
            course_code, source_pdf, page_number, exercise_number
        )

        return Exercise(
            id=exercise_id,
            text=text,
            page_number=page_number,
            exercise_number=exercise_number,
            has_images=len(images) > 0,
            image_data=images,
            has_latex=has_latex,
            latex_content=latex_content,
            source_pdf=source_pdf
        )

    def _generate_exercise_id(self, course_code: str, source_pdf: str,
                             page_number: int, exercise_number: Optional[str]) -> str:
        """Generate a unique exercise ID.

        Args:
            course_code: Course code
            source_pdf: Source PDF filename
            page_number: Page number
            exercise_number: Exercise number

        Returns:
            Unique exercise ID
        """
        # Increment counter to ensure uniqueness
        self.exercise_counter += 1

        # Create a hash from ALL components including counter for guaranteed uniqueness
        components = f"{course_code}_{source_pdf}_{page_number}_{exercise_number or 'none'}_{self.exercise_counter}"

        # Generate hash
        hash_obj = hashlib.md5(components.encode())
        short_hash = hash_obj.hexdigest()[:12]

        # Create ID: course abbreviation + counter + hash
        course_abbrev = course_code.lower().replace('b', '').replace('0', '')[:6]
        return f"{course_abbrev}_{self.exercise_counter:04d}_{short_hash}"

    def merge_split_exercises(self, exercises: List[Exercise]) -> List[Exercise]:
        """Merge exercises that were incorrectly split.

        This is a placeholder for future enhancement where we might
        use AI to detect when an exercise was split across pages.

        Args:
            exercises: List of exercises

        Returns:
            Merged list of exercises
        """
        # For now, just return as-is
        # In Phase 3, we could use LLM to detect split exercises
        return exercises

    def validate_exercise(self, exercise: Exercise) -> bool:
        """Validate if an exercise has content.

        Args:
            exercise: Exercise to validate

        Returns:
            True if exercise is valid (non-empty)
        """
        return bool(exercise.text.strip())

    def clean_exercise_text(self, text: str) -> str:
        """Clean up exercise text.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        # Remove page numbers (language-agnostic structural pattern)
        # Pattern: short line with just a number, or "word + number" where line is short
        text = re.sub(r'(?:^|\n)\s*\d+\s*(?:\n|$)', '\n', text)  # Standalone numbers
        text = re.sub(r'(?:^|\n)\s*[A-Za-z]+\s+\d+\s*(?:\n|$)', '\n', text)  # "Word 123" short lines

        # Strip leading/trailing whitespace
        text = text.strip()

        return text
