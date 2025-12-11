"""
Exercise splitting for Examina.
Splits PDF content into individual exercises based on patterns.
"""

import re
import json
import hashlib
import logging
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
    # Parent context (LLM-generated summary of parent exercise setup)
    parent_context: Optional[str] = None

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
    pattern: Optional[MarkerPattern] = None  # Pattern-based detection
    explicit_markers: Optional[List[str]] = None  # Legacy: simple marker texts
    explicit_exercises: Optional[List[ExplicitExercise]] = None  # New: with end markers


def _detect_pattern_with_llm(
    text_sample: str,
    llm_manager: "LLMManager",
) -> Optional[DetectionResult]:
    """Use LLM to detect exercise markers in document.

    Two modes:
    1. Pattern detection: LLM returns regex patterns for exercise/sub-markers
    2. Explicit markers: LLM lists the first few words of each exercise

    Args:
        text_sample: First ~10k chars of document
        llm_manager: LLM manager for inference

    Returns:
        DetectionResult with either pattern or explicit markers, None if detection fails
    """
    prompt = """Analyze this exam/exercise document and return REGEX PATTERNS for parsing.

TEXT SAMPLE:
---
{text}
---

Identify the exact patterns used and return Python regex patterns:

1. EXERCISE_PATTERN - Regex matching exercise markers. Must have a capture group for the exercise number.
   Examples:
   - "keyword\\s+(\\d+)" matches "Keyword 1", "Keyword 2" (any language keyword)
   - "(\\d+)\\." matches "1.", "2." (if no keyword, just numbers)

2. SOLUTION_PATTERN - Keyword or regex for solution sections (if any).

IMPORTANT: Do NOT include inline flags like (?i), (?m), (?s) in patterns. Case-insensitivity and multiline matching are applied automatically.

Return ONLY valid JSON:
{{"mode": "pattern", "exercise_pattern": "regex string", "solution_pattern": "keyword or null"}}

If NO consistent pattern exists, return explicit markers with question boundaries:
{{"mode": "explicit", "exercises": [
  {{"number": "1", "start_marker": "first ~50 chars of exercise 1", "end_marker": "last ~50 chars of QUESTION only"}},
  {{"number": "2", "start_marker": "first ~50 chars of exercise 2", "end_marker": "last ~50 chars of QUESTION only"}}
]}}

IMPORTANT for end_marker:
- End at the actual QUESTION text, BEFORE any non-question content like:
  - Form fields (blank lines with underscores or dots for student to fill)
  - Repeated exam instructions or rules
  - Solutions or answers
  - Page headers/footers
- The end_marker should be the last sentence of what the student needs to solve.
- Use your understanding of the document structure to identify where the question ends."""

    try:
        llm_response = llm_manager.generate(
            prompt.format(text=text_sample[:30000]),
            temperature=0.0,
        )

        # Check if the response was successful
        if hasattr(llm_response, 'success') and not llm_response.success:
            error_msg = getattr(llm_response, 'error', 'Unknown error')
            logger.warning(f"LLM pattern detection failed: {error_msg}")
            return None

        # Extract text from LLMResponse object
        response = llm_response.text if hasattr(llm_response, 'text') else str(llm_response)

        if not response or not response.strip():
            logger.warning("LLM returned empty response for pattern detection")
            return None

        # Parse JSON response
        # Handle potential markdown code blocks
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()

        data = json.loads(response)

        mode = data.get("mode", "pattern")

        if mode == "explicit":
            # New format with exercises array (includes end_marker)
            exercises = data.get("exercises", [])
            if exercises:
                explicit_exercises = []
                for ex in exercises:
                    explicit_exercises.append(ExplicitExercise(
                        number=str(ex.get("number", "")),
                        start_marker=ex.get("start_marker", ""),
                        end_marker=ex.get("end_marker"),
                    ))
                return DetectionResult(explicit_exercises=explicit_exercises)

            # Legacy format with simple markers array (backward compat)
            markers = data.get("markers", [])
            if markers:
                return DetectionResult(explicit_markers=markers)
            return None

        # Pattern mode - LLM returns regex patterns
        exercise_pattern = data.get("exercise_pattern")
        if not exercise_pattern:
            return None

        # Validate the regex patterns
        try:
            re.compile(exercise_pattern)
        except re.error as e:
            logger.warning(f"Invalid exercise_pattern from LLM: {exercise_pattern} - {e}")
            return None

        # sub_patterns no longer returned from pattern detection (Call 1)
        # Per-exercise sub_patterns determined in Call 3 after parent boundaries known

        return DetectionResult(
            pattern=MarkerPattern(
                exercise_pattern=exercise_pattern,
                sub_patterns=None,  # Determined per-exercise in Call 3
                solution_pattern=data.get("solution_pattern"),
            )
        )

    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        logger.warning(f"Failed to parse LLM pattern detection response: {e}")
        return None
    except Exception as e:
        logger.error(f"LLM pattern detection failed: {e}")
        return None


@dataclass
class ExerciseBoundary:
    """Exercise boundaries found in document."""
    number: str
    start_pos: int
    end_pos: Optional[int]  # None if end_marker not found


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
    explicit_subs: Dict[str, List[Dict[str, str]]],
    full_text: str,
    pdf_content: "PDFContent",
    course_code: str,
    page_lookup: Dict[int, int],
) -> List[Exercise]:
    """Create Exercise objects from explicit boundaries with sub-question support.

    Args:
        boundaries: List of exercise boundaries with start/end positions
        explicit_subs: Dict mapping exercise number to list of sub-question dicts
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
            # Parent has sub-questions - create sub-question exercises
            for sub_idx, sub in enumerate(subs, start=1):
                sub_num = str(sub_idx)  # Auto-generate: 1, 2, 3...
                start_marker = sub.get("start_marker", "")
                end_marker = sub.get("end_marker", "")

                if not start_marker:
                    continue

                # Find start position
                start_pos = _fuzzy_find(parent_text, start_marker)
                if start_pos < 0:
                    logger.warning(f"Could not find sub-question start: {start_marker[:30]}...")
                    continue

                # Find end position
                if end_marker:
                    end_pos = _fuzzy_find(parent_text, end_marker, start_from=start_pos)
                    if end_pos >= 0:
                        end_pos += len(end_marker)  # Include end_marker in text
                    else:
                        end_pos = len(parent_text)
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
                course_code, pdf_content.file_path.name, page_num, i + 1
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


def _find_explicit_markers(
    full_text: str,
    marker_texts: List[str],
) -> List[Marker]:
    """Find markers in document using explicit marker texts from LLM.

    Legacy function for backward compatibility with simple marker format.

    Args:
        full_text: Complete document text
        marker_texts: List of marker texts to find (first few words of each exercise)

    Returns:
        List of Marker objects sorted by position
    """
    markers: List[Marker] = []

    for i, marker_text in enumerate(marker_texts):
        pos = _fuzzy_find(full_text, marker_text)
        if pos >= 0:
            markers.append(Marker(
                marker_type=MarkerType.PARENT,
                marker_text=marker_text,
                number=str(i + 1),
                start_position=pos,
                question_start=pos,  # For explicit markers, question starts at marker
            ))
        else:
            logger.warning(f"Explicit marker not found: '{marker_text}'")

    # Sort by position and deduplicate (in case of overlapping matches)
    markers.sort(key=lambda m: m.start_position)

    # Remove duplicates (same position)
    seen_positions = set()
    unique_markers = []
    for m in markers:
        if m.start_position not in seen_positions:
            seen_positions.add(m.start_position)
            unique_markers.append(m)

    return unique_markers


def _fuzzy_find(text: str, search_term: str, start_from: int = 0) -> int:
    """Find a term in text with tolerance for OCR errors.

    Handles common OCR issues:
    - Case differences
    - Extra/missing spaces
    - Common character substitutions (l/1, O/0)

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

    # Try with normalized whitespace
    search_normalized = re.sub(r'\s+', r'\\s+', re.escape(search_term))
    pattern = re.compile(search_normalized, re.IGNORECASE)
    match = pattern.search(text, start_from)
    if match:
        return match.start()

    return -1


# ============================================================================
# SECOND-PASS: End markers and parent context generation
# ============================================================================


def _get_parent_end_markers(
    parent_markers: List[Marker],
    full_text: str,
    llm_manager: "LLMManager",
) -> Dict[str, int]:
    """Get end positions for each parent exercise using LLM.

    Sonnet call 1: Determines where each parent exercise's question ends.
    This defines boundaries for sub-question detection.

    Args:
        parent_markers: List of parent exercise markers
        full_text: Complete document text
        llm_manager: LLM manager (should be Sonnet for quality)

    Returns:
        Dict mapping parent number to end position (character offset)
    """
    if not parent_markers:
        return {}

    # Pre-compute rough end positions (next parent or end of text)
    rough_ends: Dict[str, int] = {}
    for i, marker in enumerate(parent_markers):
        if i + 1 < len(parent_markers):
            rough_ends[marker.number] = parent_markers[i + 1].start_position
        else:
            rough_ends[marker.number] = len(full_text)

    # Build context for LLM
    exercises_info = []
    for marker in parent_markers:
        start = marker.question_start
        end = rough_ends[marker.number]
        text = full_text[start:end].strip()

        # Truncate for LLM (first 500 + last 200 chars)
        if len(text) > 700:
            text_preview = text[:500] + "\n...[middle truncated]...\n" + text[-200:]
        else:
            text_preview = text

        exercises_info.append({
            "number": marker.number,
            "text": text_preview,
        })

    if not exercises_info:
        return {}

    exercises_text = "\n\n".join([
        f"EXERCISE {ex['number']}:\n\"\"\"\n{ex['text']}\n\"\"\""
        for ex in exercises_info
    ])

    prompt = f"""For each exercise below, identify the end_marker: the LAST 40-60 characters of the ENTIRE exercise question.

The end_marker should be at the END of all question content (INCLUDING any sub-questions like a), b), c) or 1., 2., 3.), BEFORE any:
- Form fields or blank lines for answers
- Solutions or answer sections
- Page headers/footers
- Exam instructions
- Junk text between exercises

EXERCISES:
{exercises_text}

Return JSON:
{{"results": [{{"number": "1", "end_marker": "last 40-60 chars"}}, ...]}}

CRITICAL:
- If exercise ends with a sub-question, end_marker should be from the LAST sub-question
- end_marker marks where THIS exercise ends, before junk or next exercise
- end_marker should be 40-60 characters"""

    try:
        llm_response = llm_manager.generate(prompt, temperature=0.0)
        response_text = llm_response.text if hasattr(llm_response, 'text') else str(llm_response)

        # Parse JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            data = json.loads(json_match.group())
            results = data.get("results", [])

            # Convert end_markers to positions
            end_positions: Dict[str, int] = {}
            for item in results:
                num = item.get("number")
                end_marker = item.get("end_marker", "")

                if not num or not end_marker:
                    continue

                # Find this parent's text range
                marker = next((m for m in parent_markers if m.number == num), None)
                if not marker:
                    continue

                start = marker.question_start
                rough_end = rough_ends.get(num, len(full_text))
                parent_text = full_text[start:rough_end]

                # Find end_marker position in parent text (search for LAST occurrence)
                clean_marker = end_marker.strip().strip("...").strip("…").strip()
                if len(clean_marker) >= 10:
                    # Try rfind first for exact match (faster, finds last occurrence)
                    pos = parent_text.rfind(clean_marker)
                    if pos < 0:
                        # Fall back to fuzzy find from start
                        pos = _fuzzy_find(parent_text, clean_marker, 0)
                    if pos >= 0:
                        # Convert to absolute position
                        end_positions[num] = start + pos + len(clean_marker)
                    else:
                        # Fall back to rough end
                        end_positions[num] = rough_end
                else:
                    end_positions[num] = rough_end

            # Fill in any missing parents with rough ends
            for marker in parent_markers:
                if marker.number not in end_positions:
                    end_positions[marker.number] = rough_ends[marker.number]

            return end_positions

    except Exception as e:
        logger.warning(f"Parent end marker detection failed: {e}")

    # Fall back to rough ends
    return {m.number: rough_ends[m.number] for m in parent_markers}


def _get_per_exercise_sub_patterns(
    parent_markers: List[Marker],
    full_text: str,
    parent_boundaries: Dict[str, Tuple[int, int]],
    llm_manager: "LLMManager",
) -> Dict[str, Optional[List[str]]]:
    """Determine sub_patterns for each exercise individually (Call 3).

    This replaces the global sub_patterns from Call 1. By analyzing each
    exercise's full text, we can distinguish real sub-questions from
    inline conditions like "i) X or ii) Y".

    Args:
        parent_markers: List of parent exercise markers
        full_text: Full document text
        parent_boundaries: Dict of exercise_number -> (start, end) positions
        llm_manager: LLM for detection

    Returns:
        Dict mapping exercise_number -> list of sub_patterns (or None if no subs)
    """
    if not parent_markers or not parent_boundaries:
        return {}

    logger.info(f"Detecting per-exercise sub_patterns for {len(parent_markers)} exercises...")

    # Build exercise texts for LLM
    MAX_EXERCISE_CHARS = 15000  # ~4k tokens per exercise
    exercises_text_parts = []

    for marker in parent_markers:
        num = marker.number
        if num not in parent_boundaries:
            continue

        start, end = parent_boundaries[num]
        exercise_text = full_text[start:end].strip()

        # Truncate if too long (keep start and end where sub-patterns likely appear)
        original_len = len(exercise_text)
        if original_len > MAX_EXERCISE_CHARS:
            half = MAX_EXERCISE_CHARS // 2
            exercise_text = exercise_text[:half] + "\n...[truncated]...\n" + exercise_text[-half:]
            logger.info(f"Exercise {num} text truncated from {original_len} to {MAX_EXERCISE_CHARS} chars")

        exercises_text_parts.append(f"Exercise {num}:\n---\n{exercise_text}\n---")

    if not exercises_text_parts:
        return {}

    exercises_text = "\n\n".join(exercises_text_parts)

    prompt = f"""For each exercise below, determine if it contains sub-questions.

Sub-questions are SEPARATE TASKS requiring SEPARATE ANSWERS.

EXERCISES:
{exercises_text}

Return JSON with per-exercise sub_patterns (regex patterns to match sub-question markers).

Regex pattern examples:
- ["([a-z])\\\\s*[).]"] matches "a)", "b.", "c)" (letter markers)
- ["(\\\\d+)\\\\.\\\\s"] matches "1. ", "2. " (numbered markers)
- ["[-•*]\\\\s+"] matches "- ", "• ", "* " (bullets)
- ["(\\\\d+)([a-z])\\\\s*[).]"] matches "1a)", "1b)", "2a)" (number+letter)
- ["([a-z])(\\\\d+)\\\\s*[).]"] matches "a1)", "a2)", "b1)" (letter+number)
- null if no sub-questions

{{"results": [
  {{"number": "<exercise_number>", "sub_patterns": ["<regex>"] or null}}
]}}

Return null for sub_patterns if exercise has no sub-questions."""

    try:
        llm_response = llm_manager.generate(prompt, temperature=0.0)
        response_text = llm_response.text if hasattr(llm_response, 'text') else str(llm_response)

        # Parse JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if not json_match:
            logger.warning("No JSON found in sub_patterns response")
            return {}

        data = json.loads(json_match.group())
        results_list = data.get("results", [])

        # Convert to dict and validate patterns
        results: Dict[str, Optional[List[str]]] = {}
        for item in results_list:
            num = item.get("number")
            patterns = item.get("sub_patterns")

            if not num:
                continue

            if patterns is None:
                results[num] = None
                logger.debug(f"Exercise {num}: no sub-questions")
                continue

            # Validate each pattern
            valid_patterns = []
            for p in patterns:
                if not p:
                    continue
                try:
                    re.compile(p)
                    valid_patterns.append(p)
                except re.error as e:
                    logger.warning(f"Invalid sub_pattern for exercise {num}: {p} - {e}")

            results[num] = valid_patterns if valid_patterns else None
            if valid_patterns:
                logger.info(f"Exercise {num}: sub_patterns={valid_patterns}")
            else:
                logger.debug(f"Exercise {num}: no valid sub-patterns")

        # Summary
        with_subs = sum(1 for p in results.values() if p)
        logger.info(f"Sub-pattern detection complete: {with_subs}/{len(parent_markers)} exercises have sub-questions")

        return results

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse sub_patterns response: {e}")
        return {}
    except Exception as e:
        logger.warning(f"Per-exercise sub_pattern detection failed: {e}")
        return {}


def _get_explicit_sub_questions(
    boundaries: List["ExerciseBoundary"],
    full_text: str,
    llm_manager: "LLMManager",
) -> Dict[str, List[Dict[str, str]]]:
    """Get explicit sub-question markers for explicit mode (Call 3 explicit).

    Uses same format as explicit parent exercises: start_marker to locate each sub.

    Args:
        boundaries: List of exercise boundaries
        full_text: Full document text
        llm_manager: LLM for detection

    Returns:
        Dict mapping exercise_number -> list of sub-question dicts from LLM
    """
    if not boundaries:
        return {}

    logger.info(f"Detecting explicit sub-questions for {len(boundaries)} exercises...")

    # Build exercise texts for LLM
    MAX_EXERCISE_CHARS = 15000
    exercises_text_parts = []

    for boundary in boundaries:
        num = boundary.number
        start = boundary.start_pos
        end = boundary.end_pos if boundary.end_pos else len(full_text)
        exercise_text = full_text[start:end].strip()

        if len(exercise_text) > MAX_EXERCISE_CHARS:
            half = MAX_EXERCISE_CHARS // 2
            exercise_text = exercise_text[:half] + "\n...[truncated]...\n" + exercise_text[-half:]

        exercises_text_parts.append(f"Exercise {num}:\n---\n{exercise_text}\n---")

    if not exercises_text_parts:
        return {}

    exercises_text = "\n\n".join(exercises_text_parts)

    prompt = f"""For each exercise below, identify sub-questions if any.

Sub-questions are SEPARATE TASKS requiring SEPARATE ANSWERS.

EXERCISES:
{exercises_text}

Return JSON:
{{"results": [
  {{"number": "<exercise_number>", "sub_questions": [
    {{"start_marker": "first ~40 chars", "end_marker": "last ~40 chars"}}
  ] or null}}
]}}

Return null for sub_questions if exercise has no sub-questions."""

    try:
        llm_response = llm_manager.generate(prompt, temperature=0.0)
        response_text = llm_response.text if hasattr(llm_response, 'text') else str(llm_response)

        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if not json_match:
            logger.warning("No JSON found in explicit sub-questions response")
            return {}

        data = json.loads(json_match.group())
        results_list = data.get("results", [])

        results: Dict[str, List[Dict[str, str]]] = {}
        for item in results_list:
            num = item.get("number")
            subs = item.get("sub_questions")

            if not num or not subs:
                continue

            # Pass through whatever LLM returned
            results[num] = subs
            logger.info(f"Exercise {num}: {len(subs)} explicit sub-questions")

        with_subs = len(results)
        logger.info(f"Explicit sub-question detection complete: {with_subs}/{len(boundaries)} exercises have sub-questions")

        return results

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse explicit sub-questions response: {e}")
        return {}
    except Exception as e:
        logger.warning(f"Explicit sub-question detection failed: {e}")
        return {}


def _get_second_pass_results(
    hierarchy: List["ExerciseNode"],
    llm_manager: "LLMManager",
) -> Optional[Dict[str, Dict[str, str]]]:
    """Second LLM pass: get end markers AND context summaries.

    Uses Sonnet (or other high-quality model) for better instruction following.

    Returns dict mapping exercise number to:
    - For sub-questions and standalone: {"end_marker": "..."}
    - For parents with children: {"context_summary": "..."}
    """
    # Build context showing each extracted exercise
    exercises_info = []

    def collect_exercises(nodes: List["ExerciseNode"], parent_num: str = ""):
        for node in nodes:
            num = node.marker.number
            full_num = f"{parent_num}.{num}" if parent_num else num

            # Show text preview (first 500 chars for context, last 200 for end detection)
            text = node.question_text
            if len(text) > 700:
                text_preview = text[:500] + "\n...[middle truncated]...\n" + text[-200:]
            else:
                text_preview = text

            has_children = len(node.children) > 0
            is_sub = node.parent is not None

            exercises_info.append({
                "number": full_num,
                "type": "sub-question" if is_sub else ("parent_with_children" if has_children else "standalone"),
                "has_children": has_children,
                "text": text_preview,
            })

            # Recurse for children
            if node.children:
                collect_exercises(node.children, full_num)

    collect_exercises(hierarchy)

    if not exercises_info:
        return None

    # Build prompt with differentiated output based on type
    exercises_text = "\n\n".join([
        f"EXERCISE {ex['number']} (type: {ex['type']}):\n\"\"\"\n{ex['text']}\n\"\"\""
        for ex in exercises_info
    ])

    prompt = f"""I extracted exercises from an exam PDF. For each one, return either:

1. For "sub-question" or "standalone" exercises:
   Return "end_marker": the LAST 40-60 characters of the actual question (before junk like form fields, page numbers, exam instructions)

2. For "parent_with_children" exercises:
   Return "context_summary": shared info that sub-questions need but don't have in their own text.
   Good context: data, values, parameters, scenario setup that sub-questions reference.
   If parent also contains its own question, include a brief summary of that question.
   Return null if: sub-questions are independent and don't need shared info to be understood.
   IMPORTANT: Always return context_summary in ENGLISH, even if source text is in another language.

EXERCISES:
{exercises_text}

Return JSON with one entry per exercise above. Format:
{{"results": [{{"number": "<exercise_number>", "end_marker": "..."}} or {{"number": "<exercise_number>", "context_summary": "..."}}, ...]}}

CRITICAL:
- end_marker must be from the END of the question, not the beginning
- end_marker should be 40-60 characters
"""

    try:
        llm_response = llm_manager.generate(prompt, temperature=0.0)
        response_text = llm_response.text if hasattr(llm_response, 'text') else str(llm_response)
        # Parse JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            data = json.loads(json_match.group())
            results = data.get("results", [])
            # Convert to dict keyed by number
            return {item["number"]: item for item in results}
    except Exception as e:
        logger.warning(f"Second-pass LLM failed: {e}")

    return None


def _apply_second_pass_results(
    hierarchy: List["ExerciseNode"],
    results: Dict[str, Dict[str, str]],
) -> None:
    """Apply second-pass results to hierarchy (in-place).

    - Apply end_marker trimming to subs and standalone
    - Store context_summary on parent nodes for later use
    """
    def apply_to_nodes(nodes: List["ExerciseNode"], parent_num: str = ""):
        for node in nodes:
            num = node.marker.number
            full_num = f"{parent_num}.{num}" if parent_num else num

            result = results.get(full_num, {})

            # Apply end_marker trimming
            end_marker = result.get("end_marker")
            if end_marker:
                # Clean up marker (strip ellipsis)
                clean_marker = end_marker.strip().strip("...").strip("…").strip()
                if len(clean_marker) >= 10:  # Minimum viable marker length
                    # Find in question_text
                    pos = _fuzzy_find(node.question_text, clean_marker, 0)
                    if pos >= 0:
                        # Trim at end of marker
                        old_len = len(node.question_text)
                        node.question_text = node.question_text[:pos + len(clean_marker)].strip()
                        trimmed = old_len - len(node.question_text)
                        if trimmed > 0:
                            logger.info(f"Exercise {full_num}: trimmed {trimmed} chars at end marker")
                    else:
                        logger.debug(f"Exercise {full_num}: end marker not found in text")

            # Store context_summary on parent node (will be passed to children later)
            context_summary = result.get("context_summary")
            if context_summary and context_summary != "null":
                node.context_summary = context_summary  # Store for _expand_exercises

            # Recurse for children
            if node.children:
                apply_to_nodes(node.children, full_num)

    apply_to_nodes(hierarchy)


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


def _find_all_markers(
    full_text: str,
    pattern: MarkerPattern,
) -> Tuple[List[Marker], List[Tuple[int, int]]]:
    """Find all exercise markers in document using LLM-provided regex patterns.

    Handles three document formats:
    1. Embedded solutions (no keyword): All text belongs to exercises
    2. Separate solution sections: Filters markers after solution keyword
    3. Appendix solutions: Filters all markers in solution section at end

    Args:
        full_text: Complete document text
        pattern: Detected marker pattern from LLM (contains regex patterns)

    Returns:
        Tuple of (List of Marker objects sorted by position, List of solution ranges)
    """
    markers: List[Marker] = []
    solution_ranges: List[Tuple[int, int]] = []  # (start, end) of solution sections

    # Use LLM-provided regex for parent markers
    # Strip any inline flags LLM may have included (e.g., (?i), (?m))
    exercise_pattern, inline_flags = _strip_inline_flags(pattern.exercise_pattern)

    # Wrap with (?:^|\n)\s* if not already anchored
    if not exercise_pattern.startswith(('(?:^', '^', '\\A')):
        exercise_pattern = rf'(?:^|\n)\s*{exercise_pattern}'

    # Combine inline flags with our defaults (IGNORECASE, MULTILINE)
    compile_flags = re.IGNORECASE | re.MULTILINE | inline_flags

    try:
        parent_regex = re.compile(exercise_pattern, compile_flags)
    except re.error as e:
        logger.error(f"Failed to compile exercise_pattern: {pattern.exercise_pattern} - {e}")
        return [], []

    # Find all parent (exercise) markers - collect raw matches first
    raw_parent_markers: List[Tuple[int, str, str, int]] = []  # (start, marker_text, number, question_start)
    for match in parent_regex.finditer(full_text):
        # Extract exercise number from first capture group
        number = match.group(1) if match.lastindex and match.lastindex >= 1 else "?"
        raw_parent_markers.append((
            match.start(),
            match.group(0).strip(),
            number,
            match.end(),
        ))

    # Find solution section positions (if solution pattern detected)
    if pattern.solution_pattern:
        sol_escaped = re.escape(pattern.solution_pattern)
        sol_regex = re.compile(
            rf'(?:^|\n)\s*({sol_escaped})',
            re.IGNORECASE | re.MULTILINE
        )

        for match in sol_regex.finditer(full_text):
            sol_start = match.start()
            sol_match_end = match.end()

            # Check if there are MORE solution keywords after this one
            next_sol_match = sol_regex.search(full_text, sol_match_end)
            has_more_solutions = next_sol_match is not None

            if has_more_solutions:
                # Format 2 (interleaved): Multiple solution sections
                # Solution ends at the next exercise marker
                sol_end = len(full_text)
                for start_pos, _, _, _ in raw_parent_markers:
                    if start_pos > sol_start:
                        sol_end = start_pos
                        break
            else:
                # Format 3 (appendix) OR last solution in Format 2
                sol_end = len(full_text)

            solution_ranges.append((sol_start, sol_end))

    def _is_in_solution_section(pos: int) -> bool:
        """Check if position falls within any solution section."""
        for start, end in solution_ranges:
            if start <= pos < end:
                return True
        return False

    # Filter parent markers - skip those in solution sections
    for start_pos, marker_text, number, question_start in raw_parent_markers:
        if _is_in_solution_section(start_pos):
            continue

        markers.append(Marker(
            marker_type=MarkerType.PARENT,
            marker_text=marker_text,
            number=number,
            start_position=start_pos,
            question_start=question_start,
        ))

    # Sort by position
    markers.sort(key=lambda m: m.start_position)

    return markers, solution_ranges


def _find_sub_markers_in_boundaries(
    full_text: str,
    sub_patterns_by_exercise: Dict[str, Optional[List[str]]],
    parent_boundaries: Dict[str, Tuple[int, int]],  # parent_number -> (start_pos, end_pos)
    solution_ranges: List[Tuple[int, int]],
) -> List[Marker]:
    """Find sub-markers within parent exercise boundaries using per-exercise patterns.

    Uses boundary-based detection with per-exercise sub_patterns.
    Only finds sub-markers within [parent_start, parent_end] for each parent,
    using patterns specific to that exercise.

    Args:
        full_text: Complete document text
        sub_patterns_by_exercise: Dict mapping parent number to list of sub_patterns (or None)
        parent_boundaries: Dict mapping parent number to (start, end) positions
        solution_ranges: List of (start, end) positions for solution sections

    Returns:
        List of sub-question Marker objects
    """
    if not sub_patterns_by_exercise or not parent_boundaries:
        return []

    def _is_in_solution_section(pos: int) -> bool:
        """Check if position falls within any solution section."""
        for start, end in solution_ranges:
            if start <= pos < end:
                return True
        return False

    markers: List[Marker] = []
    seen_positions: set[int] = set()

    # Process each parent with its specific sub_patterns
    for parent_num, (pstart, pend) in parent_boundaries.items():
        # Get patterns for this specific exercise
        sub_patterns = sub_patterns_by_exercise.get(parent_num)
        if not sub_patterns:
            # No sub-patterns for this exercise - skip
            continue

        # Extract just this parent's text for searching
        parent_text = full_text[pstart:pend]

        # Collect matches for this parent first (to handle bullet numbering)
        parent_matches = []

        for sub_pattern_raw in sub_patterns:
            # Strip inline flags and fix decimal matching issue
            sub_pattern_str, sub_inline_flags = _strip_inline_flags(sub_pattern_raw)
            sub_pattern_str = _fix_decimal_pattern(sub_pattern_str)

            # Don't anchor sub-patterns - they may appear inline after parent marker
            # e.g., "(3) (a) Question text..." where (a) follows (3) on same line

            try:
                sub_compile_flags = re.MULTILINE | sub_inline_flags
                sub_regex = re.compile(sub_pattern_str, sub_compile_flags)
            except re.error as e:
                logger.warning(f"Failed to compile sub_pattern: {sub_pattern_raw} - {e}")
                continue

            for match in sub_regex.finditer(parent_text):
                # Convert to absolute position in full_text
                relative_pos = match.start()
                start_pos = pstart + relative_pos

                # Skip if we've already added a marker at this position
                if start_pos in seen_positions:
                    continue

                # Skip sub-markers in solution sections
                if _is_in_solution_section(start_pos):
                    continue

                marker_text = match.group(0).strip()

                # Extract sub-marker value from capture groups
                if match.lastindex and match.lastindex >= 2:
                    # Combined format: first group is parent, second is sub
                    number = match.group(2)
                elif match.lastindex and match.lastindex >= 1:
                    number = match.group(1)
                else:
                    # No capture groups (bullets) - mark for auto-numbering
                    number = None

                question_start = pstart + match.end()
                parent_matches.append((start_pos, marker_text, number, question_start))
                seen_positions.add(start_pos)

        # Sort matches by position within this parent
        parent_matches.sort(key=lambda x: x[0])

        # Auto-number bullets sequentially per parent
        bullet_counter = 0
        for start_pos, marker_text, number, question_start in parent_matches:
            if number is None:
                # Bullet - assign sequential number
                bullet_counter += 1
                number = str(bullet_counter)

            markers.append(Marker(
                marker_type=MarkerType.SUB,
                marker_text=marker_text,
                number=number,
                start_position=start_pos,
                question_start=question_start,
            ))

    # Sort by position
    markers.sort(key=lambda m: m.start_position)

    return markers


def _build_hierarchy(markers: List[Marker], full_text: str) -> List[ExerciseNode]:
    """Build hierarchical exercise structure from markers.

    Args:
        markers: List of detected markers (sorted by position)
        full_text: Complete document text

    Returns:
        List of root ExerciseNode objects (parent exercises)
    """
    if not markers:
        return []

    # Pre-compute parent end positions (next parent or end of text)
    # This allows parent to capture ALL text including parts after sub-questions
    parent_end_positions: Dict[int, int] = {}
    parent_indices = [i for i, m in enumerate(markers) if m.marker_type == MarkerType.PARENT]
    for idx, pi in enumerate(parent_indices):
        if idx + 1 < len(parent_indices):
            parent_end_positions[pi] = markers[parent_indices[idx + 1]].start_position
        else:
            parent_end_positions[pi] = len(full_text)

    roots: List[ExerciseNode] = []
    current_parent: Optional[ExerciseNode] = None
    current_parent_idx: int = -1
    highest_sub_value: int = 0  # Track highest sub-marker value (number or letter ord)
    in_restart_sequence: bool = False  # Skip all subs after restart detected

    for i, marker in enumerate(markers):
        # Find end position
        if marker.marker_type == MarkerType.PARENT:
            # Parent: capture ALL text up to next parent (includes text after subs)
            end_pos = parent_end_positions.get(i, len(full_text))
            current_parent_idx = i
        else:
            # Sub-question: capture text up to next marker
            if i + 1 < len(markers):
                end_pos = markers[i + 1].start_position
            else:
                end_pos = len(full_text)

        # Extract text for this marker
        text_content = full_text[marker.question_start:end_pos].strip()

        node = ExerciseNode(
            marker=marker,
            context="",  # Will be set for children
            question_text=text_content,
        )

        if marker.marker_type == MarkerType.PARENT:
            # This is a parent exercise
            roots.append(node)
            current_parent = node
            highest_sub_value = 0  # Reset for new parent
            in_restart_sequence = False  # Reset restart flag
        else:
            # This is a sub-question
            if current_parent is not None:
                # Skip sub-markers in restart sequence (e.g., page headers)
                if in_restart_sequence:
                    continue

                # Detect restart: if sequence value drops (4→1 or d→a)
                # Handle multi-char markers (roman numerals, multi-digit numbers)
                sub_value = None
                try:
                    sub_value = int(marker.number)
                except ValueError:
                    if marker.number.isalpha():
                        if _is_roman_numeral(marker.number):
                            # Roman numerals: i=1, ii=2, iii=3, iv=4, etc.
                            sub_value = _roman_to_int(marker.number)
                        else:
                            # Regular letters: sum ordinals so "aa" > "a"
                            sub_value = sum(ord(c.lower()) for c in marker.number)

                # Restart detected if value drops significantly (not sequential)
                # Only check if we successfully computed a value
                if sub_value is not None and highest_sub_value > 0 and sub_value < highest_sub_value:
                    # Sequence restarted - skip this and all subsequent for this parent
                    in_restart_sequence = True
                    continue
                if sub_value is not None:
                    highest_sub_value = max(highest_sub_value, sub_value)
                node.parent = current_parent
                # Context is the parent's intro text (before first sub)
                if not current_parent.children:
                    # First child - parent's question_text is the context
                    node.context = current_parent.question_text
                else:
                    # Use same context as siblings
                    node.context = current_parent.children[0].context
                current_parent.children.append(node)
            else:
                # Orphan sub-question (no parent found) - treat as root
                roots.append(node)

    return roots


def _expand_exercises(
    hierarchy: List[ExerciseNode],
    source_pdf: str,
    course_code: str,
    page_lookup: Dict[int, int],  # char_position -> page_number
) -> List[Exercise]:
    """Expand hierarchical structure to flat list with context.

    Args:
        hierarchy: List of root ExerciseNode objects
        source_pdf: Source PDF filename
        course_code: Course code for ID generation
        page_lookup: Mapping of character positions to page numbers

    Returns:
        List of Exercise objects ready for analysis
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

    for parent in hierarchy:
        parent_num = parent.marker.number
        page_num = get_page_number(parent.marker.start_position)

        if parent.children:
            # Parent with children - DON'T emit parent as separate exercise
            # Use context_summary from second-pass (null means context not needed)
            parent_ctx = getattr(parent, "context_summary", None)

            for child in parent.children:
                sub_text = child.question_text.strip()

                # Skip empty sub-questions (marker detected but no actual content)
                if not sub_text:
                    continue

                child_page_num = get_page_number(child.marker.start_position)
                ex_num = f"{parent_num}.{child.marker.number}"
                exercise_id = _generate_exercise_id(
                    course_code, source_pdf, ex_num, child.marker.start_position
                )

                exercises.append(Exercise(
                    id=exercise_id,
                    text=sub_text,
                    page_number=child_page_num,
                    exercise_number=ex_num,
                    has_images=False,  # Will be enriched later
                    image_data=[],
                    has_latex=False,
                    latex_content=None,
                    source_pdf=source_pdf,
                    parent_exercise_number=parent_num,
                    sub_question_marker=child.marker.number,
                    is_sub_question=True,
                    parent_context=parent_ctx,
                ))
        else:
            # Standalone exercise - emit it directly
            exercise_id = _generate_exercise_id(
                course_code, source_pdf, parent_num, parent.marker.start_position
            )
            exercises.append(Exercise(
                id=exercise_id,
                text=parent.question_text,
                page_number=page_num,
                exercise_number=parent_num,
                has_images=False,
                image_data=[],
                has_latex=False,
                latex_content=None,
                source_pdf=source_pdf,
            ))

    return exercises


def _extract_solutions(
    exercises: List[Exercise],
    full_text: str,
    solution_ranges: List[Tuple[int, int]],
    pattern: MarkerPattern,
) -> List[Exercise]:
    """Extract solution text and attach to exercises.

    Handles:
    - Format 2 (interleaved): Multiple solution sections, each follows an exercise
    - Format 3 (appendix): Single solution section at end with all solutions

    Args:
        exercises: List of exercises (will be modified in place)
        full_text: Complete document text
        solution_ranges: List of (start, end) tuples for solution sections
        pattern: Marker pattern with exercise_pattern and solution_pattern

    Returns:
        Exercises with solution field populated
    """
    if not solution_ranges or not pattern.solution_pattern:
        return exercises

    # Determine format based on number of solution sections vs exercises
    # If multiple solution sections → Format 2 (interleaved)
    # If single solution section → Format 3 (appendix)
    is_appendix = len(solution_ranges) == 1

    if is_appendix:
        # Format 3: Single appendix with all solutions
        _extract_appendix_solutions(exercises, full_text, solution_ranges[0], pattern)
    else:
        # Format 2: Interleaved solutions
        _extract_interleaved_solutions(exercises, full_text, solution_ranges, pattern)

    return exercises


def _extract_interleaved_solutions(
    exercises: List[Exercise],
    full_text: str,
    solution_ranges: List[Tuple[int, int]],
    pattern: MarkerPattern,
) -> None:
    """Extract solutions for Format 2 (interleaved).

    Each solution section follows an exercise and contains answers for that exercise.
    """
    # Build regex for sub-markers in solutions using LLM-provided pattern
    sub_regex = None
    if pattern.sub_pattern:
        sub_pattern_str, sub_inline_flags = _strip_inline_flags(pattern.sub_pattern)
        if not sub_pattern_str.startswith(('(?:^', '^', '\\A')):
            sub_pattern_str = rf'(?:^|\n)\s*{sub_pattern_str}'
        try:
            sub_regex = re.compile(sub_pattern_str, re.MULTILINE | sub_inline_flags)
        except re.error:
            sub_regex = None

    # Group exercises by parent number
    exercises_by_parent: Dict[str, List[Exercise]] = {}
    for ex in exercises:
        parent_num = ex.parent_exercise_number or ex.exercise_number
        if parent_num:
            if parent_num not in exercises_by_parent:
                exercises_by_parent[parent_num] = []
            exercises_by_parent[parent_num].append(ex)

    # Match solution sections to exercises by position
    # In Format 2, solution N follows exercise N
    for i, (sol_start, sol_end) in enumerate(solution_ranges):
        # Find which exercise this solution section corresponds to
        # Look for the exercise that ends just before this solution
        parent_num = str(i + 1)  # Assume solution sections are in order

        if parent_num not in exercises_by_parent:
            continue

        solution_text = full_text[sol_start:sol_end].strip()

        # If we have sub-markers, extract solutions for each sub-question
        if sub_regex:
            # Find all sub-markers in this solution section
            sub_solutions: Dict[str, str] = {}
            matches = list(sub_regex.finditer(solution_text))

            for j, match in enumerate(matches):
                sub_marker = match.group(1)
                start = match.end()
                # End at next sub-marker or end of section
                end = matches[j + 1].start() if j + 1 < len(matches) else len(solution_text)
                sub_solutions[sub_marker] = solution_text[start:end].strip()

            # Match to exercises
            for ex in exercises_by_parent[parent_num]:
                if ex.sub_question_marker and ex.sub_question_marker in sub_solutions:
                    ex.solution = sub_solutions[ex.sub_question_marker]
                elif not ex.is_sub_question:
                    # Parent exercise without sub-questions gets full solution
                    ex.solution = solution_text
        else:
            # No sub-markers, full solution goes to parent exercise
            for ex in exercises_by_parent[parent_num]:
                if not ex.is_sub_question:
                    ex.solution = solution_text


def _extract_appendix_solutions(
    exercises: List[Exercise],
    full_text: str,
    solution_range: Tuple[int, int],
    pattern: MarkerPattern,
) -> None:
    """Extract solutions for Format 3 (appendix).

    Single solution section at end contains all solutions, organized by exercise number.
    """
    sol_start, sol_end = solution_range
    solution_text = full_text[sol_start:sol_end]

    # Build regex to find exercise markers in solution section using LLM pattern
    exercise_pattern, ex_inline_flags = _strip_inline_flags(pattern.exercise_pattern)
    if not exercise_pattern.startswith(('(?:^', '^', '\\A')):
        exercise_pattern = rf'(?:^|\n)\s*{exercise_pattern}'
    try:
        exercise_regex = re.compile(exercise_pattern, re.IGNORECASE | re.MULTILINE | ex_inline_flags)
    except re.error:
        return  # Can't extract without valid exercise pattern

    # Build regex for sub-markers using LLM pattern
    sub_regex = None
    if pattern.sub_pattern:
        sub_pattern_str, sub_inline_flags = _strip_inline_flags(pattern.sub_pattern)
        if not sub_pattern_str.startswith(('(?:^', '^', '\\A')):
            sub_pattern_str = rf'(?:^|\n)\s*{sub_pattern_str}'
        try:
            sub_regex = re.compile(sub_pattern_str, re.MULTILINE | sub_inline_flags)
        except re.error:
            sub_regex = None

    # Find all exercise markers in solution section
    exercise_matches = list(exercise_regex.finditer(solution_text))

    # Build solution map: {exercise_number: {sub_marker: solution_text}}
    solution_map: Dict[str, Dict[str, str]] = {}

    for i, match in enumerate(exercise_matches):
        # Exercise number is in the first capture group of LLM pattern
        ex_num = match.group(1) if match.lastindex and match.lastindex >= 1 else str(i + 1)
        ex_start = match.end()
        # End at next exercise marker or end of solution section
        ex_end = exercise_matches[i + 1].start() if i + 1 < len(exercise_matches) else len(solution_text)

        ex_solution_text = solution_text[ex_start:ex_end].strip()
        solution_map[ex_num] = {}

        if sub_regex:
            # Find sub-markers within this exercise's solution
            sub_matches = list(sub_regex.finditer(ex_solution_text))
            for j, sub_match in enumerate(sub_matches):
                sub_marker = sub_match.group(1)
                sub_start = sub_match.end()
                sub_end = sub_matches[j + 1].start() if j + 1 < len(sub_matches) else len(ex_solution_text)
                solution_map[ex_num][sub_marker] = ex_solution_text[sub_start:sub_end].strip()

            # If no sub-markers found, store full text
            if not sub_matches:
                solution_map[ex_num]["_full"] = ex_solution_text
        else:
            solution_map[ex_num]["_full"] = ex_solution_text

    # Match solutions to exercises
    for ex in exercises:
        parent_num = ex.parent_exercise_number or ex.exercise_number
        if not parent_num or parent_num not in solution_map:
            continue

        ex_solutions = solution_map[parent_num]

        if ex.sub_question_marker and ex.sub_question_marker in ex_solutions:
            ex.solution = ex_solutions[ex.sub_question_marker]
        elif "_full" in ex_solutions and not ex.is_sub_question:
            ex.solution = ex_solutions["_full"]


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
        """Split PDF using LLM-based pattern detection with sub-question context.

        This method uses LLM to detect the exercise marker pattern, then:
        1. Finds all markers in the full document
        2. Builds a hierarchical structure (parent → children)
        3. (Optional) Second-pass LLM for end markers and context summaries
        4. Expands to flat list with context prepended to sub-questions

        Args:
            pdf_content: Extracted PDF content
            course_code: Course code for ID generation
            llm_manager: LLM manager for pattern detection (first pass)
            second_pass_llm: Optional LLM for second-pass (end markers, context summaries).
                             If not provided, uses llm_manager for all calls.
                             DeepSeek produces equivalent results to Sonnet at 20x lower cost.

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
        detection = _detect_pattern_with_llm(full_text[:30000], llm_manager)

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

        # Step 3: Find markers based on detection mode
        markers: List[Marker] = []
        solution_ranges: List[Tuple[int, int]] = []
        pattern: Optional[MarkerPattern] = None

        if detection.explicit_exercises:
            # Mode 2a: Explicit exercises with end markers (new format)
            logger.info(
                f"Using explicit exercises: {len(detection.explicit_exercises)} exercises with end markers"
            )
            boundaries = _find_explicit_exercises(full_text, detection.explicit_exercises)

            # Get explicit sub-questions if second_pass_llm available
            if second_pass_llm:
                explicit_subs = _get_explicit_sub_questions(boundaries, full_text, second_pass_llm)
            else:
                explicit_subs = {}

            exercises = _create_exercises_from_explicit_boundaries(
                boundaries, explicit_subs, full_text, pdf_content, course_code, page_lookup
            )
            # Enrich with page data and return
            exercises = self._enrich_with_page_data(exercises, pdf_content)
            logger.info(f"Explicit mode produced {len(exercises)} exercises")
            return exercises

        elif detection.explicit_markers:
            # Mode 2b: Legacy explicit markers (backward compat)
            logger.info(
                f"Using explicit markers: {len(detection.explicit_markers)} markers"
            )
            markers = _find_explicit_markers(full_text, detection.explicit_markers)
        elif detection.pattern:
            # Mode 1: Pattern-based detection with boundary-based sub detection
            pattern = detection.pattern
            logger.info(
                f"Pattern detected: exercise='{pattern.exercise_pattern}'"
                + (f", solution='{pattern.solution_pattern}'" if pattern.solution_pattern else "")
            )

            # Step 3a: Find parent markers only
            parent_markers, solution_ranges = _find_all_markers(full_text, pattern)
            if solution_ranges:
                logger.info(f"Detected {len(solution_ranges)} solution sections")

            if not parent_markers:
                logger.warning("Pattern detected but no parent markers found, falling back")
                return _split_unstructured(pdf_content, course_code)

            logger.info(f"Found {len(parent_markers)} parent markers")

            # Step 3b: Get parent end positions (Call 2: boundary_llm)
            if second_pass_llm:
                logger.info("Getting parent end markers...")
                parent_end_positions = _get_parent_end_markers(
                    parent_markers, full_text, second_pass_llm
                )

                # Build boundaries dict for sub detection
                # Use question_start (after marker text) to exclude parent marker itself
                parent_boundaries: Dict[str, Tuple[int, int]] = {}
                for marker in parent_markers:
                    start = marker.question_start  # Start AFTER parent marker text
                    end = parent_end_positions.get(marker.number, len(full_text))
                    parent_boundaries[marker.number] = (start, end)

                # Step 3c: Get per-exercise sub_patterns (Call 3: sub_pattern_llm)
                sub_patterns_by_exercise = _get_per_exercise_sub_patterns(
                    parent_markers, full_text, parent_boundaries, second_pass_llm
                )

                # Step 3d: Find sub markers within boundaries using per-exercise patterns
                if sub_patterns_by_exercise:
                    logger.info("Finding sub-markers within parent boundaries...")
                    sub_markers = _find_sub_markers_in_boundaries(
                        full_text, sub_patterns_by_exercise, parent_boundaries, solution_ranges
                    )
                    logger.info(f"Found {len(sub_markers)} sub-markers")

                    # Combine and sort all markers
                    markers = sorted(
                        parent_markers + sub_markers,
                        key=lambda m: m.start_position
                    )
                else:
                    # No sub-patterns for any exercise
                    logger.info("No sub-patterns detected for any exercise")
                    markers = parent_markers
            else:
                # No second pass LLM - just use parent markers
                markers = parent_markers

        if not markers:
            logger.warning("Pattern detected but no markers found, falling back")
            return _split_unstructured(pdf_content, course_code)

        logger.info(f"Found {len(markers)} total markers")

        # Step 4: Build hierarchy
        hierarchy = _build_hierarchy(markers, full_text)
        logger.info(f"Built hierarchy with {len(hierarchy)} root exercises")

        # Step 5: Second pass for sub end markers and context summaries (Sonnet call 2)
        if second_pass_llm:
            second_pass_results = _get_second_pass_results(hierarchy, second_pass_llm)
            if second_pass_results:
                _apply_second_pass_results(hierarchy, second_pass_results)
                logger.info(f"Applied second-pass results to {len(second_pass_results)} exercises")
            else:
                logger.warning("Second-pass returned no results")

        # Step 6: Expand to flat list with context
        exercises = _expand_exercises(
            hierarchy,
            pdf_content.file_path.name,
            course_code,
            page_lookup,
        )

        # Step 6: Extract solutions (if solution sections detected)
        if solution_ranges and pattern:
            exercises = _extract_solutions(exercises, full_text, solution_ranges, pattern)
            solutions_found = sum(1 for ex in exercises if ex.solution)
            logger.info(f"Extracted solutions for {solutions_found}/{len(exercises)} exercises")

        # Step 7: Enrich with image/latex data from pages
        exercises = self._enrich_with_page_data(exercises, pdf_content)

        logger.info(f"Smart split produced {len(exercises)} exercises")
        return exercises

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
