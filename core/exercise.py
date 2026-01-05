"""
Exercise extraction for Qupled.
Uses VLM-based extraction for accurate exercise detection with visual context.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from core.exercise_scanner import extract_exercises, get_pdf_page_count, render_page_to_image


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
    # Sub-question support
    parent_exercise_number: Optional[str] = None  # "2" if this is "2.1"
    sub_question_marker: Optional[str] = None  # "1", "2", "3", etc.
    is_sub_question: bool = False
    # Exercise context (LLM-generated)
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
        lines = self.text.strip().split("\n")

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip lines with form fields (underscores, dots as blanks)
            if "_____" in line or "....." in line or "___" in line:
                continue

            # Skip very short lines (likely headers or labels)
            if len(line) < 20:
                continue

            # Skip lines that are mostly uppercase and short (likely headers)
            if len(line) < 60 and line.upper() == line:
                continue

            # Skip lines that start with "word + number" pattern (exercise markers)
            if re.match(r"^[A-Za-z\u00C0-\u024F]+\s+\d+\s*$", line):
                continue

            # Found a good line - clean it up
            cleaned = re.sub(r"^[A-Za-z\u00C0-\u024F]+\s+\d+\s*", "", line).strip()
            if not cleaned or len(cleaned) < 15:
                cleaned = line

            # Remove leading number patterns like "1.", "1)"
            cleaned = re.sub(r"^\d+[\.\)\:]\s*", "", cleaned).strip()

            # Get first sentence or truncate
            if "." in cleaned[:120] and cleaned.index(".") > 20:
                preview = cleaned[: cleaned.index(".") + 1]
            else:
                preview = cleaned[:max_length]

            # Truncate if needed
            if len(preview) > max_length:
                preview = preview[:max_length].rsplit(" ", 1)[0] + "..."
            elif len(cleaned) > len(preview):
                preview = preview.rstrip(".") + "..."

            return preview

        # Fallback - no good lines found
        text = self.text.strip()
        text = re.sub(r"[_]{3,}", "", text)
        text = re.sub(r"\s+", " ", text)
        preview = text[:max_length].strip()
        if len(text) > max_length:
            preview = preview.rsplit(" ", 1)[0] + "..."
        return preview if preview else f"#{self.exercise_number or '?'}"


def _generate_exercise_id(
    course_code: str,
    source_pdf: str,
    exercise_number: str,
    char_position: int,
) -> str:
    """Generate a unique exercise ID."""
    components = f"{course_code}_{source_pdf}_{exercise_number}_{char_position}"
    hash_obj = hashlib.md5(components.encode())
    short_hash = hash_obj.hexdigest()[:12]
    course_abbrev = course_code.lower().replace("b", "").replace("0", "")[:6]
    ex_num_clean = exercise_number.replace(".", "_")
    return f"{course_abbrev}_{ex_num_clean}_{short_hash}"


class ExerciseExtractor:
    """VLM-based exercise extractor for accurate extraction with visual context."""

    def __init__(self):
        """Initialize exercise extractor."""
        pass

    def extract(
        self,
        file_path: Path,
        course_code: str,
    ) -> List[Exercise]:
        """Unified extraction pipeline: VLM for OCR + splitting, DeepSeek for context.

        Handles both PDFs and images:
        - VLM: OCR, exercise boundaries, hierarchical structure
        - DeepSeek: context summaries for parents/standalone

        Args:
            file_path: Path to PDF or image file
            course_code: Course code for ID generation

        Returns:
            List of Exercise objects

        Raises:
            FileNotFoundError: If file not found
            VLMExtractionError: If VLM extraction fails
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check if image or PDF
        suffix = file_path.suffix.lower()
        is_image = suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")

        # Get page images
        if is_image:
            # Single image file
            page_images = [file_path.read_bytes()]
            total_pages = 1
        else:
            # PDF: render all pages
            total_pages = get_pdf_page_count(file_path)
            page_images = []
            for page_num in range(1, total_pages + 1):
                img_bytes = render_page_to_image(file_path, page_num, dpi=150)
                page_images.append(img_bytes)

        # Extract exercises using VLM
        vlm_exercises = extract_exercises(page_images)

        # Convert to Exercise objects
        exercises = []
        for i, ex in enumerate(vlm_exercises):
            ex_num = ex["exercise_number"]
            page_num = ex.get("page_number", 1)

            # Determine if sub-question
            is_sub = "." in ex_num
            parent_num = None
            sub_marker = None
            if is_sub:
                parent_num = ex_num.rsplit(".", 1)[0]
                sub_marker = ex_num.rsplit(".", 1)[1]

            exercise_id = _generate_exercise_id(
                course_code, file_path.name, ex_num, i
            )

            exercises.append(
                Exercise(
                    id=exercise_id,
                    text=ex["text"],
                    page_number=page_num,
                    exercise_number=ex_num,
                    has_images=bool(ex.get("image_context")),
                    image_data=[],  # VLM describes images, doesn't extract bytes
                    has_latex="$" in ex["text"],
                    latex_content=None,
                    source_pdf=file_path.name,
                    parent_exercise_number=parent_num,
                    sub_question_marker=sub_marker,
                    is_sub_question=is_sub,
                    exercise_context=ex.get("exercise_context"),
                )
            )

        return exercises

    def validate_exercise(self, exercise: Exercise) -> bool:
        """Validate if an exercise has content.

        Args:
            exercise: Exercise to validate

        Returns:
            True if exercise is valid (non-empty)
        """
        text = exercise.text.strip()
        if not text:
            return False
        # Sub-questions can be short (e.g., "(a) RAID 0")
        # Main exercises need more content
        min_length = 5 if exercise.is_sub_question else 15
        if len(text) < min_length:
            return False
        return True


# Backward compatibility alias
ExerciseSplitter = ExerciseExtractor
