"""
Note splitting for Qupled.
Splits lecture notes PDFs into topic sections based on headers/chapters.

NoteSplitter detects document structure: headers, chapters, numbered sections.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from core.note_scanner import scan_notes


@dataclass
class NoteSection:
    """Represents a single section extracted from lecture notes."""

    id: str
    title: Optional[str]
    content: str
    page_number: int
    end_page: Optional[int]  # For multi-page sections
    source_pdf: str
    section_level: int  # 1 = chapter, 2 = section, 3 = subsection
    has_images: bool
    image_paths: List[str]

    def get_preview(self, max_length: int = 200) -> str:
        """Get a preview of the section content."""
        text = self.content.strip()
        if len(text) <= max_length:
            return text
        return text[:max_length].rsplit(" ", 1)[0] + "..."


class NoteSplitter:
    """Split lecture notes into topic sections."""

    # Header patterns (ordered by priority/level)
    HEADER_PATTERNS = [
        # Markdown-style headers
        (r"^#{1}\s+(.+)$", 1),  # # Chapter
        (r"^#{2}\s+(.+)$", 2),  # ## Section
        (r"^#{3}\s+(.+)$", 3),  # ### Subsection
        # Numbered chapters/sections
        (r"^(\d+)\.\s+([A-Z].+)$", 1),  # 1. Chapter Title
        (r"^(\d+\.\d+)\s+(.+)$", 2),  # 1.1 Section
        (r"^(\d+\.\d+\.\d+)\s+(.+)$", 3),  # 1.1.1 Subsection
        # Parenthesized numbers
        (r"^(\d+)\)\s+([A-Z].+)$", 1),  # 1) Chapter Title
        # Roman numerals
        (r"^([IVXLCDM]+)\.\s+(.+)$", 1),  # I. Chapter
        (r"^([ivxlcdm]+)\.\s+(.+)$", 2),  # i. Section
        # ALL CAPS titles (likely chapters)
        (r"^([A-Z][A-Z\s]{10,})$", 1),  # CHAPTER TITLE
        # Bold markers (common in PDF extraction)
        (r"^\*\*(.+)\*\*$", 2),  # **Section Title**
        # Capitolo/Chapter explicit markers
        (r"^(?:Capitolo|Chapter|Chapitre|Kapitel)\s+(\d+)[\.:]\s*(.+)?$", 1),
        # Italian-style headers (line ending with colon, 3-50 chars)
        (r"^([A-Z][a-zàèéìòùA-Z\s]{2,48}):[ \t]*$", 2),
        # Italian headers with articles (Il, La, etc.)
        (r"^((?:Il|La|Lo|I|Le|Gli|Un|Una|Uno)\s+[a-zàèéìòùA-Z][a-zàèéìòùA-Z\s]{2,45}):[ \t]*$", 2),
    ]

    # Minimum content length to consider a section valid
    MIN_SECTION_LENGTH = 100

    def __init__(self):
        """Initialize note splitter."""
        self.compiled_patterns = [
            (re.compile(pattern, re.MULTILINE), level) for pattern, level in self.HEADER_PATTERNS
        ]
        self.section_counter = 0

    def split_notes(self, text: str, source_pdf: str = "notes.pdf") -> List[NoteSection]:
        """Split note text into sections by detecting headers.

        Args:
            text: Full text content of the notes
            source_pdf: Source PDF filename

        Returns:
            List of NoteSection objects
        """
        self.section_counter = 0
        sections = []

        # Find all headers with their positions
        headers = self._find_headers(text)

        if not headers:
            # No headers found - treat entire document as single section
            return [
                self._create_section(
                    title=None,
                    content=text,
                    page_number=1,
                    source_pdf=source_pdf,
                    section_level=1,
                )
            ]

        # Split text at headers
        for i, (pos, title, level) in enumerate(headers):
            # Find end position (start of next header or end of text)
            if i + 1 < len(headers):
                end_pos = headers[i + 1][0]
            else:
                end_pos = len(text)

            content = text[pos:end_pos].strip()

            # Remove the header line from content
            content_lines = content.split("\n", 1)
            if len(content_lines) > 1:
                content = content_lines[1].strip()
            else:
                content = ""

            # Skip sections that are too short
            if len(content) < self.MIN_SECTION_LENGTH:
                continue

            sections.append(
                self._create_section(
                    title=title,
                    content=content,
                    page_number=1,  # VLM doesn't track page numbers
                    source_pdf=source_pdf,
                    section_level=level,
                )
            )

        return sections

    def split_pdf(self, pdf_path: Path) -> List[NoteSection]:
        """Split a PDF into sections using VLM-based OCR.

        This is the primary method for processing notes.
        Uses note_scanner for OCR, then splits on headers.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of NoteSection objects
        """
        # Use VLM for OCR
        full_text = scan_notes(pdf_path)

        # Split into sections
        return self.split_notes(text=full_text, source_pdf=pdf_path.name)

    def _find_headers(self, text: str) -> List[Tuple[int, str, int]]:
        """Find all headers in the text.

        Returns:
            List of (position, title, level) tuples, sorted by position
        """
        headers = []

        for pattern, level in self.compiled_patterns:
            for match in pattern.finditer(text):
                # Extract title from match
                groups = match.groups()
                if len(groups) == 1:
                    title = groups[0].strip()
                elif len(groups) == 2:
                    # Pattern like "1. Title" - combine number and title
                    title = f"{groups[0]} {groups[1]}".strip() if groups[1] else groups[0].strip()
                else:
                    title = match.group(0).strip()

                # Clean up title
                title = self._clean_title(title)

                if title and len(title) > 2:  # Skip very short titles
                    headers.append((match.start(), title, level))

        # Sort by position and remove duplicates (same position)
        headers.sort(key=lambda x: x[0])
        unique_headers = []
        seen_positions = set()
        for pos, title, level in headers:
            # Allow headers within 5 chars of each other (same line)
            rounded_pos = pos // 10 * 10
            if rounded_pos not in seen_positions:
                seen_positions.add(rounded_pos)
                unique_headers.append((pos, title, level))

        return unique_headers

    def _clean_title(self, title: str) -> str:
        """Clean up extracted title."""
        # Remove markdown formatting
        title = re.sub(r"^#+\s*", "", title)
        title = re.sub(r"\*+", "", title)

        # Remove trailing colons
        title = title.rstrip(":")

        # Normalize whitespace
        title = " ".join(title.split())

        return title.strip()

    def _create_section(
        self,
        title: Optional[str],
        content: str,
        page_number: int,
        source_pdf: str,
        section_level: int,
        end_page: Optional[int] = None,
        has_images: bool = False,
        image_paths: Optional[List[str]] = None,
    ) -> NoteSection:
        """Create a NoteSection with generated ID."""
        self.section_counter += 1

        # Generate unique ID
        content_hash = hashlib.md5(content[:500].encode()).hexdigest()[:8]
        section_id = f"note_{self.section_counter}_{content_hash}"

        return NoteSection(
            id=section_id,
            title=title,
            content=content,
            page_number=page_number,
            end_page=end_page,
            source_pdf=source_pdf,
            section_level=section_level,
            has_images=has_images,
            image_paths=image_paths or [],
        )
