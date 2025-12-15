"""
Smart exercise splitter for Examina.
Combines pattern-based and LLM-based splitting for unstructured materials.
"""

import json
from typing import List, Optional, Tuple
from dataclasses import dataclass
import hashlib

from core.exercise_splitter import ExerciseSplitter, Exercise
from core.pdf_processor import PDFContent, PDFPage
from models.llm_manager import LLMManager
from config import Config


@dataclass
class LearningMaterial:
    """Represents a learning material (theory, worked example, reference)."""

    id: str
    title: Optional[str]
    content: str
    material_type: str  # 'theory', 'worked_example', 'reference'
    page_number: int
    has_images: bool
    image_data: List[bytes]
    has_latex: bool
    latex_content: Optional[str]
    source_pdf: str


@dataclass
class DetectedContent:
    """Content detected by LLM with metadata."""

    start_char: int
    end_char: int
    content_type: str  # 'theory', 'worked_example', 'practice_exercise'
    title: Optional[str]
    confidence: float
    has_solution_inline: bool


@dataclass
class SplitResult:
    """Result of smart splitting operation."""

    exercises: List[Exercise]
    learning_materials: List[LearningMaterial]
    pattern_based_count: int
    llm_based_count: int
    theory_count: int
    worked_example_count: int
    total_pages: int
    llm_pages_processed: int
    total_cost_estimate: float


class SmartExerciseSplitter:
    """
    Hybrid exercise splitter combining pattern-based and LLM-based detection.

    Strategy:
    1. Try fast pattern-based splitting first (free, instant)
    2. For pages with no exercises found, use LLM detection (costs tokens)
    3. Respects cost controls (max pages, caching)
    """

    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        enable_smart_detection: bool = True,
        notes_mode: bool = False,
    ):
        """
        Initialize smart splitter.

        Args:
            llm_manager: LLM manager for smart detection (required if enable_smart_detection=True)
            enable_smart_detection: Enable LLM-based detection for unstructured materials
            notes_mode: If True, process ALL pages with LLM (not just pages without pattern-based exercises)
        """
        self.llm = llm_manager
        self.pattern_splitter = ExerciseSplitter()
        self.enable_smart = enable_smart_detection and llm_manager is not None
        self.notes_mode = notes_mode

        # Load config
        self.confidence_threshold = Config.SMART_SPLIT_CONFIDENCE_THRESHOLD
        self.max_pages = Config.SMART_SPLIT_MAX_PAGES
        self.cache_enabled = Config.SMART_SPLIT_CACHE_ENABLED

        # Cache for LLM results (page_hash -> detected exercises)
        self._detection_cache = {}

        if self.enable_smart and not self.llm:
            raise ValueError("LLM manager required when enable_smart_detection=True")

    def split_pdf_content(self, pdf_content: PDFContent, course_code: str) -> SplitResult:
        """
        Split PDF content into exercises and learning materials using hybrid approach.

        Args:
            pdf_content: Extracted PDF content
            course_code: Course code for ID generation

        Returns:
            SplitResult with exercises, learning materials, and metadata
        """
        all_exercises = []
        all_materials = []
        pattern_count = 0
        llm_count = 0
        theory_count = 0
        worked_example_count = 0
        llm_pages_processed = 0

        # Cost control: limit pages if smart detection enabled
        pages_to_process = pdf_content.pages
        if self.enable_smart and len(pages_to_process) > self.max_pages:
            print(
                f"⚠️  Warning: PDF has {len(pages_to_process)} pages. "
                f"Processing only first {self.max_pages} pages to control costs."
            )
            print(f"   Increase with: export EXAMINA_SMART_SPLIT_MAX_PAGES={len(pages_to_process)}")
            pages_to_process = pages_to_process[: self.max_pages]

        # Phase 1: Pattern-based splitting (fast, free)
        pattern_exercises = self.pattern_splitter.split_pdf_content(pdf_content, course_code)

        # Track which pages had exercises found
        pages_with_exercises = set()
        for ex in pattern_exercises:
            pages_with_exercises.add(ex.page_number)

        pattern_count = len(pattern_exercises)
        all_exercises.extend(pattern_exercises)

        # Phase 2: LLM-based detection
        if self.enable_smart:
            for page in pages_to_process:
                # In notes mode, process ALL pages to detect theory/worked examples
                # In exams mode, only process pages without pattern-based exercises
                if not self.notes_mode and page.page_number in pages_with_exercises:
                    continue  # Already found exercises with patterns (exams mode only)

                if not page.text.strip():
                    continue  # Empty page

                # Skip instruction-only pages
                if self.pattern_splitter._is_instruction_page(page.text):
                    continue

                # Try LLM detection - returns both exercises and materials
                detected_exercises, detected_materials = self._detect_exercises_with_llm(
                    page, pdf_content.file_path.name, course_code
                )

                if detected_exercises or detected_materials:
                    llm_pages_processed += 1
                    llm_count += len(detected_exercises)
                    all_exercises.extend(detected_exercises)

                    # Count materials by type
                    for material in detected_materials:
                        if material.material_type == "theory":
                            theory_count += 1
                        elif material.material_type == "worked_example":
                            worked_example_count += 1

                    all_materials.extend(detected_materials)

        # Estimate cost (rough approximation)
        tokens_per_page = 1500  # Average page length
        cost_per_1k_tokens = 0.0002  # Approximate for Groq/Anthropic
        estimated_cost = (llm_pages_processed * tokens_per_page / 1000) * cost_per_1k_tokens

        return SplitResult(
            exercises=all_exercises,
            learning_materials=all_materials,
            pattern_based_count=pattern_count,
            llm_based_count=llm_count,
            theory_count=theory_count,
            worked_example_count=worked_example_count,
            total_pages=len(pdf_content.pages),
            llm_pages_processed=llm_pages_processed,
            total_cost_estimate=estimated_cost,
        )

    def _detect_exercises_with_llm(
        self, page: PDFPage, source_pdf: str, course_code: str
    ) -> Tuple[List[Exercise], List[LearningMaterial]]:
        """
        Use LLM to detect content boundaries in unstructured text.

        Args:
            page: PDF page to analyze
            source_pdf: Source PDF filename
            course_code: Course code

        Returns:
            Tuple of (exercises_list, materials_list)
        """
        # Check cache first
        if self.cache_enabled:
            page_hash = self._hash_page(page.text)
            if page_hash in self._detection_cache:
                cached = self._detection_cache[page_hash]
                # Reconstruct Exercise and Material objects with proper IDs
                exercises = self._create_exercises_from_detected(
                    cached, page, source_pdf, course_code
                )
                materials = self._create_materials_from_detected(
                    cached, page, source_pdf, course_code
                )
                return exercises, materials

        # Build prompt
        prompt = self._build_detection_prompt(page.text)

        # Call LLM through manager (provider-agnostic)
        try:
            response = self.llm.generate(prompt=prompt, temperature=0.0, max_tokens=1000)

            # Check if LLM call was successful
            if not response.success:
                print(f"⚠️  LLM call failed for page {page.page_number}: {response.error}")
                return [], []

            if not response.text or not response.text.strip():
                print(f"⚠️  LLM returned empty response for page {page.page_number}")
                return [], []

            # Parse JSON response
            detected_content = self._parse_detection_response(response.text)

            # Cache result
            if self.cache_enabled:
                page_hash = self._hash_page(page.text)
                self._detection_cache[page_hash] = detected_content

            # Convert to Exercise and LearningMaterial objects
            exercises = self._create_exercises_from_detected(
                detected_content, page, source_pdf, course_code
            )
            materials = self._create_materials_from_detected(
                detected_content, page, source_pdf, course_code
            )

            return exercises, materials

        except Exception as e:
            # Graceful degradation: if LLM fails, return empty lists
            print(f"⚠️  LLM detection failed for page {page.page_number}: {e}")
            return [], []

    def _build_detection_prompt(self, page_text: str) -> str:
        """
        Build generic prompt for content classification.

        IMPORTANT: This prompt is GENERIC and works for ANY:
        - Subject (CS, Math, Physics, Chemistry, Biology, etc.)
        - Language (English, Italian, Spanish, French, etc.)
        - Format (structured, unstructured, mixed)

        Args:
            page_text: Text to analyze

        Returns:
            Prompt string
        """
        # Truncate if too long (to avoid token limits)
        if len(page_text) > 4000:
            page_text = page_text[:4000] + "...[truncated]"

        prompt = f"""Analyze this educational text and classify each distinct section by content type.

Text to analyze:
---
{page_text}
---

Classify each section as one of:
1. **theory**: Explanatory text, definitions, concepts, background information
2. **worked_example**: Examples with solutions shown step-by-step
3. **practice_exercise**: Problems to solve (without solutions shown)

For each section found, identify:
- Start and end character positions in the text
- Content type (theory/worked_example/practice_exercise)
- Optional title (for theory sections and worked examples, extract or infer a descriptive title)
- Confidence (0.0-1.0)
- Whether it has a solution shown inline (mainly for worked_example and practice_exercise)

Return ONLY valid JSON in this format (no markdown, no code fences):
{{
  "has_content": true/false,
  "content_items": [
    {{
      "start_char": 0,
      "end_char": 500,
      "content_type": "theory",
      "title": "Introduction to Binary Trees",
      "confidence": 0.95,
      "has_solution_inline": false
    }},
    {{
      "start_char": 501,
      "end_char": 1200,
      "content_type": "worked_example",
      "title": "Example: Tree Traversal",
      "confidence": 0.90,
      "has_solution_inline": true
    }},
    {{
      "start_char": 1201,
      "end_char": 1500,
      "content_type": "practice_exercise",
      "title": null,
      "confidence": 0.85,
      "has_solution_inline": false
    }}
  ]
}}

If no content found, return: {{"has_content": false, "content_items": []}}
"""
        return prompt

    def _parse_detection_response(self, response_text: str) -> List[DetectedContent]:
        """
        Parse LLM response into DetectedContent objects.

        Args:
            response_text: Raw LLM response

        Returns:
            List of DetectedContent objects
        """
        # Strip markdown code fences if present (LLM sometimes adds them)
        response_text = response_text.strip()
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response_text = "\n".join(lines)

        # Parse JSON
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse LLM response as JSON: {e}")
            print(f"Response was: {response_text[:200]}")
            return []

        # Validate structure
        if not isinstance(data, dict) or "has_content" not in data:
            print(f"⚠️  Invalid response structure: {data}")
            return []

        if not data["has_content"]:
            return []

        # Parse content items
        detected_content = []
        for content_data in data.get("content_items", []):
            try:
                detected = DetectedContent(
                    start_char=content_data["start_char"],
                    end_char=content_data["end_char"],
                    content_type=content_data["content_type"],
                    title=content_data.get("title"),
                    confidence=float(content_data["confidence"]),
                    has_solution_inline=content_data["has_solution_inline"],
                )

                # Filter by confidence threshold
                if detected.confidence >= self.confidence_threshold:
                    detected_content.append(detected)

            except (KeyError, ValueError) as e:
                print(f"⚠️  Skipping malformed content entry: {e}")
                continue

        return detected_content

    def _create_exercises_from_detected(
        self, detected_list: List[DetectedContent], page: PDFPage, source_pdf: str, course_code: str
    ) -> List[Exercise]:
        """
        Convert DetectedContent objects to Exercise objects (for practice_exercise type only).

        Args:
            detected_list: List of detected content
            page: PDF page
            source_pdf: Source PDF filename
            course_code: Course code

        Returns:
            List of Exercise objects
        """
        exercises = []

        for i, detected in enumerate(detected_list):
            # Only process practice_exercise content type
            if detected.content_type != "practice_exercise":
                continue

            # Extract text from detected boundaries
            exercise_text = page.text[detected.start_char : detected.end_char].strip()

            if not exercise_text:
                continue

            # Generate unique ID
            exercise_id = self._generate_exercise_id(
                course_code, source_pdf, page.page_number, f"llm_{i + 1}", detected.confidence
            )

            # Create Exercise object
            exercise = Exercise(
                id=exercise_id,
                text=exercise_text,
                page_number=page.page_number,
                exercise_number=f"LLM-{i + 1}",  # Mark as LLM-detected
                has_images=len(page.images) > 0,
                image_data=page.images if page.images else [],
                has_latex=page.has_latex,
                latex_content=page.latex_content,
                source_pdf=source_pdf,
            )

            exercises.append(exercise)

        return exercises

    def _create_materials_from_detected(
        self, detected_list: List[DetectedContent], page: PDFPage, source_pdf: str, course_code: str
    ) -> List[LearningMaterial]:
        """
        Convert DetectedContent objects to LearningMaterial objects (for theory/worked_example types).

        Args:
            detected_list: List of detected content
            page: PDF page
            source_pdf: Source PDF filename
            course_code: Course code

        Returns:
            List of LearningMaterial objects
        """
        materials = []

        for i, detected in enumerate(detected_list):
            # Only process theory and worked_example content types
            if detected.content_type not in ["theory", "worked_example"]:
                continue

            # Extract text from detected boundaries
            content_text = page.text[detected.start_char : detected.end_char].strip()

            if not content_text:
                continue

            # Generate unique material ID
            material_id = self._generate_material_id(
                course_code, source_pdf, page.page_number, f"llm_{i + 1}", detected.confidence
            )

            # Create LearningMaterial object
            material = LearningMaterial(
                id=material_id,
                title=detected.title,
                content=content_text,
                material_type=detected.content_type,
                page_number=page.page_number,
                has_images=len(page.images) > 0,
                image_data=page.images if page.images else [],
                has_latex=page.has_latex,
                latex_content=page.latex_content,
                source_pdf=source_pdf,
            )

            materials.append(material)

        return materials

    def _generate_exercise_id(
        self, course_code: str, source_pdf: str, page_number: int, llm_index: str, confidence: float
    ) -> str:
        """
        Generate unique exercise ID for LLM-detected exercises.

        Args:
            course_code: Course code
            source_pdf: Source PDF filename
            page_number: Page number
            llm_index: LLM detection index
            confidence: Detection confidence

        Returns:
            Unique exercise ID
        """
        components = f"{course_code}_{source_pdf}_{page_number}_{llm_index}_{confidence:.2f}"
        hash_obj = hashlib.md5(components.encode())
        short_hash = hash_obj.hexdigest()[:12]

        course_abbrev = course_code.lower().replace("b", "").replace("0", "")[:6]
        return f"{course_abbrev}_smart_{short_hash}"

    def _generate_material_id(
        self, course_code: str, source_pdf: str, page_number: int, llm_index: str, confidence: float
    ) -> str:
        """
        Generate unique material ID for LLM-detected learning materials.

        Args:
            course_code: Course code
            source_pdf: Source PDF filename
            page_number: Page number
            llm_index: LLM detection index
            confidence: Detection confidence

        Returns:
            Unique material ID
        """
        components = f"{course_code}_{source_pdf}_{page_number}_{llm_index}_{confidence:.2f}_mat"
        hash_obj = hashlib.md5(components.encode())
        short_hash = hash_obj.hexdigest()[:12]

        course_abbrev = course_code.lower().replace("b", "").replace("0", "")[:6]
        return f"{course_abbrev}_mat_{short_hash}"

    def _hash_page(self, page_text: str) -> str:
        """
        Generate hash for page content (for caching).

        Args:
            page_text: Page text

        Returns:
            Hash string
        """
        return hashlib.md5(page_text.encode()).hexdigest()

    def validate_exercise(self, exercise: Exercise, min_length: int = 20) -> bool:
        """
        Validate if an exercise has sufficient content.

        Args:
            exercise: Exercise to validate
            min_length: Minimum text length

        Returns:
            True if exercise is valid
        """
        # Delegate to pattern splitter's validation
        return self.pattern_splitter.validate_exercise(exercise, min_length)

    def clean_exercise_text(self, text: str) -> str:
        """
        Clean up exercise text.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Delegate to pattern splitter's cleaning logic
        return self.pattern_splitter.clean_exercise_text(text)
