"""
VLM-based exercise extraction for Qupled.
Uses Vision Language Models to extract exercises from exam PDFs/images.

Two-pass pipeline:
1. VLM (Qwen): OCR + exercise structure detection
2. DeepSeek: Context extraction for parent/standalone exercises
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from PIL import Image  # noqa: F401

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from pdf2image import convert_from_path

    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

from core.pdf import PDFProcessor

__all__ = [
    "extract_exercises",
    "render_page_to_image",
    "get_pdf_page_count",
    "VLMExtractionError",
]


def get_pdf_page_count(pdf_path: Path) -> int:
    """Get the number of pages in a PDF."""
    return PDFProcessor().get_pdf_page_count(pdf_path)


class VLMExtractionError(Exception):
    """Raised when VLM API call fails for exercise extraction."""

    pass


# System prompt for VLM
EXERCISE_EXTRACTION_SYSTEM = """You are an expert exam document analyzer. Your task is to accurately extract hierarchical exercise structures. Think step-by-step with user-provided decision trees. Analyze the document structure carefully before extracting."""

# Prompt for VLM-based exercise extraction (Pass 1: structure + text, no context)
EXERCISE_EXTRACTION_PROMPT = """Extract ALL exercises from these exam pages as a flat list.

DECISION TREE - Apply to each item:

Q1: "Does this start a NEW DISTINCT PROBLEM?"
→ YES: Main exercise (1, 2, 3...)
→ NO: Q2

Q2: "Does this GIVE INFORMATION to student (definitions, setup, given data)?"
→ YES: Part of parent text
→ NO: Q3

Q3: "Does this ask student to DO something (produce answer, calculation, drawing)?"
→ YES: Sub-question (1.1, 1.2...)
→ NO: Part of parent text

RULES:
- Parent text = FULL exercise block (intro + sub-questions + any text after)
- Sub-questions can be marked: a), b), c), 1., 2., i), ii), -, •, or unmarked
- NOT sub-questions: multiple choice options - treat as ONE exercise
- END BEFORE: form fields, blank lines for answers, solutions, page headers/footers, junk, exam instructions
- exercise_number: use hierarchical format (1, 1.1, 1.2) not document numbering
- page_number: 1-indexed
- image_context: describe visual elements; null if none
- Use LaTeX: $inline$ or $$block$$
- Ignore solution sections

Return valid JSON:
{
  "exercises": [
    {"exercise_number": "1", "text": "<full block>", "image_context": "<description or null>", "page_number": 1},
    {"exercise_number": "1.1", "text": "<sub-question>", "page_number": 1},
    {"exercise_number": "1.2", "text": "<sub-question>", "page_number": 1},
    {"exercise_number": "2", "text": "<standalone>", "image_context": "<description or null>", "page_number": 2}
  ]
}"""


# Prompt for DeepSeek context extraction (Pass 2)
CONTEXT_EXTRACTION_PROMPT_PARENT = """Extract the **shared context** that sub-questions need from this parent exercise.

**Good context**: data values, parameters, scenario setup, definitions that sub-questions reference.
Return **null** if sub-questions are **independent** and don't need shared info.
**IMPORTANT**: Return context_summary in **ENGLISH**, even if source is another language.

PARENT EXERCISE:
\"\"\"
{exercise_text}
\"\"\"

Return JSON:
{{"context_summary": "shared context in English" or null}}"""

CONTEXT_EXTRACTION_PROMPT_STANDALONE = """Summarize this exercise for context.

Focus on:
- The **core skill/concept** being tested
- Key **data values**, **parameters**, or given information
- What the student must **DO**

Keep it **concise**.
**IMPORTANT**: Return summary in **ENGLISH**, even if source is another language.

EXERCISE:
\"\"\"
{exercise_text}
\"\"\"

Return JSON:
{{"context_summary": "concise exercise summary in English" or null}}"""


def render_page_to_image(pdf_path: Path, page_num: int, dpi: int = 150) -> bytes:
    """Render a single PDF page to PNG image bytes.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (1-indexed)
        dpi: Resolution for rendering (150 is good balance of quality/size)

    Returns:
        PNG image bytes
    """
    if not PDF2IMAGE_AVAILABLE:
        raise ImportError("pdf2image not available. Install: pip install pdf2image")

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    import io

    images = convert_from_path(
        pdf_path,
        first_page=page_num,
        last_page=page_num,
        dpi=dpi,
    )

    if not images:
        raise ValueError(f"Page {page_num} not found in PDF")

    buf = io.BytesIO()
    images[0].save(buf, format="PNG")
    return buf.getvalue()


def _get_context_summaries(
    parent_data: Dict[str, Dict[str, Any]],
    standalone_exercises: List[Dict[str, Any]],
    logger,
) -> Dict[str, Optional[str]]:
    """Pass 2: Get context summaries from DeepSeek for parents and standalone exercises."""
    import json
    import re

    import requests

    from config import Config

    api_key = Config.OPENROUTER_API_KEY
    if not api_key:
        logger.warning("OPENROUTER_API_KEY not configured, skipping context extraction")
        return {}

    model = "deepseek/deepseek-chat-v3-0324"

    results = {}

    # Process parents
    for parent_num, data in parent_data.items():
        prompt = CONTEXT_EXTRACTION_PROMPT_PARENT.format(exercise_text=data["text"])
        context = _call_deepseek_for_context(api_key, model, prompt, logger)
        results[parent_num] = context

    # Process standalone exercises
    for ex in standalone_exercises:
        prompt = CONTEXT_EXTRACTION_PROMPT_STANDALONE.format(exercise_text=ex["text"])
        context = _call_deepseek_for_context(api_key, model, prompt, logger)
        results[ex["exercise_number"]] = context

    parent_count = len(parent_data)
    standalone_count = len(standalone_exercises)
    null_count = sum(1 for v in results.values() if v is None)
    logger.info(
        f"DeepSeek Pass 2: {parent_count} parents, {standalone_count} standalone, "
        f"{null_count} returned null"
    )

    return results


def _call_deepseek_for_context(
    api_key: str,
    model: str,
    prompt: str,
    logger,
) -> Optional[str]:
    """Call DeepSeek API for context extraction."""
    import json
    import re

    import requests

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 500,
            },
            timeout=30,
        )
        response.raise_for_status()

        result = response.json()
        text = result["choices"][0]["message"]["content"]

        # Parse JSON response
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            data = json.loads(json_match.group())
            summary = data.get("context_summary")
            if summary and summary != "null":
                return summary

    except Exception as e:
        logger.warning(f"DeepSeek context extraction failed: {e}")

    return None


def extract_exercises(
    images: bytes | Path | List[bytes],
) -> List[Dict[str, Any]]:
    """Extract exercises from exam page(s) using VLM.

    Two-pass pipeline:
    1. VLM: OCR + exercise structure detection
    2. DeepSeek: Context extraction for parent/standalone

    Args:
        images: Single image (bytes or Path) or list of page images (bytes)

    Returns:
        List of exercise dicts with fields:
        - exercise_number: str ("1", "1.1", "2", etc.)
        - text: str (exercise content with LaTeX)
        - page_number: int (1-indexed)
        - image_context: str | None (diagram description)
        - exercise_context: str | None (context for sub-questions)

    Raises:
        VLMExtractionError: API failure, invalid JSON response, etc.

    Example:
        >>> # Single image file
        >>> exercises = extract_exercises(Path("exam.png"))
        >>> # PDF pages rendered to images
        >>> page_images = [render_page_to_image(pdf, p) for p in range(1, 4)]
        >>> exercises = extract_exercises(page_images)
    """
    import base64
    import json
    import logging

    import requests

    from config import Config

    logger = logging.getLogger(__name__)

    # Normalize input to list of bytes
    if isinstance(images, Path):
        if not images.exists():
            raise VLMExtractionError(f"File not found: {images}")
        with open(images, "rb") as f:
            image_list = [f.read()]
    elif isinstance(images, bytes):
        image_list = [images]
    elif isinstance(images, list):
        image_list = images
    else:
        raise VLMExtractionError(f"Invalid images type: {type(images)}")

    if not image_list:
        return []

    # Resize large images to max 2048px to reduce API costs
    resized_images = []
    for img_bytes in image_list:
        resized = _resize_image_if_needed(img_bytes, max_size=2048)
        resized_images.append(resized)

    # Build multi-image content for OpenRouter API
    content = [{"type": "text", "text": EXERCISE_EXTRACTION_PROMPT}]
    for img_bytes in resized_images:
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    # Call OpenRouter API
    api_key = Config.OPENROUTER_API_KEY
    if not api_key:
        raise VLMExtractionError("OPENROUTER_API_KEY not configured")

    model = Config.OPENROUTER_VLM_MODEL

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": EXERCISE_EXTRACTION_SYSTEM},
                    {"role": "user", "content": content},
                ],
                "temperature": 0.1,
                "max_tokens": 8000,
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        raise VLMExtractionError(f"API call failed: {e}")

    result = response.json()

    # Extract text from response
    try:
        text = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise VLMExtractionError(f"Unexpected API response format: {e}")

    # Parse JSON from response (may be wrapped in markdown code block)
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON from VLM: {text[:500]}...")
        raise VLMExtractionError(f"Invalid JSON response: {e}")

    exercises = data.get("exercises", [])

    # Parse all exercises
    all_exercises = []
    for ex in exercises:
        if not isinstance(ex, dict):
            continue
        if "exercise_number" not in ex or "text" not in ex:
            continue

        all_exercises.append({
            "exercise_number": str(ex["exercise_number"]),
            "text": str(ex["text"]),
            "page_number": int(ex.get("page_number", 1)),
            "image_context": ex.get("image_context"),
        })

    logger.info(f"VLM Pass 1: extracted {len(all_exercises)} exercises from {len(image_list)} page(s)")

    # Identify parents (exercises that have sub-questions)
    exercise_nums = {ex["exercise_number"] for ex in all_exercises}
    parent_nums = set()
    for ex_num in exercise_nums:
        if "." in ex_num:
            parent_num = ex_num.rsplit(".", 1)[0]
            if parent_num in exercise_nums:
                parent_nums.add(parent_num)

    # Pass 2: Get context from DeepSeek for parents and standalone
    parent_data = {}
    standalone_exercises = []

    for ex in all_exercises:
        ex_num = ex["exercise_number"]
        if ex_num in parent_nums:
            parent_data[ex_num] = {
                "text": ex["text"],
                "image_context": ex.get("image_context"),
            }
        elif "." not in ex_num:
            standalone_exercises.append(ex)

    # Call DeepSeek for context (parents + standalone)
    if parent_data or standalone_exercises:
        context_results = _get_context_summaries(
            parent_data, standalone_exercises, logger
        )
    else:
        context_results = {}

    # Build final exercise list (subs only, with inherited context)
    final_exercises = []
    for ex in all_exercises:
        ex_num = ex["exercise_number"]

        if ex_num in parent_nums:
            # Skip parent entries - context is inherited by subs
            continue

        if "." in ex_num:
            # Sub-question - inherit context from immediate parent
            parent_num = ex_num.rsplit(".", 1)[0]
            parent = parent_data.get(parent_num, {})
            ex["exercise_context"] = context_results.get(parent_num)
            if ex.get("image_context") is None:
                ex["image_context"] = parent.get("image_context")
        else:
            # Standalone - use its own context
            ex["exercise_context"] = context_results.get(ex_num)

        final_exercises.append(ex)

    logger.info(f"Two-pass extraction complete: {len(final_exercises)} exercises")
    return final_exercises


def _resize_image_if_needed(image_bytes: bytes, max_size: int = 2048) -> bytes:
    """Resize image if either dimension exceeds max_size."""
    if not PIL_AVAILABLE:
        return image_bytes

    import io

    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes))
    width, height = img.size

    if width <= max_size and height <= max_size:
        return image_bytes

    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)

    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
