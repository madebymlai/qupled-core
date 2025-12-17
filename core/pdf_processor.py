"""
PDF processing for Examina.
Extracts text, images, and LaTeX from exam PDFs.
Supports Mathpix and Vision LLM for math-heavy and scanned PDFs.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from pdf2image import convert_from_path

    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

# Vision LLM support
VISION_AVAILABLE = False
try:
    from models.llm_manager import LLMManager

    VISION_AVAILABLE = True
except ImportError:
    pass

# Mathpix OCR support
MATHPIX_AVAILABLE = False
try:
    from config import Config
    if Config.MATHPIX_APP_ID and Config.MATHPIX_APP_KEY:
        MATHPIX_AVAILABLE = True
except ImportError:
    pass


@dataclass
class PDFPage:
    """Represents a single page from a PDF."""

    page_number: int
    text: str
    images: List[bytes]
    has_latex: bool
    latex_content: Optional[str] = None


@dataclass
class PDFContent:
    """Complete PDF content extraction."""

    file_path: Path
    total_pages: int
    pages: List[PDFPage]
    metadata: Dict[str, Any]


class PDFProcessor:
    """Processes PDF files to extract text, images, and formulas."""

    def __init__(self):
        """Initialize PDF processor."""
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF is required. Install: pip install pymupdf")

    def process_pdf(self, pdf_path: Path) -> PDFContent:
        """Process a PDF file and extract all content.

        Args:
            pdf_path: Path to PDF file

        Returns:
            PDFContent with extracted information
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        return self._process_with_pymupdf(pdf_path)

    def _process_with_pymupdf(self, pdf_path: Path) -> PDFContent:
        """Process PDF using PyMuPDF (fitz).

        Args:
            pdf_path: Path to PDF file

        Returns:
            PDFContent with extracted information
        """
        doc = fitz.open(pdf_path)
        pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Extract text
            text = page.get_text()

            # Extract images
            images = []
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    images.append(image_bytes)
                except Exception:
                    # Skip problematic images
                    continue

            # Check for LaTeX (simple heuristic)
            has_latex, latex_content = self._detect_latex(text)

            pages.append(
                PDFPage(
                    page_number=page_num + 1,
                    text=text,
                    images=images,
                    has_latex=has_latex,
                    latex_content=latex_content,
                )
            )

        # Extract metadata
        metadata = doc.metadata or {}

        doc.close()

        return PDFContent(
            file_path=pdf_path, total_pages=len(pages), pages=pages, metadata=metadata
        )

    def _detect_latex(self, text: str) -> Tuple[bool, Optional[str]]:
        """Detect LaTeX formulas in text.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (has_latex: bool, latex_content: str or None)
        """
        # Common LaTeX patterns
        latex_patterns = [
            r"\$.*?\$",  # Inline math $...$
            r"\$\$.*?\$\$",  # Display math $$...$$
            r"\\begin\{equation\}.*?\\end\{equation\}",
            r"\\begin\{align\}.*?\\end\{align\}",
            r"\\begin\{math\}.*?\\end\{math\}",
            r"\\frac\{.*?\}\{.*?\}",  # Fractions
            r"\\sum",
            r"\\int",
            r"\\prod",  # Math operators
            r"\\alpha",
            r"\\beta",
            r"\\gamma",  # Greek letters
        ]

        latex_content = []
        has_latex = False

        for pattern in latex_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                has_latex = True
                latex_content.extend(matches)

        if has_latex:
            return True, "\n".join(latex_content[:10])  # Limit to first 10 matches
        return False, None

    def extract_text_from_page(self, pdf_path: Path, page_number: int) -> str:
        """Extract text from a specific page.

        Args:
            pdf_path: Path to PDF file
            page_number: Page number (1-indexed)

        Returns:
            Extracted text
        """
        doc = fitz.open(pdf_path)
        page = doc[page_number - 1]
        text = page.get_text()
        doc.close()
        return text

    def extract_images_from_page(self, pdf_path: Path, page_number: int) -> List[bytes]:
        """Extract images from a specific page.

        Args:
            pdf_path: Path to PDF file
            page_number: Page number (1-indexed)

        Returns:
            List of image bytes
        """
        images = []
        doc = fitz.open(pdf_path)
        page = doc[page_number - 1]

        image_list = page.get_images()
        for img in image_list:
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                images.append(image_bytes)
            except Exception:
                continue

        doc.close()
        return images

    def get_pdf_page_count(self, pdf_path: Path) -> int:
        """Get the number of pages in a PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Number of pages
        """
        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count

    def is_scanned_pdf(self, pdf_path: Path, sample_pages: int = 3) -> bool:
        """Detect if PDF is scanned (image-based) or digital (text-based).

        Args:
            pdf_path: Path to PDF file
            sample_pages: Number of pages to sample

        Returns:
            True if PDF appears to be scanned (needs OCR)
        """
        total_pages = self.get_pdf_page_count(pdf_path)
        pages_to_check = min(sample_pages, total_pages)

        text_chars = 0
        for page_num in range(1, pages_to_check + 1):
            text = self.extract_text_from_page(pdf_path, page_num)
            text_chars += len(text.strip())

        # If very little text extracted, likely scanned
        avg_chars_per_page = text_chars / pages_to_check if pages_to_check > 0 else 0
        return avg_chars_per_page < 100  # Threshold: less than 100 chars/page = scanned

    def process_pdf_with_mathpix(self, pdf_path: Path) -> PDFContent:
        """Process PDF using Mathpix OCR for high-quality text + LaTeX extraction.

        Mathpix is specialized for math OCR and produces clean LaTeX output.

        Args:
            pdf_path: Path to PDF file

        Returns:
            PDFContent with Mathpix-extracted text

        Raises:
            ImportError: If Mathpix not configured
            FileNotFoundError: If PDF not found
        """
        if not MATHPIX_AVAILABLE:
            raise ImportError(
                "Mathpix not configured. Set MATHPIX_APP_ID and MATHPIX_APP_KEY."
            )

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        import requests
        import time

        # Read PDF file
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        # Send to Mathpix API
        url = "https://api.mathpix.com/v3/pdf"
        headers = {
            "app_id": Config.MATHPIX_APP_ID,
            "app_key": Config.MATHPIX_APP_KEY,
        }

        # Upload PDF and start conversion
        response = requests.post(
            url,
            headers=headers,
            files={"file": (pdf_path.name, pdf_bytes, "application/pdf")},
            data={
                "options_json": '{"math_inline_delimiters": ["$", "$"], "math_display_delimiters": ["$$", "$$"]}'
            },
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        pdf_id = result.get("pdf_id")

        if not pdf_id:
            raise RuntimeError(f"Mathpix upload failed: {result}")

        # Poll for completion
        status_url = f"https://api.mathpix.com/v3/pdf/{pdf_id}"
        max_wait = 300  # 5 minutes max
        poll_interval = 2
        waited = 0

        while waited < max_wait:
            status_resp = requests.get(status_url, headers=headers, timeout=30)
            status_resp.raise_for_status()
            status = status_resp.json()

            if status.get("status") == "completed":
                break
            elif status.get("status") == "error":
                raise RuntimeError(f"Mathpix processing error: {status}")

            time.sleep(poll_interval)
            waited += poll_interval

        if waited >= max_wait:
            raise TimeoutError("Mathpix processing timed out")

        # Get the extracted text (mmd format = markdown with math)
        text_url = f"https://api.mathpix.com/v3/pdf/{pdf_id}.mmd"
        text_resp = requests.get(text_url, headers=headers, timeout=30)
        text_resp.raise_for_status()
        full_text = text_resp.text

        # Split by page markers if present, otherwise treat as single page
        # Mathpix uses \newpage or page markers
        page_texts = full_text.split("\\newpage") if "\\newpage" in full_text else [full_text]

        pages = []
        for page_num, page_text in enumerate(page_texts, start=1):
            page_text = page_text.strip()
            if not page_text:
                continue

            # Check for LaTeX patterns
            has_latex, latex_content = self._detect_latex(page_text)

            # Extract embedded images using pymupdf
            images = self.extract_images_from_page(pdf_path, page_num) if page_num <= self.get_pdf_page_count(pdf_path) else []

            pages.append(
                PDFPage(
                    page_number=page_num,
                    text=page_text,
                    images=images,
                    has_latex=has_latex,
                    latex_content=latex_content,
                )
            )

        # Get metadata using pymupdf
        doc = fitz.open(pdf_path)
        metadata = doc.metadata or {}
        total_pages = len(doc)
        doc.close()

        # If Mathpix returned fewer pages, pad with empty pages
        while len(pages) < total_pages:
            pages.append(
                PDFPage(
                    page_number=len(pages) + 1,
                    text="",
                    images=[],
                    has_latex=False,
                    latex_content=[],
                )
            )

        return PDFContent(
            file_path=pdf_path, total_pages=total_pages, pages=pages, metadata=metadata
        )

    def process_image_with_mathpix(self, image_path: Path) -> str:
        """Process a single image (PNG/JPG) using Mathpix OCR.

        Uses Mathpix /v3/text endpoint for high-quality math OCR from images.

        Args:
            image_path: Path to image file (PNG, JPG, JPEG)

        Returns:
            Extracted text with LaTeX formatting

        Raises:
            ImportError: If Mathpix not configured
            FileNotFoundError: If image not found
            ValueError: If unsupported image format
        """
        if not MATHPIX_AVAILABLE:
            raise ImportError(
                "Mathpix not configured. Set MATHPIX_APP_ID and MATHPIX_APP_KEY."
            )

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Validate file extension
        suffix = image_path.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg"}:
            raise ValueError(f"Unsupported image format: {suffix}. Use PNG or JPG.")

        import requests
        import base64

        # Read and encode image
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # Determine content type
        content_type = "image/png" if suffix == ".png" else "image/jpeg"
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:{content_type};base64,{image_b64}"

        # Send to Mathpix /v3/text API
        url = "https://api.mathpix.com/v3/text"
        headers = {
            "app_id": Config.MATHPIX_APP_ID,
            "app_key": Config.MATHPIX_APP_KEY,
            "Content-type": "application/json",
        }

        payload = {
            "src": data_uri,
            "formats": ["text", "latex_styled"],
            "math_inline_delimiters": ["$", "$"],
            "math_display_delimiters": ["$$", "$$"],
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        # Prefer latex_styled if available, otherwise use text
        text = result.get("latex_styled") or result.get("text", "")

        return text

    def process_file_with_mathpix(self, file_path: Path) -> str:
        """Process any supported file (PDF or image) using Mathpix.

        Routes to appropriate Mathpix endpoint based on file type:
        - PDF: Uses /v3/pdf (async polling)
        - Images: Uses /v3/text (sync)

        Args:
            file_path: Path to file (PDF, PNG, JPG)

        Returns:
            Extracted text with LaTeX formatting

        Raises:
            ImportError: If Mathpix not configured
            FileNotFoundError: If file not found
            ValueError: If unsupported file format
        """
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            # Use PDF endpoint - returns PDFContent, extract text from pages
            content = self.process_pdf_with_mathpix(file_path)
            return "\n\n".join(page.text for page in content.pages if page.text)
        elif suffix in {".png", ".jpg", ".jpeg"}:
            # Use image endpoint
            return self.process_image_with_mathpix(file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. Supported: PDF, PNG, JPG."
            )

    def process_pdf_with_vision(
        self,
        pdf_path: Path,
        llm_manager: "LLMManager" = None,
        dpi: int = 200,
    ) -> PDFContent:
        """Process PDF using Vision LLM for OCR with proper LaTeX extraction.

        This is the primary pipeline for math-heavy PDFs. Renders pages as
        images then uses DeepSeek Vision to extract text with proper LaTeX.

        Args:
            pdf_path: Path to PDF file
            llm_manager: LLMManager instance (defaults to DeepSeek)
            dpi: Resolution for rendering (200 is good balance of quality/speed)

        Returns:
            PDFContent with Vision-OCR extracted text

        Raises:
            ImportError: If pdf2image not installed
            FileNotFoundError: If PDF not found
        """
        if not PDF2IMAGE_AVAILABLE:
            raise ImportError(
                "pdf2image not available. Install: pip install pdf2image"
            )

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Create LLMManager if not provided
        if llm_manager is None:
            if not VISION_AVAILABLE:
                raise ImportError(
                    "LLMManager not available. Ensure models.llm_manager is importable."
                )
            llm_manager = LLMManager(provider="deepseek")

        # Render PDF pages as images
        page_images = convert_from_path(pdf_path, dpi=dpi)

        pages = []
        for page_num, img in enumerate(page_images, start=1):
            # Convert PIL Image to bytes for Vision API
            import io
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()

            # OCR using Vision LLM
            text = self._ocr_page_with_vision(img_bytes, llm_manager)

            # Extract embedded images using pymupdf
            images = self.extract_images_from_page(pdf_path, page_num)

            # Check for LaTeX patterns in OCR text
            has_latex, latex_content = self._detect_latex(text)

            pages.append(
                PDFPage(
                    page_number=page_num,
                    text=text,
                    images=images,
                    has_latex=has_latex,
                    latex_content=latex_content,
                )
            )

        # Get metadata using pymupdf
        doc = fitz.open(pdf_path)
        metadata = doc.metadata or {}
        doc.close()

        return PDFContent(
            file_path=pdf_path, total_pages=len(pages), pages=pages, metadata=metadata
        )

    def _ocr_page_with_vision(
        self,
        image_bytes: bytes,
        llm_manager: "LLMManager",
    ) -> str:
        """Extract text from PDF page image using Vision LLM.

        Args:
            image_bytes: PNG image data
            llm_manager: LLMManager with vision support

        Returns:
            Extracted text with proper LaTeX formatting
        """
        prompt = """Extract ALL text from this PDF page.

CRITICAL for math notation:
- Use proper LaTeX for equations: $...$ inline, $$...$$ block
- Matrices: $$\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}$$
- Vectors: $\\vec{v}$ or $(x, y, z)$
- Fractions: $\\frac{a}{b}$
- Greek letters: $\\alpha$, $\\beta$, $\\gamma$, etc.
- Subscripts: $x_1$, superscripts: $x^2$
- Preserve exercise numbering (1., 2., a), b), i), ii), etc.)

Output the text with proper formatting. Preserve paragraph structure."""

        response = llm_manager.generate_with_image(
            prompt=prompt,
            image_bytes=image_bytes,
            max_tokens=4000,
            temperature=0.1,  # Low temperature for accurate extraction
        )

        if response.success:
            return response.text
        else:
            # Log error but return empty string to continue processing
            import logging
            logging.getLogger(__name__).warning(
                f"Vision OCR failed: {response.error}"
            )
            return ""

    def describe_image(
        self,
        image_bytes: bytes,
        llm_manager: "LLMManager" = None,
    ) -> str:
        """Get text description of an image for exercise context.

        Used to generate image_context for exercises with diagrams.

        Args:
            image_bytes: Image data (PNG, JPEG)
            llm_manager: LLMManager instance (defaults to DeepSeek)

        Returns:
            Text description of the image content
        """
        if llm_manager is None:
            if not VISION_AVAILABLE:
                return ""
            llm_manager = LLMManager(provider="deepseek")

        prompt = """Describe this image for a student studying. Include:
- What it shows (diagram type, components)
- Any labels, text, or annotations visible
- Key values or measurements
- Relationships between elements

Be concise but complete. Focus on information needed to understand the exercise."""

        response = llm_manager.generate_with_image(
            prompt=prompt,
            image_bytes=image_bytes,
            max_tokens=500,
            temperature=0.3,
        )

        if response.success:
            return response.text
        return ""

    def extract_images_with_context(
        self,
        pdf_path: Path,
        page_number: int,
        llm_manager: "LLMManager" = None,
        min_size: int = 50,
        max_page_ratio: float = 0.8,
    ) -> List[Dict[str, Any]]:
        """Extract images from page with Vision LLM descriptions.

        Filters out noise (icons, backgrounds) and describes meaningful images.

        Args:
            pdf_path: Path to PDF file
            page_number: Page number (1-indexed)
            llm_manager: LLMManager instance
            min_size: Minimum image dimension (pixels)
            max_page_ratio: Maximum ratio of page size (filter backgrounds)

        Returns:
            List of dicts with 'bytes', 'position', 'description' keys
        """
        doc = fitz.open(pdf_path)
        page = doc[page_number - 1]
        page_rect = page.rect

        results = []
        image_list = page.get_images()

        for img in image_list:
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                # Filter: too small (icons, bullets)
                if width < min_size or height < min_size:
                    continue

                # Filter: too large (backgrounds)
                page_width = page_rect.width
                page_height = page_rect.height
                if width > page_width * max_page_ratio and height > page_height * max_page_ratio:
                    continue

                # Get image position on page
                img_rects = page.get_image_rects(xref)
                position = None
                if img_rects:
                    rect = img_rects[0]
                    position = {
                        "x": rect.x0,
                        "y": rect.y0,
                        "width": rect.width,
                        "height": rect.height,
                    }

                # Get description using Vision LLM
                description = ""
                if llm_manager:
                    description = self.describe_image(image_bytes, llm_manager)

                results.append({
                    "bytes": image_bytes,
                    "position": position,
                    "description": description,
                    "width": width,
                    "height": height,
                })

            except Exception:
                continue

        doc.close()
        return results
