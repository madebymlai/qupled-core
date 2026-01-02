"""
PDF processing utilities for Qupled.
Extracts text, images, and LaTeX from PDFs using PyMuPDF and Mathpix.

Used by:
- NoteSplitter for notes processing
- ExerciseSplitter for page count utility
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

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
        """Process PDF using PyMuPDF (fitz)."""
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

        metadata = doc.metadata or {}
        doc.close()

        return PDFContent(
            file_path=pdf_path, total_pages=len(pages), pages=pages, metadata=metadata
        )

    def _detect_latex(self, text: str) -> Tuple[bool, Optional[str]]:
        """Detect LaTeX formulas in text."""
        latex_patterns = [
            r"\$.*?\$",
            r"\$\$.*?\$\$",
            r"\\begin\{equation\}.*?\\end\{equation\}",
            r"\\begin\{align\}.*?\\end\{align\}",
            r"\\begin\{math\}.*?\\end\{math\}",
            r"\\frac\{.*?\}\{.*?\}",
            r"\\sum",
            r"\\int",
            r"\\prod",
            r"\\alpha",
            r"\\beta",
            r"\\gamma",
        ]

        latex_content = []
        has_latex = False

        for pattern in latex_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                has_latex = True
                latex_content.extend(matches)

        if has_latex:
            return True, "\n".join(latex_content[:10])
        return False, None

    def extract_text_from_page(self, pdf_path: Path, page_number: int) -> str:
        """Extract text from a specific page (1-indexed)."""
        doc = fitz.open(pdf_path)
        page = doc[page_number - 1]
        text = page.get_text()
        doc.close()
        return text

    def extract_images_from_page(self, pdf_path: Path, page_number: int) -> List[bytes]:
        """Extract images from a specific page (1-indexed)."""
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
        """Get the number of pages in a PDF."""
        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count

    def is_scanned_pdf(self, pdf_path: Path, sample_pages: int = 3) -> bool:
        """Detect if PDF is scanned (image-based) or digital (text-based)."""
        total_pages = self.get_pdf_page_count(pdf_path)
        pages_to_check = min(sample_pages, total_pages)

        text_chars = 0
        for page_num in range(1, pages_to_check + 1):
            text = self.extract_text_from_page(pdf_path, page_num)
            text_chars += len(text.strip())

        avg_chars_per_page = text_chars / pages_to_check if pages_to_check > 0 else 0
        return avg_chars_per_page < 100

    def process_pdf_with_mathpix(self, pdf_path: Path) -> PDFContent:
        """Process PDF using Mathpix OCR for high-quality text + LaTeX extraction."""
        if not MATHPIX_AVAILABLE:
            raise ImportError(
                "Mathpix not configured. Set MATHPIX_APP_ID and MATHPIX_APP_KEY."
            )

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        import time

        import requests

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        url = "https://api.mathpix.com/v3/pdf"
        headers = {
            "app_id": Config.MATHPIX_APP_ID,
            "app_key": Config.MATHPIX_APP_KEY,
        }

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

        status_url = f"https://api.mathpix.com/v3/pdf/{pdf_id}"
        max_wait = 300
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

        text_url = f"https://api.mathpix.com/v3/pdf/{pdf_id}.mmd"
        text_resp = requests.get(text_url, headers=headers, timeout=30)
        text_resp.raise_for_status()
        full_text = text_resp.text

        page_texts = full_text.split("\\newpage") if "\\newpage" in full_text else [full_text]

        pages = []
        for page_num, page_text in enumerate(page_texts, start=1):
            page_text = page_text.strip()
            if not page_text:
                continue

            has_latex, latex_content = self._detect_latex(page_text)
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

        doc = fitz.open(pdf_path)
        metadata = doc.metadata or {}
        total_pages = len(doc)
        doc.close()

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
        """Process a single image (PNG/JPG) using Mathpix OCR."""
        if not MATHPIX_AVAILABLE:
            raise ImportError(
                "Mathpix not configured. Set MATHPIX_APP_ID and MATHPIX_APP_KEY."
            )

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        suffix = image_path.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg"}:
            raise ValueError(f"Unsupported image format: {suffix}. Use PNG or JPG.")

        import base64

        import requests

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        content_type = "image/png" if suffix == ".png" else "image/jpeg"
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:{content_type};base64,{image_b64}"

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

        return result.get("latex_styled") or result.get("text", "")

    def process_file_with_mathpix(self, file_path: Path) -> str:
        """Process any supported file (PDF or image) using Mathpix."""
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            content = self.process_pdf_with_mathpix(file_path)
            return "\n\n".join(page.text for page in content.pages if page.text)
        elif suffix in {".png", ".jpg", ".jpeg"}:
            return self.process_image_with_mathpix(file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. Supported: PDF, PNG, JPG."
            )
