"""
PDF processing for Examina.
Extracts text, images, and LaTeX from exam PDFs.
Supports OCR for math-heavy and scanned PDFs.
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
    import pdfplumber

    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pytesseract
    from pdf2image import convert_from_path

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import pycountry

    PYCOUNTRY_AVAILABLE = True
except ImportError:
    PYCOUNTRY_AVAILABLE = False


def get_tesseract_lang(lang_code: str) -> str:
    """Convert ISO 639-1 (en, it) to Tesseract language code (eng, ita).

    Args:
        lang_code: ISO 639-1 two-letter language code

    Returns:
        Tesseract three-letter language code, defaults to 'eng'
    """
    if not lang_code:
        return "eng"

    if PYCOUNTRY_AVAILABLE:
        try:
            lang = pycountry.languages.get(alpha_2=lang_code.lower())
            if lang and hasattr(lang, "alpha_3"):
                return lang.alpha_3
        except Exception:
            pass

    # Fallback for common languages if pycountry fails
    fallback_map = {
        "en": "eng",
        "it": "ita",
        "fr": "fra",
        "de": "deu",
        "es": "spa",
        "pt": "por",
        "nl": "nld",
    }
    return fallback_map.get(lang_code.lower(), "eng")


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
        if not PYMUPDF_AVAILABLE and not PDFPLUMBER_AVAILABLE:
            raise ImportError(
                "Neither PyMuPDF nor pdfplumber is available. "
                "Install at least one: pip install pymupdf or pip install pdfplumber"
            )

    def process_pdf(self, pdf_path: Path) -> PDFContent:
        """Process a PDF file and extract all content.

        Args:
            pdf_path: Path to PDF file

        Returns:
            PDFContent with extracted information
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Try PyMuPDF first (more reliable for images), fallback to pdfplumber
        if PYMUPDF_AVAILABLE:
            return self._process_with_pymupdf(pdf_path)
        elif PDFPLUMBER_AVAILABLE:
            return self._process_with_pdfplumber(pdf_path)
        else:
            raise RuntimeError("No PDF processor available")

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

    def _process_with_pdfplumber(self, pdf_path: Path) -> PDFContent:
        """Process PDF using pdfplumber.

        Args:
            pdf_path: Path to PDF file

        Returns:
            PDFContent with extracted information
        """
        pages = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract text
                text = page.extract_text() or ""

                # pdfplumber doesn't easily extract images as bytes
                # We'll leave images empty for now if using pdfplumber
                images = []

                # Check for LaTeX
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

            # Metadata
            metadata = pdf.metadata or {}

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
        if PYMUPDF_AVAILABLE:
            doc = fitz.open(pdf_path)
            page = doc[page_number - 1]
            text = page.get_text()
            doc.close()
            return text
        elif PDFPLUMBER_AVAILABLE:
            with pdfplumber.open(pdf_path) as pdf:
                page = pdf.pages[page_number - 1]
                return page.extract_text() or ""
        return ""

    def extract_images_from_page(self, pdf_path: Path, page_number: int) -> List[bytes]:
        """Extract images from a specific page.

        Args:
            pdf_path: Path to PDF file
            page_number: Page number (1-indexed)

        Returns:
            List of image bytes
        """
        if not PYMUPDF_AVAILABLE:
            return []  # pdfplumber doesn't easily extract images

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
        if PYMUPDF_AVAILABLE:
            doc = fitz.open(pdf_path)
            count = len(doc)
            doc.close()
            return count
        elif PDFPLUMBER_AVAILABLE:
            with pdfplumber.open(pdf_path) as pdf:
                return len(pdf.pages)
        return 0

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

    def process_pdf_with_ocr(self, pdf_path: Path, lang: str = "en", dpi: int = 300) -> PDFContent:
        """Process PDF using OCR for better math notation extraction.

        This is the primary pipeline for math-heavy PDFs. Renders pages as
        images then OCRs them, preserving spatial layout that pymupdf loses.

        Args:
            pdf_path: Path to PDF file
            lang: ISO 639-1 language code (en, it, fr, etc.)
            dpi: Resolution for rendering (300 recommended for OCR)

        Returns:
            PDFContent with OCR-extracted text

        Raises:
            ImportError: If OCR dependencies not installed
            FileNotFoundError: If PDF not found
        """
        if not OCR_AVAILABLE:
            raise ImportError(
                "OCR dependencies not available. Install: pip install pytesseract pdf2image"
            )

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Convert language code for Tesseract
        tess_lang = get_tesseract_lang(lang)

        # Render PDF pages as images
        page_images = convert_from_path(pdf_path, dpi=dpi)

        pages = []
        for page_num, img in enumerate(page_images, start=1):
            # OCR the page image
            text = pytesseract.image_to_string(img, lang=tess_lang)

            # Extract embedded images using pymupdf (if available)
            images = []
            if PYMUPDF_AVAILABLE:
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

        # Get metadata using pymupdf or pdfplumber
        metadata = {}
        if PYMUPDF_AVAILABLE:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata or {}
            doc.close()
        elif PDFPLUMBER_AVAILABLE:
            with pdfplumber.open(pdf_path) as pdf:
                metadata = pdf.metadata or {}

        return PDFContent(
            file_path=pdf_path, total_pages=len(pages), pages=pages, metadata=metadata
        )
