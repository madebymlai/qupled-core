"""
File management for Examina.
Handles ZIP extraction, PDF storage, and image storage.
"""

import shutil
import hashlib
from pathlib import Path
from typing import List
from zipfile import ZipFile, is_zipfile

from config import Config


class FileManager:
    """Manages file operations for Examina."""

    def __init__(self):
        """Initialize file manager."""
        self.base_path = Config.FILES_PATH
        self.pdfs_path = Config.PDFS_PATH
        self.images_path = Config.IMAGES_PATH

    def extract_zip(self, zip_path: str, course_code: str) -> List[Path]:
        """Extract PDFs from ZIP file.

        Args:
            zip_path: Path to ZIP file
            course_code: Course code for organization

        Returns:
            List of extracted PDF file paths

        Raises:
            ValueError: If file is not a valid ZIP
            FileNotFoundError: If ZIP file doesn't exist
        """
        zip_path = Path(zip_path)

        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP file not found: {zip_path}")

        if not is_zipfile(zip_path):
            raise ValueError(f"Not a valid ZIP file: {zip_path}")

        # Create course PDF directory
        course_pdf_dir = Config.get_course_pdf_dir(course_code)

        # Extract PDFs
        extracted_pdfs = []
        with ZipFile(zip_path, "r") as zip_file:
            for file_info in zip_file.filelist:
                # Only extract PDF files
                if file_info.filename.lower().endswith(".pdf"):
                    # Extract to course directory
                    extracted_path = zip_file.extract(file_info, course_pdf_dir)
                    extracted_pdfs.append(Path(extracted_path))

        return extracted_pdfs

    def store_pdf(self, pdf_path: Path, course_code: str) -> Path:
        """Store a PDF file in the course directory.

        Args:
            pdf_path: Path to PDF file
            course_code: Course code

        Returns:
            Path to stored PDF (relative to FILES_PATH)
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        course_pdf_dir = Config.get_course_pdf_dir(course_code)
        dest_path = course_pdf_dir / pdf_path.name

        # Copy file if not already in destination
        if pdf_path.resolve() != dest_path.resolve():
            shutil.copy2(pdf_path, dest_path)

        # Return relative path from FILES_PATH
        return dest_path.relative_to(self.base_path)

    def store_image(
        self,
        image_data: bytes,
        course_code: str,
        exercise_id: str,
        image_index: int = 0,
        ext: str = "png",
    ) -> Path:
        """Store an extracted image.

        Args:
            image_data: Image bytes
            course_code: Course code
            exercise_id: Exercise ID
            image_index: Index of image within exercise (for multiple images)
            ext: File extension (default: png)

        Returns:
            Path to stored image (relative to FILES_PATH)
        """
        course_images_dir = Config.get_course_images_dir(course_code)

        # Generate filename
        if image_index == 0:
            filename = f"{exercise_id}.{ext}"
        else:
            filename = f"{exercise_id}_{image_index}.{ext}"

        dest_path = course_images_dir / filename

        # Write image data
        with open(dest_path, "wb") as f:
            f.write(image_data)

        # Return relative path from FILES_PATH
        return dest_path.relative_to(self.base_path)

    def get_pdf_path(self, relative_path: Path) -> Path:
        """Get absolute path for a PDF file.

        Args:
            relative_path: Path relative to FILES_PATH

        Returns:
            Absolute path to PDF
        """
        return self.base_path / relative_path

    def get_image_path(self, relative_path: Path) -> Path:
        """Get absolute path for an image file.

        Args:
            relative_path: Path relative to FILES_PATH

        Returns:
            Absolute path to image
        """
        return self.base_path / relative_path

    def list_course_pdfs(self, course_code: str) -> List[Path]:
        """List all PDFs for a course.

        Args:
            course_code: Course code

        Returns:
            List of PDF paths (relative to FILES_PATH)
        """
        course_pdf_dir = Config.get_course_pdf_dir(course_code)

        if not course_pdf_dir.exists():
            return []

        pdfs = []
        for pdf_file in course_pdf_dir.rglob("*.pdf"):
            pdfs.append(pdf_file.relative_to(self.base_path))

        return sorted(pdfs)

    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file.

        Args:
            file_path: Path to file

        Returns:
            Hex digest of file hash
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks for large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def get_file_size(self, file_path: Path) -> int:
        """Get file size in bytes.

        Args:
            file_path: Path to file

        Returns:
            File size in bytes
        """
        return file_path.stat().st_size

    def get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in megabytes.

        Args:
            file_path: Path to file

        Returns:
            File size in MB
        """
        return self.get_file_size(file_path) / (1024 * 1024)

    def validate_pdf_size(self, pdf_path: Path) -> bool:
        """Check if PDF size is within limits.

        Args:
            pdf_path: Path to PDF file

        Returns:
            True if size is acceptable, False otherwise
        """
        size_mb = self.get_file_size_mb(pdf_path)
        return size_mb <= Config.PDF_MAX_SIZE_MB

    def cleanup_course_files(self, course_code: str, confirm: bool = False):
        """Delete all files for a course.

        Args:
            course_code: Course code
            confirm: Must be True to actually delete

        Raises:
            ValueError: If confirm is not True
        """
        if not confirm:
            raise ValueError("Must confirm deletion by setting confirm=True")

        # Delete PDFs
        course_pdf_dir = self.pdfs_path / course_code
        if course_pdf_dir.exists():
            shutil.rmtree(course_pdf_dir)

        # Delete images
        course_images_dir = self.images_path / course_code
        if course_images_dir.exists():
            shutil.rmtree(course_images_dir)
