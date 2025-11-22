"""
Configuration settings for Examina.
"""

import os
from pathlib import Path

class Config:
    """Main configuration class for Examina."""

    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    DB_PATH = DATA_DIR / "examina.db"
    CHROMA_PATH = DATA_DIR / "chroma"
    FILES_PATH = DATA_DIR / "files"
    PDFS_PATH = FILES_PATH / "pdfs"
    IMAGES_PATH = FILES_PATH / "images"
    CACHE_PATH = DATA_DIR / "cache"

    # LLM Settings
    LLM_PROVIDER = os.getenv("EXAMINA_LLM_PROVIDER", "ollama")
    LLM_PRIMARY_MODEL = os.getenv("EXAMINA_PRIMARY_MODEL", "qwen2.5:14b")
    LLM_FAST_MODEL = os.getenv("EXAMINA_FAST_MODEL", "llama3.1:8b")
    LLM_EMBED_MODEL = os.getenv("EXAMINA_EMBED_MODEL", "nomic-embed-text")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # API Keys (optional fallback)
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "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")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Anthropic Settings
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")  # Sonnet 4.5

    # Groq Settings
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")  # Best rate limits on Groq free tier

    # Processing Settings
    PDF_MAX_SIZE_MB = 50
    IMAGE_MAX_SIZE_MB = 10
    BATCH_SIZE = 10  # Process N exercises at once

    # Quiz Settings
    DEFAULT_QUIZ_LENGTH = 10
    SPACED_REPETITION_INTERVALS = [1, 3, 7, 14, 30]  # days
    MASTERY_THRESHOLD = 0.8
    HINT_PENALTY = 0.0  # No penalty for using hints (learning focused)

    # Cache Settings
    CACHE_ENABLED = True
    CACHE_TTL = 3600  # seconds

    # Analysis Settings
    MIN_EXERCISES_FOR_CORE_LOOP = 2  # Minimum exercises to establish a core loop
    CORE_LOOP_SIMILARITY_THRESHOLD = 0.85  # Similarity threshold for merging core loops

    # Supported languages
    SUPPORTED_LANGUAGES = ["it", "en"]  # Italian and English
    DEFAULT_LANGUAGE = os.getenv("EXAMINA_LANGUAGE", "en")  # Default to English

    @classmethod
    def ensure_dirs(cls):
        """Create all necessary directories."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.FILES_PATH.mkdir(exist_ok=True)
        cls.PDFS_PATH.mkdir(exist_ok=True)
        cls.IMAGES_PATH.mkdir(exist_ok=True)
        cls.CHROMA_PATH.mkdir(exist_ok=True)
        cls.CACHE_PATH.mkdir(exist_ok=True)

    @classmethod
    def get_course_pdf_dir(cls, course_code):
        """Get PDF directory for a specific course."""
        path = cls.PDFS_PATH / course_code
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def get_course_images_dir(cls, course_code):
        """Get images directory for a specific course."""
        path = cls.IMAGES_PATH / course_code
        path.mkdir(parents=True, exist_ok=True)
        return path
