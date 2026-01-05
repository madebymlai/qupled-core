"""
Configuration settings for Qupled.

This module provides the Config class with all settings.
When installed as a package, paths are relative to the package location
or can be overridden via environment variables.
"""

import os
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv

    # Try multiple locations for .env
    for env_path in [
        Path.cwd() / ".env",  # Current working directory
        Path(__file__).parent.parent / ".env",  # Package root (when running from source)
        Path.home() / ".qupled" / ".env",  # User config directory
    ]:
        if env_path.exists():
            load_dotenv(env_path, override=True)
            break
except ImportError:
    # python-dotenv not installed, will use system environment variables only
    pass


def _get_base_dir():
    """Get base directory, preferring environment variable or user config."""
    if os.getenv("QUPLED_BASE_DIR"):
        return Path(os.getenv("QUPLED_BASE_DIR"))
    # Default: ~/.qupled for installed package, or package parent for dev
    user_dir = Path.home() / ".qupled"
    if user_dir.exists():
        return user_dir
    # Fallback to package parent (for running from source)
    return Path(__file__).parent.parent


class Config:
    """Main configuration class for Qupled."""

    # Paths - can be overridden via QUPLED_BASE_DIR env var
    BASE_DIR = _get_base_dir()
    DATA_DIR = BASE_DIR / "data"
    DB_PATH = DATA_DIR / "qupled.db"
    FILES_PATH = DATA_DIR / "files"
    PDFS_PATH = FILES_PATH / "pdfs"
    IMAGES_PATH = FILES_PATH / "images"
    CACHE_PATH = DATA_DIR / "cache"

    # LLM Settings
    LLM_PROVIDER = os.getenv("QUPLED_LLM_PROVIDER", "deepseek")

    # Ollama Settings (local fallback)
    LLM_PRIMARY_MODEL = os.getenv("QUPLED_PRIMARY_MODEL", "qwen2.5:14b")
    LLM_FAST_MODEL = os.getenv("QUPLED_FAST_MODEL", "llama3.1:8b")
    LLM_EMBED_MODEL = os.getenv("QUPLED_EMBED_MODEL", "nomic-embed-text")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # API Keys (primary)
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    # API Keys (optional - for backward compatibility)
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # DeepSeek Settings (direct API - text tasks)
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    DEEPSEEK_REASONER_MODEL = os.getenv("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner")

    # OpenRouter Settings (vision + image generation)
    OPENROUTER_VISION_MODEL = os.getenv("OPENROUTER_VISION_MODEL", "google/gemini-2.0-flash-001")
    OPENROUTER_VLM_MODEL = os.getenv("OPENROUTER_VLM_MODEL", "google/gemini-2.5-flash")
    OPENROUTER_IMAGE_MODEL = os.getenv("OPENROUTER_IMAGE_MODEL", "black-forest-labs/flux-2-pro")

    # Processing Settings
    PDF_MAX_SIZE_MB = 50
    IMAGE_MAX_SIZE_MB = 10
    BATCH_SIZE = 30  # Process N exercises at once (optimized for DeepSeek/no rate limits)

    # Quiz Settings
    DEFAULT_QUIZ_LENGTH = 10
    SPACED_REPETITION_INTERVALS = [1, 3, 7, 14, 30]  # days
    MASTERY_THRESHOLD = 0.8
    HINT_PENALTY = 0.0  # No penalty for using hints (learning focused)

    # Cache Settings
    CACHE_ENABLED = True
    CACHE_TTL = 3600  # seconds

    # Procedure Pattern Caching (Option 3 - Performance Optimization)
    PROCEDURE_CACHE_ENABLED = os.getenv("QUPLED_PROCEDURE_CACHE_ENABLED", "true").lower() == "true"
    PROCEDURE_CACHE_MIN_CONFIDENCE = float(
        os.getenv("QUPLED_PROCEDURE_CACHE_MIN_CONFIDENCE", "0.85")
    )
    PROCEDURE_CACHE_SCOPE = os.getenv("QUPLED_PROCEDURE_CACHE_SCOPE", "course")
    PROCEDURE_CACHE_EMBEDDING_THRESHOLD = float(
        os.getenv("QUPLED_PROCEDURE_CACHE_EMBEDDING_THRESHOLD", "0.90")
    )
    PROCEDURE_CACHE_TEXT_VALIDATION_THRESHOLD = float(
        os.getenv("QUPLED_PROCEDURE_CACHE_TEXT_THRESHOLD", "0.70")
    )
    PROCEDURE_CACHE_PRELOAD = os.getenv("QUPLED_PROCEDURE_CACHE_PRELOAD", "true").lower() == "true"

    # Analysis Settings
    MIN_EXERCISES_FOR_KNOWLEDGE_ITEM = 2
    KNOWLEDGE_ITEM_SIMILARITY_THRESHOLD = 0.85
    MIN_ANALYSIS_CONFIDENCE = float(os.getenv("QUPLED_MIN_CONFIDENCE", "0.5"))

    # Semantic Similarity Settings
    SEMANTIC_SIMILARITY_ENABLED = os.getenv("QUPLED_SEMANTIC_ENABLED", "true").lower() == "true"
    SEMANTIC_SIMILARITY_THRESHOLD = float(os.getenv("QUPLED_SEMANTIC_THRESHOLD", "0.85"))
    SEMANTIC_LOG_NEAR_MISSES = os.getenv("QUPLED_LOG_NEAR_MISSES", "true").lower() == "true"
    SEMANTIC_EMBEDDING_MODEL = os.getenv(
        "QUPLED_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Study Strategy Settings
    STUDY_STRATEGY_CACHE_DIR = DATA_DIR / "strategy_cache"
    STUDY_STRATEGY_CACHE_ENABLED = True

    # Supported languages
    SUPPORTED_LANGUAGES = ["it", "en"]
    DEFAULT_LANGUAGE = os.getenv("QUPLED_LANGUAGE", "en")

    # Topic Splitting Settings
    GENERIC_TOPIC_THRESHOLD = int(os.getenv("QUPLED_GENERIC_TOPIC_THRESHOLD", "10"))
    TOPIC_CLUSTER_MIN = 4
    TOPIC_CLUSTER_MAX = 6
    TOPIC_SPLITTING_ENABLED = os.getenv("QUPLED_TOPIC_SPLITTING_ENABLED", "true").lower() == "true"

    # Bilingual Translation Detection Settings
    TRANSLATION_DETECTION_ENABLED = (
        os.getenv("QUPLED_TRANSLATION_ENABLED", "true").lower() == "true"
    )
    TRANSLATION_DETECTION_THRESHOLD = float(os.getenv("QUPLED_TRANSLATION_THRESHOLD", "0.70"))
    PREFERRED_LANGUAGES = ["english", "en"]

    # Language Detection Settings
    LANGUAGE_DETECTION_ENABLED = (
        os.getenv("QUPLED_LANGUAGE_DETECTION_ENABLED", "true").lower() == "true"
    )
    AUTO_MERGE_TRANSLATIONS = os.getenv("QUPLED_AUTO_MERGE_TRANSLATIONS", "true").lower() == "true"
    LANGUAGE_CACHE_TTL = int(os.getenv("QUPLED_LANGUAGE_CACHE_TTL", "86400"))

    # Monolingual Mode Settings
    MONOLINGUAL_MODE_ENABLED = os.getenv("QUPLED_MONOLINGUAL_ENABLED", "false").lower() == "true"

    # Smart Splitting Settings
    SMART_SPLIT_ENABLED = os.getenv("QUPLED_SMART_SPLIT_ENABLED", "false").lower() == "true"
    SMART_SPLIT_CONFIDENCE_THRESHOLD = float(os.getenv("QUPLED_SMART_SPLIT_THRESHOLD", "0.7"))
    SMART_SPLIT_MAX_PAGES = int(os.getenv("QUPLED_SMART_SPLIT_MAX_PAGES", "20"))
    SMART_SPLIT_CACHE_ENABLED = os.getenv("QUPLED_SMART_SPLIT_CACHE", "true").lower() == "true"

    # Phase 10: Learning Materials Settings
    LEARNING_MATERIALS_ENABLED = (
        os.getenv("QUPLED_LEARNING_MATERIALS_ENABLED", "true").lower() == "true"
    )
    SHOW_THEORY_BY_DEFAULT = os.getenv("QUPLED_SHOW_THEORY", "true").lower() == "true"
    SHOW_WORKED_EXAMPLES_BY_DEFAULT = (
        os.getenv("QUPLED_SHOW_WORKED_EXAMPLES", "true").lower() == "true"
    )
    MAX_THEORY_SECTIONS_IN_LEARN = int(os.getenv("QUPLED_MAX_THEORY_SECTIONS", "3"))
    MAX_WORKED_EXAMPLES_IN_LEARN = int(os.getenv("QUPLED_MAX_WORKED_EXAMPLES", "2"))
    MATERIAL_TOPIC_SIMILARITY_THRESHOLD = float(
        os.getenv("QUPLED_MATERIAL_TOPIC_THRESHOLD", "0.85")
    )
    WORKED_EXAMPLE_EXERCISE_SIMILARITY_THRESHOLD = float(
        os.getenv("QUPLED_WORKED_EXAMPLE_THRESHOLD", "0.70")
    )

    # Rate Limiting Settings
    PROVIDER_RATE_LIMITS = {
        "ollama": {"requests_per_minute": None, "tokens_per_minute": None, "burst_size": 1},
        "deepseek": {"requests_per_minute": None, "tokens_per_minute": None, "burst_size": 1},
        "openrouter": {"requests_per_minute": None, "tokens_per_minute": None, "burst_size": 1},
    }

    @classmethod
    def ensure_dirs(cls):
        """Create all necessary directories."""
        cls.DATA_DIR.mkdir(exist_ok=True, parents=True)
        cls.FILES_PATH.mkdir(exist_ok=True, parents=True)
        cls.PDFS_PATH.mkdir(exist_ok=True, parents=True)
        cls.IMAGES_PATH.mkdir(exist_ok=True, parents=True)
        cls.CACHE_PATH.mkdir(exist_ok=True, parents=True)
        cls.STUDY_STRATEGY_CACHE_DIR.mkdir(exist_ok=True, parents=True)

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
