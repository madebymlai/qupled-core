"""
Configuration settings for Examina.

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
        Path.home() / ".examina" / ".env",  # User config directory
    ]:
        if env_path.exists():
            load_dotenv(env_path, override=True)
            break
except ImportError:
    # python-dotenv not installed, will use system environment variables only
    pass


def _get_base_dir():
    """Get base directory, preferring environment variable or user config."""
    if os.getenv("EXAMINA_BASE_DIR"):
        return Path(os.getenv("EXAMINA_BASE_DIR"))
    # Default: ~/.examina for installed package, or package parent for dev
    user_dir = Path.home() / ".examina"
    if user_dir.exists():
        return user_dir
    # Fallback to package parent (for running from source)
    return Path(__file__).parent.parent


class Config:
    """Main configuration class for Examina."""

    # Paths - can be overridden via EXAMINA_BASE_DIR env var
    BASE_DIR = _get_base_dir()
    DATA_DIR = BASE_DIR / "data"
    DB_PATH = DATA_DIR / "examina.db"
    CHROMA_PATH = DATA_DIR / "chroma"
    FILES_PATH = DATA_DIR / "files"
    PDFS_PATH = FILES_PATH / "pdfs"
    IMAGES_PATH = FILES_PATH / "images"
    CACHE_PATH = DATA_DIR / "cache"

    # LLM Settings
    LLM_PROVIDER = os.getenv("EXAMINA_LLM_PROVIDER", "anthropic")

    # Provider Routing Settings (NEW - Provider Routing Architecture)
    # Default provider profile to use when --profile flag not specified
    PROVIDER_PROFILE = os.getenv("EXAMINA_PROVIDER_PROFILE", "free")  # Options: free, pro, local
    PROVIDER_PROFILES_PATH = (
        None  # Will be set to BASE_DIR / "config" / "provider_profiles.yaml" below
    )

    # Ollama Settings (only used if provider=ollama)
    LLM_PRIMARY_MODEL = os.getenv("EXAMINA_PRIMARY_MODEL", "qwen2.5:14b")
    LLM_FAST_MODEL = os.getenv("EXAMINA_FAST_MODEL", "llama3.1:8b")
    LLM_EMBED_MODEL = os.getenv("EXAMINA_EMBED_MODEL", "nomic-embed-text")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # API Keys
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

    # Anthropic Settings
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")  # Sonnet 4.5

    # Groq Settings
    GROQ_MODEL = os.getenv(
        "GROQ_MODEL", "llama-3.3-70b-versatile"
    )  # Best rate limits on Groq free tier

    # DeepSeek Settings
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")  # DeepSeek v3 (671B MoE)
    DEEPSEEK_REASONER_MODEL = os.getenv(
        "DEEPSEEK_REASONER_MODEL", "deepseek-reasoner"
    )  # R1 with CoT

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
    PROCEDURE_CACHE_ENABLED = os.getenv("EXAMINA_PROCEDURE_CACHE_ENABLED", "true").lower() == "true"
    PROCEDURE_CACHE_MIN_CONFIDENCE = float(
        os.getenv("EXAMINA_PROCEDURE_CACHE_MIN_CONFIDENCE", "0.85")
    )
    PROCEDURE_CACHE_SCOPE = os.getenv("EXAMINA_PROCEDURE_CACHE_SCOPE", "course")
    PROCEDURE_CACHE_EMBEDDING_THRESHOLD = float(
        os.getenv("EXAMINA_PROCEDURE_CACHE_EMBEDDING_THRESHOLD", "0.90")
    )
    PROCEDURE_CACHE_TEXT_VALIDATION_THRESHOLD = float(
        os.getenv("EXAMINA_PROCEDURE_CACHE_TEXT_THRESHOLD", "0.70")
    )
    PROCEDURE_CACHE_PRELOAD = os.getenv("EXAMINA_PROCEDURE_CACHE_PRELOAD", "true").lower() == "true"

    # Analysis Settings
    MIN_EXERCISES_FOR_KNOWLEDGE_ITEM = 2
    KNOWLEDGE_ITEM_SIMILARITY_THRESHOLD = 0.85
    MIN_ANALYSIS_CONFIDENCE = float(os.getenv("EXAMINA_MIN_CONFIDENCE", "0.5"))

    # Semantic Similarity Settings
    SEMANTIC_SIMILARITY_ENABLED = os.getenv("EXAMINA_SEMANTIC_ENABLED", "true").lower() == "true"
    SEMANTIC_SIMILARITY_THRESHOLD = float(os.getenv("EXAMINA_SEMANTIC_THRESHOLD", "0.85"))
    SEMANTIC_LOG_NEAR_MISSES = os.getenv("EXAMINA_LOG_NEAR_MISSES", "true").lower() == "true"
    SEMANTIC_EMBEDDING_MODEL = os.getenv(
        "EXAMINA_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Study Strategy Settings
    STUDY_STRATEGY_CACHE_DIR = DATA_DIR / "strategy_cache"
    STUDY_STRATEGY_CACHE_ENABLED = True

    # Supported languages
    SUPPORTED_LANGUAGES = ["it", "en"]
    DEFAULT_LANGUAGE = os.getenv("EXAMINA_LANGUAGE", "en")

    # Topic Splitting Settings
    GENERIC_TOPIC_THRESHOLD = int(os.getenv("EXAMINA_GENERIC_TOPIC_THRESHOLD", "10"))
    TOPIC_CLUSTER_MIN = 4
    TOPIC_CLUSTER_MAX = 6
    TOPIC_SPLITTING_ENABLED = os.getenv("EXAMINA_TOPIC_SPLITTING_ENABLED", "true").lower() == "true"

    # Bilingual Translation Detection Settings
    TRANSLATION_DETECTION_ENABLED = (
        os.getenv("EXAMINA_TRANSLATION_ENABLED", "true").lower() == "true"
    )
    TRANSLATION_DETECTION_THRESHOLD = float(os.getenv("EXAMINA_TRANSLATION_THRESHOLD", "0.70"))
    PREFERRED_LANGUAGES = ["english", "en"]

    # Language Detection Settings
    LANGUAGE_DETECTION_ENABLED = (
        os.getenv("EXAMINA_LANGUAGE_DETECTION_ENABLED", "true").lower() == "true"
    )
    AUTO_MERGE_TRANSLATIONS = os.getenv("EXAMINA_AUTO_MERGE_TRANSLATIONS", "true").lower() == "true"
    LANGUAGE_CACHE_TTL = int(os.getenv("EXAMINA_LANGUAGE_CACHE_TTL", "86400"))

    # Monolingual Mode Settings
    MONOLINGUAL_MODE_ENABLED = os.getenv("EXAMINA_MONOLINGUAL_ENABLED", "false").lower() == "true"

    # Smart Splitting Settings
    SMART_SPLIT_ENABLED = os.getenv("EXAMINA_SMART_SPLIT_ENABLED", "false").lower() == "true"
    SMART_SPLIT_CONFIDENCE_THRESHOLD = float(os.getenv("EXAMINA_SMART_SPLIT_THRESHOLD", "0.7"))
    SMART_SPLIT_MAX_PAGES = int(os.getenv("EXAMINA_SMART_SPLIT_MAX_PAGES", "20"))
    SMART_SPLIT_CACHE_ENABLED = os.getenv("EXAMINA_SMART_SPLIT_CACHE", "true").lower() == "true"

    # Phase 10: Learning Materials Settings
    LEARNING_MATERIALS_ENABLED = (
        os.getenv("EXAMINA_LEARNING_MATERIALS_ENABLED", "true").lower() == "true"
    )
    SHOW_THEORY_BY_DEFAULT = os.getenv("EXAMINA_SHOW_THEORY", "true").lower() == "true"
    SHOW_WORKED_EXAMPLES_BY_DEFAULT = (
        os.getenv("EXAMINA_SHOW_WORKED_EXAMPLES", "true").lower() == "true"
    )
    MAX_THEORY_SECTIONS_IN_LEARN = int(os.getenv("EXAMINA_MAX_THEORY_SECTIONS", "3"))
    MAX_WORKED_EXAMPLES_IN_LEARN = int(os.getenv("EXAMINA_MAX_WORKED_EXAMPLES", "2"))
    MATERIAL_TOPIC_SIMILARITY_THRESHOLD = float(
        os.getenv("EXAMINA_MATERIAL_TOPIC_THRESHOLD", "0.85")
    )
    WORKED_EXAMPLE_EXERCISE_SIMILARITY_THRESHOLD = float(
        os.getenv("EXAMINA_WORKED_EXAMPLE_THRESHOLD", "0.70")
    )

    # Rate Limiting Settings
    PROVIDER_RATE_LIMITS = {
        "anthropic": {
            "requests_per_minute": int(os.getenv("ANTHROPIC_RPM", "50")),
            "tokens_per_minute": int(os.getenv("ANTHROPIC_TPM", "40000")),
            "burst_size": 5,
        },
        "groq": {
            "requests_per_minute": int(os.getenv("GROQ_RPM", "30")),
            "tokens_per_minute": int(os.getenv("GROQ_TPM", "6000")),
            "burst_size": 3,
        },
        "ollama": {"requests_per_minute": None, "tokens_per_minute": None, "burst_size": 1},
        "openai": {
            "requests_per_minute": int(os.getenv("OPENAI_RPM", "60")),
            "tokens_per_minute": int(os.getenv("OPENAI_TPM", "90000")),
            "burst_size": 5,
        },
        "deepseek": {"requests_per_minute": None, "tokens_per_minute": None, "burst_size": 1},
    }

    @classmethod
    def ensure_dirs(cls):
        """Create all necessary directories."""
        cls.DATA_DIR.mkdir(exist_ok=True, parents=True)
        cls.FILES_PATH.mkdir(exist_ok=True, parents=True)
        cls.PDFS_PATH.mkdir(exist_ok=True, parents=True)
        cls.IMAGES_PATH.mkdir(exist_ok=True, parents=True)
        cls.CHROMA_PATH.mkdir(exist_ok=True, parents=True)
        cls.CACHE_PATH.mkdir(exist_ok=True, parents=True)
        cls.STUDY_STRATEGY_CACHE_DIR.mkdir(exist_ok=True, parents=True)

        # Ensure config directory exists for provider profiles
        config_dir = cls.BASE_DIR / "config"
        config_dir.mkdir(exist_ok=True, parents=True)

        # Set provider profiles path if not already set
        if cls.PROVIDER_PROFILES_PATH is None:
            cls.PROVIDER_PROFILES_PATH = config_dir / "provider_profiles.yaml"

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
