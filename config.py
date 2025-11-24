"""
Configuration settings for Examina.
"""

import os
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)  # Override system env vars with .env values
except ImportError:
    # python-dotenv not installed, will use system environment variables only
    pass

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
    LLM_PROVIDER = os.getenv("EXAMINA_LLM_PROVIDER", "anthropic")

    # Provider Routing Settings (NEW - Provider Routing Architecture)
    # Default provider profile to use when --profile flag not specified
    PROVIDER_PROFILE = os.getenv("EXAMINA_PROVIDER_PROFILE", "free")  # Options: free, pro, local
    PROVIDER_PROFILES_PATH = None  # Will be set to BASE_DIR / "config" / "provider_profiles.yaml" below

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
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")  # Best rate limits on Groq free tier

    # DeepSeek Settings
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")  # DeepSeek v3 (671B MoE)

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

    # Analysis Settings
    MIN_EXERCISES_FOR_CORE_LOOP = 2  # Minimum exercises to establish a core loop
    CORE_LOOP_SIMILARITY_THRESHOLD = 0.85  # Similarity threshold for merging core loops (string-based)
    MIN_ANALYSIS_CONFIDENCE = float(os.getenv("EXAMINA_MIN_CONFIDENCE", "0.5"))  # Minimum confidence for analysis results

    # Semantic Similarity Settings (NEW)
    SEMANTIC_SIMILARITY_ENABLED = os.getenv("EXAMINA_SEMANTIC_ENABLED", "true").lower() == "true"  # Enable semantic matching
    SEMANTIC_SIMILARITY_THRESHOLD = float(os.getenv("EXAMINA_SEMANTIC_THRESHOLD", "0.85"))  # Threshold for semantic similarity
    SEMANTIC_LOG_NEAR_MISSES = os.getenv("EXAMINA_LOG_NEAR_MISSES", "true").lower() == "true"  # Log items with high similarity but semantic difference

    # Study Strategy Settings
    STUDY_STRATEGY_CACHE_DIR = DATA_DIR / "strategy_cache"  # Cache directory for generated strategies
    STUDY_STRATEGY_CACHE_ENABLED = True  # Cache generated strategies for reuse

    # Supported languages
    SUPPORTED_LANGUAGES = ["it", "en"]  # Italian and English
    DEFAULT_LANGUAGE = os.getenv("EXAMINA_LANGUAGE", "en")  # Default to English

    # Topic Splitting Settings
    GENERIC_TOPIC_THRESHOLD = int(os.getenv("EXAMINA_GENERIC_TOPIC_THRESHOLD", "10"))  # Min core loops to trigger splitting
    TOPIC_CLUSTER_MIN = 4  # Minimum subtopics to create when splitting
    TOPIC_CLUSTER_MAX = 6  # Maximum subtopics to create when splitting
    TOPIC_SPLITTING_ENABLED = os.getenv("EXAMINA_TOPIC_SPLITTING_ENABLED", "true").lower() == "true"  # Enable automatic splitting

    # Bilingual Translation Detection Settings (NEW - Phase: Generic Bilingual Deduplication)
    # LLM-based translation detection for deduplication across ANY language pair (IT/EN, ES/EN, FR/EN, etc.)
    # Replaces hardcoded translation dictionaries with dynamic, language-agnostic detection
    TRANSLATION_DETECTION_ENABLED = os.getenv("EXAMINA_TRANSLATION_ENABLED", "true").lower() == "true"  # Enable LLM-based translation detection
    TRANSLATION_DETECTION_THRESHOLD = float(os.getenv("EXAMINA_TRANSLATION_THRESHOLD", "0.70"))  # Min embedding similarity before LLM check
    PREFERRED_LANGUAGES = ["english", "en"]  # Prefer these languages when merging translations (most universal)

    # Language Detection Settings (Phase: Automatic Language Detection)
    # Automatic language detection for procedures and topics
    # Enables cross-language merging and language-aware search
    LANGUAGE_DETECTION_ENABLED = os.getenv("EXAMINA_LANGUAGE_DETECTION_ENABLED", "true").lower() == "true"  # Auto-detect during analysis
    AUTO_MERGE_TRANSLATIONS = os.getenv("EXAMINA_AUTO_MERGE_TRANSLATIONS", "true").lower() == "true"  # Merge cross-language duplicates automatically
    LANGUAGE_CACHE_TTL = int(os.getenv("EXAMINA_LANGUAGE_CACHE_TTL", "86400"))  # Cache language detection for 24 hours

    # Monolingual Mode Settings (Phase 6 TODO)
    # Strictly monolingual analysis - ensure all procedures are in single language
    # Prevents cross-language duplicates by detecting and translating procedures to primary language
    MONOLINGUAL_MODE_ENABLED = os.getenv("EXAMINA_MONOLINGUAL_ENABLED", "false").lower() == "true"  # Default: OFF (bilingual mode)

    # Smart Splitting Settings (Phase: Enhanced Exercise Splitting)
    # LLM-based exercise detection for unstructured materials (lecture notes, embedded examples)
    # Complements pattern-based splitting with AI detection for edge cases
    SMART_SPLIT_ENABLED = os.getenv("EXAMINA_SMART_SPLIT_ENABLED", "false").lower() == "true"  # Enable LLM-based splitting (opt-in via --smart-split flag)
    SMART_SPLIT_CONFIDENCE_THRESHOLD = float(os.getenv("EXAMINA_SMART_SPLIT_THRESHOLD", "0.7"))  # Min confidence for detected exercises
    SMART_SPLIT_MAX_PAGES = int(os.getenv("EXAMINA_SMART_SPLIT_MAX_PAGES", "50"))  # Cost control: max pages to process with LLM
    SMART_SPLIT_CACHE_ENABLED = os.getenv("EXAMINA_SMART_SPLIT_CACHE", "true").lower() == "true"  # Cache LLM detection results

    # Phase 10: Learning Materials Settings
    # Configuration for theory sections, worked examples, and learning flow
    LEARNING_MATERIALS_ENABLED = os.getenv("EXAMINA_LEARNING_MATERIALS_ENABLED", "true").lower() == "true"  # Enable learning materials feature
    SHOW_THEORY_BY_DEFAULT = os.getenv("EXAMINA_SHOW_THEORY", "true").lower() == "true"  # Show theory materials in learn command by default
    SHOW_WORKED_EXAMPLES_BY_DEFAULT = os.getenv("EXAMINA_SHOW_WORKED_EXAMPLES", "true").lower() == "true"  # Show worked examples by default
    MAX_THEORY_SECTIONS_IN_LEARN = int(os.getenv("EXAMINA_MAX_THEORY_SECTIONS", "3"))  # Max theory sections to show in learn command
    MAX_WORKED_EXAMPLES_IN_LEARN = int(os.getenv("EXAMINA_MAX_WORKED_EXAMPLES", "2"))  # Max worked examples to show in learn command
    MATERIAL_TOPIC_SIMILARITY_THRESHOLD = float(os.getenv("EXAMINA_MATERIAL_TOPIC_THRESHOLD", "0.85"))  # Similarity threshold for linking materials to topics
    WORKED_EXAMPLE_EXERCISE_SIMILARITY_THRESHOLD = float(os.getenv("EXAMINA_WORKED_EXAMPLE_THRESHOLD", "0.70"))  # Similarity threshold for linking worked examples to exercises

    # Rate Limiting Settings
    # Provider-agnostic rate limits (configurable per provider)
    # Set to None for unlimited (e.g., local providers like Ollama)
    PROVIDER_RATE_LIMITS = {
        "anthropic": {
            "requests_per_minute": int(os.getenv("ANTHROPIC_RPM", "50")),
            "tokens_per_minute": int(os.getenv("ANTHROPIC_TPM", "40000")),
            "burst_size": 5  # Allow small bursts
        },
        "groq": {
            "requests_per_minute": int(os.getenv("GROQ_RPM", "30")),  # Free tier limit
            "tokens_per_minute": int(os.getenv("GROQ_TPM", "6000")),  # Free tier limit
            "burst_size": 3
        },
        "ollama": {
            "requests_per_minute": None,  # No limit (local)
            "tokens_per_minute": None,
            "burst_size": 1
        },
        "openai": {
            "requests_per_minute": int(os.getenv("OPENAI_RPM", "60")),
            "tokens_per_minute": int(os.getenv("OPENAI_TPM", "90000")),
            "burst_size": 5
        },
        "deepseek": {
            "requests_per_minute": None,  # No rate limit (DeepSeek has very high/no limits)
            "tokens_per_minute": None,  # No rate limit
            "burst_size": 1
        }
        # Add more providers as needed - the system is fully generic
    }

    @classmethod
    def ensure_dirs(cls):
        """Create all necessary directories."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.FILES_PATH.mkdir(exist_ok=True)
        cls.PDFS_PATH.mkdir(exist_ok=True)
        cls.IMAGES_PATH.mkdir(exist_ok=True)
        cls.CHROMA_PATH.mkdir(exist_ok=True)
        cls.CACHE_PATH.mkdir(exist_ok=True)
        cls.STUDY_STRATEGY_CACHE_DIR.mkdir(exist_ok=True)

        # Ensure config directory exists for provider profiles
        config_dir = cls.BASE_DIR / "config"
        config_dir.mkdir(exist_ok=True)

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
