"""
Provider Router for Task-Based LLM Provider Selection.

This module implements intelligent routing of tasks to appropriate LLM providers
based on user profiles, task types, and provider availability.

Architecture:
- Profile-driven: All routing rules defined in YAML configuration
- Task-based: Routes based on task category (bulk/interactive/premium)
- Fallback-aware: Automatic fallback with explicit logging
- Thread-safe: Safe for concurrent operations
- Stateless: No per-request state (integrates with RateLimitTracker)

Design Goals:
1. Optimize cost by routing bulk operations to cheaper providers
2. Maintain quality for interactive/premium tasks
3. Respect privacy by minimizing provider exposure
4. Handle failures gracefully with fallback providers
5. Enable future multi-tenant SaaS (profile per user)
"""

import yaml
import logging
import threading
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass

from core.task_types import TaskType
from config import Config

logger = logging.getLogger(__name__)


@dataclass
class TaskConfig:
    """Configuration for a specific task type within a profile."""

    enabled: bool
    primary: Optional[str]  # Primary provider name
    fallback: Optional[str]  # Fallback provider name (None = no fallback)
    max_retries: int = 3  # Maximum retries before fallback
    retry_delay: float = 2.0  # Delay between retries (seconds)

    def __post_init__(self):
        """Validate task configuration."""
        if self.enabled and not self.primary:
            raise ValueError("enabled=true requires a primary provider")


@dataclass
class ProviderProfile:
    """Complete profile defining provider routing for all task types."""

    name: str
    description: str
    tasks: Dict[str, TaskConfig]  # Maps task_type -> TaskConfig

    def get_task_config(self, task_type: TaskType) -> TaskConfig:
        """Get configuration for a specific task type.

        Args:
            task_type: Task type to get config for

        Returns:
            TaskConfig for this task type

        Raises:
            ValueError: If task type not configured or disabled
        """
        task_key = task_type.value
        if task_key not in self.tasks:
            raise ValueError(
                f"Task type '{task_type.value}' not configured in profile '{self.name}'"
            )

        config = self.tasks[task_key]
        if not config.enabled:
            raise ValueError(
                f"Task type '{task_type.value}' is disabled in profile '{self.name}'. "
                f"Upgrade to a higher tier profile to access this feature."
            )

        return config


class ProviderRouter:
    """
    Routes tasks to appropriate LLM providers based on profiles.

    This class implements the core routing logic for the Provider Routing Architecture.
    It loads profile configurations from YAML, checks provider availability,
    handles fallbacks, and logs all routing decisions.

    Thread Safety:
        - Profile loading is protected by a lock
        - Routing operations are thread-safe
        - Integrates with thread-safe RateLimitTracker

    Usage:
        # Initialize router (typically done once at startup)
        router = ProviderRouter()

        # Route a task
        provider = router.route(TaskType.BULK_ANALYSIS, "free")
        llm = LLMManager(provider=provider)

        # Route with custom profile
        provider = router.route(TaskType.INTERACTIVE, "pro")
    """

    def __init__(self, profiles_path: Optional[Path] = None):
        """Initialize provider router.

        Args:
            profiles_path: Path to provider profiles YAML file.
                          Defaults to Config.PROVIDER_PROFILES_PATH
        """
        self.profiles_path = profiles_path or self._get_default_profiles_path()
        self.profiles: Dict[str, ProviderProfile] = {}
        self._lock = threading.RLock()

        # Load profiles
        self._load_profiles()

        logger.info(f"ProviderRouter initialized with {len(self.profiles)} profiles")

    def _get_default_profiles_path(self) -> Path:
        """Get default path for provider profiles configuration."""
        return Config.BASE_DIR / "config" / "provider_profiles.yaml"

    def _load_profiles(self):
        """Load provider profiles from YAML configuration.

        Raises:
            FileNotFoundError: If profiles file not found
            ValueError: If profiles file is invalid
        """
        with self._lock:
            if not self.profiles_path.exists():
                raise FileNotFoundError(
                    f"Provider profiles configuration not found: {self.profiles_path}\n"
                    f"Please create the configuration file with profile definitions."
                )

            try:
                with open(self.profiles_path, "r") as f:
                    config_data = yaml.safe_load(f)

                if not config_data or "profiles" not in config_data:
                    raise ValueError("Invalid profiles configuration: missing 'profiles' key")

                # Parse profiles
                for profile_name, profile_data in config_data["profiles"].items():
                    tasks = {}

                    # Parse task configurations
                    for task_name, task_data in profile_data.get("tasks", {}).items():
                        # Validate task_name is a valid TaskType
                        try:
                            TaskType.from_string(task_name)
                        except ValueError as e:
                            logger.warning(
                                f"Skipping invalid task type '{task_name}' in profile '{profile_name}': {e}"
                            )
                            continue

                        # Parse task config
                        tasks[task_name] = TaskConfig(
                            enabled=task_data.get("enabled", True),
                            primary=task_data.get("primary"),
                            fallback=task_data.get("fallback"),
                            max_retries=task_data.get("max_retries", 3),
                            retry_delay=task_data.get("retry_delay", 2.0),
                        )

                    # Create profile
                    self.profiles[profile_name] = ProviderProfile(
                        name=profile_name,
                        description=profile_data.get("description", ""),
                        tasks=tasks,
                    )

                logger.info(
                    f"Loaded {len(self.profiles)} provider profiles from {self.profiles_path}"
                )

            except yaml.YAMLError as e:
                raise ValueError(f"Failed to parse provider profiles YAML: {e}")
            except Exception as e:
                raise ValueError(f"Failed to load provider profiles: {e}")

    def reload_profiles(self):
        """Reload profiles from configuration file.

        Useful for hot-reloading configuration changes without restarting.
        """
        logger.info("Reloading provider profiles...")
        self._load_profiles()

    def route(self, task_type: TaskType, profile_name: str) -> str:
        """Route a task to the appropriate provider.

        This is the main entry point for provider routing. It:
        1. Loads the profile configuration
        2. Gets the task configuration
        3. Checks if primary provider is available
        4. Falls back if primary unavailable
        5. Logs all routing decisions

        Args:
            task_type: Type of task to route
            profile_name: Profile name (e.g., "free", "pro", "local")

        Returns:
            Provider name to use (e.g., "anthropic", "groq", "deepseek")

        Raises:
            ValueError: If profile not found, task disabled, or no providers available
        """
        # Get profile
        if profile_name not in self.profiles:
            available = ", ".join(self.profiles.keys())
            raise ValueError(f"Profile '{profile_name}' not found. Available profiles: {available}")

        profile = self.profiles[profile_name]

        # Get task configuration
        try:
            task_config = profile.get_task_config(task_type)
        except ValueError as e:
            logger.error(f"[ROUTING ERROR] {e}")
            raise

        # Try primary provider
        primary = task_config.primary
        logger.info(
            f"[ROUTING] Task: {task_type.value}, Profile: {profile_name}, Primary: {primary}"
        )

        if self._is_provider_available(primary):
            logger.info(f"[ROUTING] {primary.title()} available, using as primary")
            return primary

        # Primary unavailable, try fallback
        fallback = task_config.fallback

        if fallback is None:
            logger.error(
                f"[ROUTING ERROR] Primary provider '{primary}' unavailable and no fallback configured"
            )
            raise ValueError(
                f"Primary provider '{primary}' is unavailable for task '{task_type.value}' "
                f"in profile '{profile_name}', and no fallback is configured."
            )

        # Execute fallback
        logger.warning(
            f"[FALLBACK] Primary provider '{primary}' unavailable, attempting fallback to '{fallback}'"
        )

        return self._execute_fallback(task_type, profile_name, primary, fallback)

    def _execute_fallback(
        self, task_type: TaskType, profile_name: str, primary: str, fallback: str
    ) -> str:
        """Execute fallback to alternative provider.

        Args:
            task_type: Task type being routed
            profile_name: Profile name
            primary: Primary provider that failed
            fallback: Fallback provider to try

        Returns:
            Fallback provider name

        Raises:
            ValueError: If fallback provider also unavailable
        """
        if self._is_provider_available(fallback):
            logger.warning(
                f"[FALLBACK SUCCESS] Using '{fallback}' as fallback for '{primary}' "
                f"(task: {task_type.value}, profile: {profile_name})"
            )
            logger.warning(f"[WARNING] Using fallback provider may affect response quality or cost")
            return fallback

        # Both primary and fallback unavailable
        logger.error(
            f"[FALLBACK FAILED] Both primary ('{primary}') and fallback ('{fallback}') "
            f"unavailable for task '{task_type.value}' in profile '{profile_name}'"
        )
        raise ValueError(
            f"No available providers for task '{task_type.value}' in profile '{profile_name}'. "
            f"Primary '{primary}' and fallback '{fallback}' are both unavailable. "
            f"Please check API keys and provider status."
        )

    def _is_provider_available(self, provider: str) -> bool:
        """Check if a provider is available.

        Availability checks:
        1. API key exists (for cloud providers)
        2. Rate limits not exceeded (checked by RateLimitTracker)
        3. Provider service is reachable (future: health check)

        Args:
            provider: Provider name (e.g., "anthropic", "groq")

        Returns:
            True if provider is available, False otherwise
        """
        # Special case: ollama has no API key requirement
        if provider == "ollama":
            # TODO: Could add connectivity check to Ollama server
            return True

        # Check for API key
        api_key = self._get_provider_api_key(provider)
        if not api_key:
            logger.debug(f"[AVAILABILITY] Provider '{provider}' unavailable: no API key")
            return False

        # TODO: Could add rate limit check here by querying RateLimitTracker
        # For now, we rely on RateLimitTracker to handle rate limiting
        # during the actual request in LLMManager

        # TODO: Could add health check (ping provider API)
        # This would add latency but could catch service outages

        return True

    def _get_provider_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider.

        Args:
            provider: Provider name

        Returns:
            API key or None if not configured
        """
        key_map = {
            "anthropic": Config.ANTHROPIC_API_KEY,
            "groq": Config.GROQ_API_KEY,
            "openai": Config.OPENAI_API_KEY,
            "ollama": None,  # No key needed for local Ollama
        }

        # Support alias: deepseek uses groq
        if provider == "deepseek":
            provider = "groq"

        return key_map.get(provider)

    def get_profile(self, profile_name: str) -> ProviderProfile:
        """Get a profile by name.

        Args:
            profile_name: Profile name

        Returns:
            ProviderProfile

        Raises:
            ValueError: If profile not found
        """
        if profile_name not in self.profiles:
            available = ", ".join(self.profiles.keys())
            raise ValueError(f"Profile '{profile_name}' not found. Available profiles: {available}")

        return self.profiles[profile_name]

    def list_profiles(self) -> List[str]:
        """List all available profile names.

        Returns:
            List of profile names
        """
        return list(self.profiles.keys())

    def get_profile_info(self, profile_name: str) -> Dict[str, Any]:
        """Get detailed information about a profile.

        Args:
            profile_name: Profile name

        Returns:
            Dict with profile information

        Raises:
            ValueError: If profile not found
        """
        profile = self.get_profile(profile_name)

        tasks_info = {}
        for task_name, task_config in profile.tasks.items():
            tasks_info[task_name] = {
                "enabled": task_config.enabled,
                "primary": task_config.primary,
                "fallback": task_config.fallback,
                "max_retries": task_config.max_retries,
                "retry_delay": task_config.retry_delay,
            }

        return {"name": profile.name, "description": profile.description, "tasks": tasks_info}

    def validate_profiles(self) -> Dict[str, List[str]]:
        """Validate all profiles and return any issues.

        Returns:
            Dict mapping profile names to list of validation issues.
            Empty list means profile is valid.
        """
        issues = {}

        for profile_name, profile in self.profiles.items():
            profile_issues = []

            # Check that all task types are configured
            for task_type in TaskType:
                if task_type.value not in profile.tasks:
                    profile_issues.append(
                        f"Missing configuration for task type '{task_type.value}'"
                    )

            # Check that enabled tasks have valid providers
            for task_name, task_config in profile.tasks.items():
                if task_config.enabled:
                    # Check primary provider
                    if not task_config.primary:
                        profile_issues.append(
                            f"Task '{task_name}' is enabled but has no primary provider"
                        )
                    elif not self._is_provider_available(task_config.primary):
                        profile_issues.append(
                            f"Task '{task_name}' primary provider '{task_config.primary}' is not available"
                        )

                    # Check fallback provider if specified
                    if task_config.fallback and not self._is_provider_available(
                        task_config.fallback
                    ):
                        profile_issues.append(
                            f"Task '{task_name}' fallback provider '{task_config.fallback}' is not available"
                        )

            issues[profile_name] = profile_issues

        return issues
