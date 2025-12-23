"""
Generic rate limiting tracker for LLM API providers.

This module provides provider-agnostic rate limiting with:
- Sliding window tracking
- Request and token counting
- Thread-safe operations
- Persistent caching across CLI runs
- Configurable limits per provider
"""

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class UsageWindow:
    """Tracks usage within a time window."""

    requests: deque  # Timestamps of requests
    tokens: deque  # (timestamp, token_count) tuples
    last_reset: float  # Last reset timestamp

    def __init__(self):
        self.requests = deque()
        self.tokens = deque()
        self.last_reset = time.time()


@dataclass
class ProviderLimits:
    """Rate limits for a provider."""

    requests_per_minute: Optional[int] = None
    tokens_per_minute: Optional[int] = None
    burst_size: int = 1

    def has_limits(self) -> bool:
        """Check if provider has any rate limits."""
        return self.requests_per_minute is not None or self.tokens_per_minute is not None


class RateLimitTracker:
    """Generic rate limiting tracker for any LLM provider.

    Features:
    - Sliding window rate limiting (not fixed minute boundaries)
    - Thread-safe operations
    - Persistent state across CLI runs
    - Configurable per-provider limits
    - Automatic cleanup of old entries

    Example:
        limits = {
            "anthropic": {"requests_per_minute": 50, "tokens_per_minute": 40000},
            "groq": {"requests_per_minute": 30, "tokens_per_minute": 6000},
            "ollama": {"requests_per_minute": None, "tokens_per_minute": None}
        }
        tracker = RateLimitTracker(limits)

        # Before making request
        wait_time = tracker.wait_if_needed("groq")

        # After request completes
        tracker.record_request("groq", tokens_used=150)
    """

    def __init__(self, provider_limits: Dict[str, Dict], cache_path: Optional[Path] = None):
        """Initialize rate limiter.

        Args:
            provider_limits: Dict mapping provider names to their limits.
                Example: {
                    "anthropic": {"requests_per_minute": 50, "tokens_per_minute": 40000},
                    "groq": {"requests_per_minute": 30, "tokens_per_minute": 6000}
                }
            cache_path: Optional path for persistent cache (default: data/cache/rate_limits.json)
        """
        self.limits = {name: ProviderLimits(**limits) for name, limits in provider_limits.items()}
        self.usage: Dict[str, UsageWindow] = {}
        self.lock = threading.RLock()

        # Setup cache
        if cache_path is None:
            from config import Config

            Config.ensure_dirs()
            cache_path = Config.CACHE_PATH / "rate_limits.json"
        self.cache_path = cache_path

        # Load cached state
        self._load_cache()

    def _load_cache(self):
        """Load cached usage data from disk."""
        if not self.cache_path.exists():
            return

        try:
            with open(self.cache_path, "r") as f:
                cache_data = json.load(f)

            current_time = time.time()

            # Restore usage windows for each provider
            for provider, data in cache_data.items():
                if provider not in self.limits:
                    continue

                window = UsageWindow()

                # Restore requests (filter out old ones)
                for timestamp in data.get("requests", []):
                    if current_time - timestamp < 60:  # Only keep last minute
                        window.requests.append(timestamp)

                # Restore tokens (filter out old ones)
                for entry in data.get("tokens", []):
                    timestamp, count = entry
                    if current_time - timestamp < 60:
                        window.tokens.append((timestamp, count))

                window.last_reset = data.get("last_reset", current_time)
                self.usage[provider] = window

            logger.debug(f"Loaded rate limit cache from {self.cache_path}")
        except Exception as e:
            logger.warning(f"Failed to load rate limit cache: {e}")

    def _save_cache(self):
        """Save usage data to disk for persistence."""
        try:
            cache_data = {}

            for provider, window in self.usage.items():
                cache_data[provider] = {
                    "requests": list(window.requests),
                    "tokens": list(window.tokens),
                    "last_reset": window.last_reset,
                }

            with open(self.cache_path, "w") as f:
                json.dump(cache_data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save rate limit cache: {e}")

    def _get_or_create_window(self, provider: str) -> UsageWindow:
        """Get or create usage window for provider."""
        if provider not in self.usage:
            self.usage[provider] = UsageWindow()
        return self.usage[provider]

    def _cleanup_old_entries(self, window: UsageWindow, current_time: float):
        """Remove entries older than 60 seconds."""
        cutoff_time = current_time - 60

        # Clean requests
        while window.requests and window.requests[0] < cutoff_time:
            window.requests.popleft()

        # Clean tokens
        while window.tokens and window.tokens[0][0] < cutoff_time:
            window.tokens.popleft()

    def check_limit(self, provider: str) -> bool:
        """Check if provider is within rate limits.

        Args:
            provider: Provider name

        Returns:
            True if within limits, False if would exceed limits
        """
        with self.lock:
            # Check if provider has limits configured
            if provider not in self.limits:
                logger.debug(f"Provider '{provider}' not in configuration, skipping rate limit")
                return True

            limits = self.limits[provider]
            if not limits.has_limits():
                return True  # No limits

            window = self._get_or_create_window(provider)
            current_time = time.time()

            # Clean old entries
            self._cleanup_old_entries(window, current_time)

            # Check request limit
            if limits.requests_per_minute is not None:
                if len(window.requests) >= limits.requests_per_minute:
                    return False

            # Check token limit
            if limits.tokens_per_minute is not None:
                total_tokens = sum(count for _, count in window.tokens)
                if total_tokens >= limits.tokens_per_minute:
                    return False

            return True

    def record_request(self, provider: str, tokens_used: int = 0):
        """Record a request for rate tracking.

        Args:
            provider: Provider name
            tokens_used: Number of tokens used (if available)
        """
        with self.lock:
            if provider not in self.limits:
                return  # Skip if provider not configured

            window = self._get_or_create_window(provider)
            current_time = time.time()

            # Record request
            window.requests.append(current_time)

            # Record tokens if provided
            if tokens_used > 0:
                window.tokens.append((current_time, tokens_used))

            # Clean old entries
            self._cleanup_old_entries(window, current_time)

            # Save to cache
            self._save_cache()

    def wait_if_needed(self, provider: str) -> float:
        """Wait if rate limit would be exceeded. Returns wait time.

        Args:
            provider: Provider name

        Returns:
            Time waited in seconds (0 if no wait needed)
        """
        with self.lock:
            # Check if provider has limits
            if provider not in self.limits:
                return 0.0

            limits = self.limits[provider]
            if not limits.has_limits():
                return 0.0

            window = self._get_or_create_window(provider)
            current_time = time.time()

            # Clean old entries
            self._cleanup_old_entries(window, current_time)

            # Check if we need to wait
            if not self.check_limit(provider):
                # Calculate wait time
                oldest_timestamp = None

                # Check which limit we're hitting
                if (
                    limits.requests_per_minute
                    and len(window.requests) >= limits.requests_per_minute
                ):
                    oldest_timestamp = window.requests[0]

                if limits.tokens_per_minute:
                    total_tokens = sum(count for _, count in window.tokens)
                    if total_tokens >= limits.tokens_per_minute and window.tokens:
                        token_timestamp = window.tokens[0][0]
                        if oldest_timestamp is None or token_timestamp < oldest_timestamp:
                            oldest_timestamp = token_timestamp

                if oldest_timestamp:
                    # Wait until oldest entry expires (60 seconds old)
                    wait_time = 60 - (current_time - oldest_timestamp) + 0.1  # Add buffer
                    if wait_time > 0:
                        logger.info(
                            f"Rate limit reached for '{provider}', waiting {wait_time:.1f}s"
                        )
                        time.sleep(wait_time)
                        return wait_time

        return 0.0

    def get_usage_stats(self, provider: str) -> Dict[str, Any]:
        """Get current usage statistics.

        Args:
            provider: Provider name

        Returns:
            Dict with usage stats including current usage, limits, and remaining capacity
        """
        with self.lock:
            if provider not in self.limits:
                return {
                    "provider": provider,
                    "error": "Provider not configured",
                    "has_limits": False,
                }

            limits = self.limits[provider]

            if not limits.has_limits():
                return {
                    "provider": provider,
                    "has_limits": False,
                    "requests": {"limit": None, "used": 0, "remaining": "unlimited"},
                    "tokens": {"limit": None, "used": 0, "remaining": "unlimited"},
                }

            window = self._get_or_create_window(provider)
            current_time = time.time()

            # Clean old entries
            self._cleanup_old_entries(window, current_time)

            # Calculate stats
            requests_used = len(window.requests)
            tokens_used = sum(count for _, count in window.tokens)

            # Calculate time until reset (based on oldest entry)
            time_until_reset = 0.0
            if window.requests:
                oldest = window.requests[0]
                time_until_reset = max(0, 60 - (current_time - oldest))

            stats = {
                "provider": provider,
                "has_limits": True,
                "requests": {
                    "limit": limits.requests_per_minute,
                    "used": requests_used,
                    "remaining": limits.requests_per_minute - requests_used
                    if limits.requests_per_minute
                    else "unlimited",
                    "percentage": round(requests_used / limits.requests_per_minute * 100, 1)
                    if limits.requests_per_minute
                    else 0,
                },
                "tokens": {
                    "limit": limits.tokens_per_minute,
                    "used": tokens_used,
                    "remaining": limits.tokens_per_minute - tokens_used
                    if limits.tokens_per_minute
                    else "unlimited",
                    "percentage": round(tokens_used / limits.tokens_per_minute * 100, 1)
                    if limits.tokens_per_minute
                    else 0,
                },
                "time_until_reset": round(time_until_reset, 1),
                "burst_size": limits.burst_size,
            }

            return stats

    def reset(self, provider: str):
        """Reset tracking for a provider.

        Args:
            provider: Provider name
        """
        with self.lock:
            if provider in self.usage:
                self.usage[provider] = UsageWindow()
                self._save_cache()
                logger.info(f"Reset rate limits for '{provider}'")

    def reset_all(self):
        """Reset tracking for all providers."""
        with self.lock:
            self.usage.clear()
            self._save_cache()
            logger.info("Reset rate limits for all providers")

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get usage stats for all configured providers.

        Returns:
            Dict mapping provider names to their stats
        """
        return {provider: self.get_usage_stats(provider) for provider in self.limits.keys()}
