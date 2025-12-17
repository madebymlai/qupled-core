"""
LLM Manager for Examina.
Handles interactions with Ollama and other LLM providers.
"""

import json
import requests
import hashlib
import time
import logging
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from config import Config
# NOTE: RateLimitTracker imported lazily in __init__ to avoid circular import
# Chain: models.llm_manager → core.rate_limiter → core/__init__ → core.analyzer → models.llm_manager

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM."""

    text: str
    model: str
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMManager:
    """Manages LLM interactions for Examina."""

    def __init__(
        self, provider: str = "deepseek", base_url: Optional[str] = None, quiet: bool = False
    ):
        """Initialize LLM manager.

        Args:
            provider: LLM provider ("ollama", "anthropic", "openai", "groq", "deepseek", "openrouter")
            base_url: Base URL for API (for Ollama)
            quiet: If True, suppress cache hit/miss messages (use get_cache_stats() for summary)
        """
        self.provider = provider
        self.base_url = base_url or Config.OLLAMA_BASE_URL
        self.quiet = quiet

        # Model selection
        if provider == "groq":
            self.primary_model = Config.GROQ_MODEL
            self.fast_model = Config.GROQ_MODEL
            self.embed_model = Config.LLM_EMBED_MODEL  # Still use Ollama for embeddings
        elif provider == "anthropic":
            self.primary_model = Config.ANTHROPIC_MODEL
            self.fast_model = Config.ANTHROPIC_MODEL
            self.embed_model = Config.LLM_EMBED_MODEL  # Still use Ollama for embeddings
        elif provider == "deepseek":
            self.primary_model = Config.DEEPSEEK_MODEL
            self.fast_model = Config.DEEPSEEK_MODEL
            self.embed_model = Config.LLM_EMBED_MODEL  # Still use Ollama for embeddings
        elif provider == "openrouter":
            self.primary_model = Config.OPENROUTER_MODEL
            self.fast_model = Config.OPENROUTER_MODEL
            self.vision_model = Config.OPENROUTER_VISION_MODEL
            self.embed_model = Config.LLM_EMBED_MODEL  # Still use Ollama for embeddings
        else:
            self.primary_model = Config.LLM_PRIMARY_MODEL  # Heavy reasoning
            self.fast_model = Config.LLM_FAST_MODEL  # Quick tasks
            self.embed_model = Config.LLM_EMBED_MODEL  # Embeddings

        # Cache settings
        self.cache_enabled = Config.CACHE_ENABLED
        self.cache_ttl = Config.CACHE_TTL
        self.cache_dir = Config.CACHE_PATH / "llm"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0

        # Initialize rate limiter (lazy import to avoid circular dependency)
        from core.rate_limiter import RateLimitTracker

        self.rate_limiter = RateLimitTracker(Config.PROVIDER_RATE_LIMITS)
        logger.debug(f"Initialized LLMManager with provider '{provider}' and rate limiting")

        # Async HTTP session (initialized in __aenter__)
        self._session: Optional["aiohttp.ClientSession"] = None

    async def __aenter__(self):
        """Async context manager entry - initializes aiohttp session."""
        if not AIOHTTP_AVAILABLE:
            raise ImportError(
                "aiohttp is required for async operations. Install with: pip install aiohttp"
            )

        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - closes aiohttp session."""
        if self._session:
            await self._session.close()
            self._session = None

    def _generate_cache_key(
        self,
        provider: str,
        model: str,
        prompt: str,
        system: Optional[str],
        temperature: float,
        json_mode: bool,
    ) -> str:
        """Generate cache key from request parameters.

        Args:
            provider: LLM provider name
            model: Model name
            prompt: User prompt
            system: System prompt
            temperature: Sampling temperature
            json_mode: Whether JSON mode is enabled

        Returns:
            SHA256 hash as cache key
        """
        # Create deterministic string from all parameters that affect output
        cache_string = json.dumps(
            {
                "provider": provider,
                "model": model,
                "prompt": prompt,
                "system": system or "",
                "temperature": temperature,
                "json_mode": json_mode,
            },
            sort_keys=True,
        )

        # Generate hash
        return hashlib.sha256(cache_string.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[LLMResponse]:
        """Retrieve cached response if available and not expired.

        Args:
            cache_key: Cache key

        Returns:
            Cached LLMResponse or None if not found/expired
        """
        if not self.cache_enabled:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r") as f:
                cache_data = json.load(f)

            # Check TTL
            cached_time = cache_data.get("timestamp", 0)
            if time.time() - cached_time > self.cache_ttl:
                # Cache expired
                cache_file.unlink()  # Delete expired cache
                return None

            # Cache hit!
            self.cache_hits += 1
            if not self.quiet:
                print(f"  [CACHE HIT] Using cached response")

            return LLMResponse(
                text=cache_data.get("text", ""),
                model=cache_data.get("model", ""),
                success=cache_data.get("success", False),
                error=cache_data.get("error"),
                metadata=cache_data.get("metadata"),
            )

        except Exception as e:
            print(f"  [CACHE ERROR] Failed to read cache: {e}")
            return None

    def _save_to_cache(self, cache_key: str, response: LLMResponse):
        """Save response to cache.

        Args:
            cache_key: Cache key
            response: LLM response to cache
        """
        if not self.cache_enabled:
            return

        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            cache_data = {
                "timestamp": time.time(),
                "text": response.text,
                "model": response.model,
                "success": response.success,
                "error": response.error,
                "metadata": response.metadata,
            }

            with open(cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)

            if not self.quiet:
                print(f"  [CACHE MISS] Response cached for future use")
            self.cache_misses += 1

        except Exception as e:
            print(f"  [CACHE ERROR] Failed to save cache: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache hits, misses, and hit rate
        """
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0

        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "total": total,
            "hit_rate": round(hit_rate, 2),
        }

    def reset_cache_stats(self):
        """Reset cache statistics counters."""
        self.cache_hits = 0
        self.cache_misses = 0

    def get_rate_limit_stats(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Get rate limit statistics.

        Args:
            provider: Provider name (defaults to current provider)

        Returns:
            Dict with rate limit usage stats
        """
        provider = provider or self.provider
        return self.rate_limiter.get_usage_stats(provider)

    def get_all_rate_limit_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get rate limit statistics for all providers.

        Returns:
            Dict mapping provider names to their stats
        """
        return self.rate_limiter.get_all_stats()

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate text from LLM.

        Args:
            prompt: User prompt
            model: Model to use (defaults to fast_model)
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            json_mode: Force JSON output

        Returns:
            LLMResponse with generated text
        """
        model = model or self.fast_model

        # Apply rate limiting before making request
        wait_time = self.rate_limiter.wait_if_needed(self.provider)
        if wait_time > 0:
            print(
                f"  [RATE LIMIT] Waiting {wait_time:.1f}s for '{self.provider}' (rate limit protection)"
            )

        # Make the API call
        if self.provider == "ollama":
            response = self._ollama_generate(
                prompt, model, system, temperature, max_tokens, json_mode
            )
        elif self.provider == "groq":
            response = self._groq_generate(
                prompt, model, system, temperature, max_tokens, json_mode
            )
        elif self.provider == "anthropic":
            response = self._anthropic_generate(
                prompt, model, system, temperature, max_tokens, json_mode
            )
        elif self.provider == "deepseek":
            response = self._deepseek_generate(
                prompt, model, system, temperature, max_tokens, json_mode
            )
        elif self.provider == "openrouter":
            response = self._openrouter_generate(
                prompt, model, system, temperature, max_tokens, json_mode
            )
        else:
            response = LLMResponse(
                text="",
                model=model,
                success=False,
                error=f"Provider {self.provider} not implemented yet",
            )

        # Record usage if request was successful
        if response.success:
            tokens_used = 0
            if response.metadata:
                # Extract token count from metadata (provider-specific)
                usage = response.metadata.get("usage", {})
                if isinstance(usage, dict):
                    # Anthropic/Groq format
                    tokens_used = usage.get("total_tokens", 0)
                    if tokens_used == 0:
                        # Try input + output tokens
                        tokens_used = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

            self.rate_limiter.record_request(self.provider, tokens_used=tokens_used)

        return response

    async def generate_async(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate text from LLM asynchronously.

        Args:
            prompt: User prompt
            model: Model to use (defaults to fast_model)
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            json_mode: Force JSON output

        Returns:
            LLMResponse with generated text
        """
        model = model or self.fast_model

        # Apply rate limiting before making request
        wait_time = self.rate_limiter.wait_if_needed(self.provider)
        if wait_time > 0:
            print(
                f"  [RATE LIMIT] Waiting {wait_time:.1f}s for '{self.provider}' (rate limit protection)"
            )
            await asyncio.sleep(wait_time)

        # Make the API call
        if self.provider == "ollama":
            # Ollama doesn't have async implementation yet, fall back to sync
            response = self._ollama_generate(
                prompt, model, system, temperature, max_tokens, json_mode
            )
        elif self.provider == "groq":
            response = await self._groq_generate_async(
                prompt, model, system, temperature, max_tokens, json_mode
            )
        elif self.provider == "anthropic":
            response = await self._anthropic_generate_async(
                prompt, model, system, temperature, max_tokens, json_mode
            )
        elif self.provider == "deepseek":
            response = await self._deepseek_generate_async(
                prompt, model, system, temperature, max_tokens, json_mode
            )
        elif self.provider == "openrouter":
            response = await self._openrouter_generate_async(
                prompt, model, system, temperature, max_tokens, json_mode
            )
        else:
            response = LLMResponse(
                text="",
                model=model,
                success=False,
                error=f"Provider {self.provider} not implemented yet",
            )

        # Record usage if request was successful
        if response.success:
            tokens_used = 0
            if response.metadata:
                # Extract token count from metadata (provider-specific)
                usage = response.metadata.get("usage", {})
                if isinstance(usage, dict):
                    # Anthropic/Groq format
                    tokens_used = usage.get("total_tokens", 0)
                    if tokens_used == 0:
                        # Try input + output tokens
                        tokens_used = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

            self.rate_limiter.record_request(self.provider, tokens_used=tokens_used)

        return response

    def _ollama_generate(
        self,
        prompt: str,
        model: str,
        system: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        json_mode: bool,
    ) -> LLMResponse:
        """Generate using Ollama API.

        Args:
            prompt: User prompt
            model: Model name
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            json_mode: Force JSON output

        Returns:
            LLMResponse
        """
        try:
            url = f"{self.base_url}/api/generate"

            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                },
            }

            if system:
                payload["system"] = system

            if max_tokens:
                payload["options"]["num_predict"] = max_tokens

            if json_mode:
                payload["format"] = "json"

            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()

            result = response.json()

            return LLMResponse(
                text=result.get("response", ""),
                model=model,
                success=True,
                metadata={
                    "total_duration": result.get("total_duration"),
                    "load_duration": result.get("load_duration"),
                    "eval_count": result.get("eval_count"),
                },
            )

        except requests.exceptions.ConnectionError:
            return LLMResponse(
                text="",
                model=model,
                success=False,
                error="Cannot connect to Ollama. Is it running? (ollama serve)",
            )
        except requests.exceptions.Timeout:
            return LLMResponse(
                text="",
                model=model,
                success=False,
                error="Request timed out. Model might be too slow.",
            )
        except Exception as e:
            return LLMResponse(text="", model=model, success=False, error=f"Ollama error: {str(e)}")

    def _groq_generate(
        self,
        prompt: str,
        model: str,
        system: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        json_mode: bool,
    ) -> LLMResponse:
        """Generate using Groq API with automatic retry on rate limits.

        Args:
            prompt: User prompt
            model: Model name
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            json_mode: Force JSON output

        Returns:
            LLMResponse
        """
        import time

        # Check for API key upfront
        if not Config.GROQ_API_KEY:
            return LLMResponse(
                text="",
                model=model,
                success=False,
                error="GROQ_API_KEY not set. Get one at https://console.groq.com",
            )

        # Check cache first
        cache_key = self._generate_cache_key(
            provider="groq",
            model=model,
            prompt=prompt,
            system=system,
            temperature=temperature,
            json_mode=json_mode,
        )

        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response

        max_retries = 3
        base_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                url = "https://api.groq.com/openai/v1/chat/completions"

                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})

                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }

                if max_tokens:
                    payload["max_tokens"] = max_tokens

                if json_mode:
                    payload["response_format"] = {"type": "json_object"}

                headers = {
                    "Authorization": f"Bearer {Config.GROQ_API_KEY}",
                    "Content-Type": "application/json",
                }

                response = requests.post(url, json=payload, headers=headers, timeout=60)
                response.raise_for_status()

                result = response.json()
                text = result["choices"][0]["message"]["content"]

                llm_response = LLMResponse(
                    text=text,
                    model=model,
                    success=True,
                    metadata={
                        "usage": result.get("usage"),
                        "finish_reason": result["choices"][0].get("finish_reason"),
                    },
                )

                # Cache successful response
                self._save_to_cache(cache_key, llm_response)

                return llm_response

            except requests.exceptions.HTTPError as e:
                error_msg = f"Groq API error: {e}"
                if e.response.status_code == 401:
                    error_msg = "Invalid GROQ_API_KEY. Check your API key."
                    # Don't retry on auth errors
                    return LLMResponse(text="", model=model, success=False, error=error_msg)
                elif e.response.status_code == 429:
                    # Rate limit - retry with backoff
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)  # Exponential backoff
                        print(
                            f"  Rate limit hit, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        error_msg = "Rate limit exceeded. All retries exhausted."
                elif e.response.status_code == 400:
                    try:
                        error_detail = e.response.json()
                        error_msg = f"Groq API error: {error_detail}"
                    except:
                        error_msg = f"Groq API error: {e} - {e.response.text}"

                # Return error if not retrying
                return LLMResponse(text="", model=model, success=False, error=error_msg)
            except Exception as e:
                return LLMResponse(
                    text="", model=model, success=False, error=f"Groq error: {str(e)}"
                )

        # Should never reach here, but just in case
        return LLMResponse(text="", model=model, success=False, error="Max retries exceeded")

    async def _groq_generate_async(
        self,
        prompt: str,
        model: str,
        system: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        json_mode: bool,
    ) -> LLMResponse:
        """Generate using Groq API asynchronously with automatic retry on rate limits.

        Args:
            prompt: User prompt
            model: Model name
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            json_mode: Force JSON output

        Returns:
            LLMResponse
        """
        # Check for API key upfront
        if not Config.GROQ_API_KEY:
            return LLMResponse(
                text="",
                model=model,
                success=False,
                error="GROQ_API_KEY not set. Get one at https://console.groq.com",
            )

        # Check cache first
        cache_key = self._generate_cache_key(
            provider="groq",
            model=model,
            prompt=prompt,
            system=system,
            temperature=temperature,
            json_mode=json_mode,
        )

        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response

        if not self._session:
            return LLMResponse(
                text="",
                model=model,
                success=False,
                error="Async session not initialized. Use 'async with LLMManager()' context manager.",
            )

        max_retries = 3
        base_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                url = "https://api.groq.com/openai/v1/chat/completions"

                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})

                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }

                if max_tokens:
                    payload["max_tokens"] = max_tokens

                if json_mode:
                    payload["response_format"] = {"type": "json_object"}

                headers = {
                    "Authorization": f"Bearer {Config.GROQ_API_KEY}",
                    "Content-Type": "application/json",
                }

                timeout = aiohttp.ClientTimeout(total=120)
                async with self._session.post(
                    url, json=payload, headers=headers, timeout=timeout
                ) as response:
                    if response.status == 401:
                        error_msg = "Invalid GROQ_API_KEY. Check your API key."
                        return LLMResponse(text="", model=model, success=False, error=error_msg)
                    elif response.status == 429:
                        # Rate limit - retry with backoff
                        if attempt < max_retries - 1:
                            delay = base_delay * (2**attempt)  # Exponential backoff
                            print(
                                f"  Rate limit hit, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})"
                            )
                            await asyncio.sleep(delay)
                            continue
                        else:
                            error_msg = "Rate limit exceeded. All retries exhausted."
                            return LLMResponse(text="", model=model, success=False, error=error_msg)
                    elif response.status == 400:
                        try:
                            error_detail = await response.json()
                            error_msg = f"Groq API error: {error_detail}"
                        except:
                            text = await response.text()
                            error_msg = f"Groq API error: {response.status} - {text}"
                        return LLMResponse(text="", model=model, success=False, error=error_msg)

                    response.raise_for_status()
                    result = await response.json()
                    text = result["choices"][0]["message"]["content"]

                    llm_response = LLMResponse(
                        text=text,
                        model=model,
                        success=True,
                        metadata={
                            "usage": result.get("usage"),
                            "finish_reason": result["choices"][0].get("finish_reason"),
                        },
                    )

                    # Cache successful response
                    self._save_to_cache(cache_key, llm_response)

                    return llm_response

            except asyncio.TimeoutError:
                return LLMResponse(text="", model=model, success=False, error="Request timed out")
            except aiohttp.ClientError as e:
                return LLMResponse(
                    text="", model=model, success=False, error=f"Groq API error: {str(e)}"
                )
            except Exception as e:
                return LLMResponse(
                    text="", model=model, success=False, error=f"Groq error: {str(e)}"
                )

        # Should never reach here, but just in case
        return LLMResponse(text="", model=model, success=False, error="Max retries exceeded")

    def _anthropic_generate(
        self,
        prompt: str,
        model: str,
        system: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        json_mode: bool,
    ) -> LLMResponse:
        """Generate using Anthropic API with automatic retry on rate limits.

        Args:
            prompt: User prompt
            model: Model name
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            json_mode: Force JSON output

        Returns:
            LLMResponse
        """
        import time

        # Check for API key upfront
        if not Config.ANTHROPIC_API_KEY:
            return LLMResponse(
                text="",
                model=model,
                success=False,
                error="ANTHROPIC_API_KEY not set. Get one at https://console.anthropic.com",
            )

        # Check cache first
        cache_key = self._generate_cache_key(
            provider="anthropic",
            model=model,
            prompt=prompt,
            system=system,
            temperature=temperature,
            json_mode=json_mode,
        )

        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response

        max_retries = 3
        base_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                url = "https://api.anthropic.com/v1/messages"

                headers = {
                    "x-api-key": Config.ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                }

                # Build messages
                messages = [{"role": "user", "content": prompt}]

                payload = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens or 4096,
                    "temperature": temperature,
                }

                # Add system prompt if provided
                if system:
                    payload["system"] = system

                response = requests.post(url, json=payload, headers=headers, timeout=60)
                response.raise_for_status()

                result = response.json()
                text = result["content"][0]["text"]

                llm_response = LLMResponse(
                    text=text,
                    model=model,
                    success=True,
                    metadata={
                        "usage": result.get("usage"),
                        "stop_reason": result.get("stop_reason"),
                    },
                )

                # Cache successful response
                self._save_to_cache(cache_key, llm_response)

                return llm_response

            except requests.exceptions.HTTPError as e:
                error_msg = f"Anthropic API error: {e}"
                if e.response.status_code == 401:
                    error_msg = "Invalid ANTHROPIC_API_KEY. Check your API key."
                    # Don't retry on auth errors
                    return LLMResponse(text="", model=model, success=False, error=error_msg)
                elif e.response.status_code == 429:
                    # Rate limit - retry with backoff
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)  # Exponential backoff
                        print(
                            f"  Rate limit hit, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        error_msg = "Rate limit exceeded. All retries exhausted."
                elif e.response.status_code == 400:
                    try:
                        error_detail = e.response.json()
                        error_msg = f"Anthropic API error: {error_detail}"
                    except:
                        error_msg = f"Anthropic API error: {e} - {e.response.text}"

                # Return error if not retrying
                return LLMResponse(text="", model=model, success=False, error=error_msg)
            except Exception as e:
                return LLMResponse(
                    text="", model=model, success=False, error=f"Anthropic error: {str(e)}"
                )

        # Should never reach here, but just in case
        return LLMResponse(text="", model=model, success=False, error="Max retries exceeded")

    async def _anthropic_generate_async(
        self,
        prompt: str,
        model: str,
        system: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        json_mode: bool,
    ) -> LLMResponse:
        """Generate using Anthropic API asynchronously with automatic retry on rate limits.

        Args:
            prompt: User prompt
            model: Model name
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            json_mode: Force JSON output

        Returns:
            LLMResponse
        """
        # Check for API key upfront
        if not Config.ANTHROPIC_API_KEY:
            return LLMResponse(
                text="",
                model=model,
                success=False,
                error="ANTHROPIC_API_KEY not set. Get one at https://console.anthropic.com",
            )

        # Check cache first
        cache_key = self._generate_cache_key(
            provider="anthropic",
            model=model,
            prompt=prompt,
            system=system,
            temperature=temperature,
            json_mode=json_mode,
        )

        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response

        if not self._session:
            return LLMResponse(
                text="",
                model=model,
                success=False,
                error="Async session not initialized. Use 'async with LLMManager()' context manager.",
            )

        max_retries = 3
        base_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                url = "https://api.anthropic.com/v1/messages"

                headers = {
                    "x-api-key": Config.ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                }

                # Build messages
                messages = [{"role": "user", "content": prompt}]

                payload = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens or 4096,
                    "temperature": temperature,
                }

                # Add system prompt if provided
                if system:
                    payload["system"] = system

                timeout = aiohttp.ClientTimeout(total=120)
                async with self._session.post(
                    url, json=payload, headers=headers, timeout=timeout
                ) as response:
                    if response.status == 401:
                        error_msg = "Invalid ANTHROPIC_API_KEY. Check your API key."
                        return LLMResponse(text="", model=model, success=False, error=error_msg)
                    elif response.status == 429:
                        # Rate limit - retry with backoff
                        if attempt < max_retries - 1:
                            delay = base_delay * (2**attempt)  # Exponential backoff
                            print(
                                f"  Rate limit hit, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})"
                            )
                            await asyncio.sleep(delay)
                            continue
                        else:
                            error_msg = "Rate limit exceeded. All retries exhausted."
                            return LLMResponse(text="", model=model, success=False, error=error_msg)
                    elif response.status == 400:
                        try:
                            error_detail = await response.json()
                            error_msg = f"Anthropic API error: {error_detail}"
                        except:
                            text = await response.text()
                            error_msg = f"Anthropic API error: {response.status} - {text}"
                        return LLMResponse(text="", model=model, success=False, error=error_msg)

                    response.raise_for_status()
                    result = await response.json()
                    text = result["content"][0]["text"]

                    llm_response = LLMResponse(
                        text=text,
                        model=model,
                        success=True,
                        metadata={
                            "usage": result.get("usage"),
                            "stop_reason": result.get("stop_reason"),
                        },
                    )

                    # Cache successful response
                    self._save_to_cache(cache_key, llm_response)

                    return llm_response

            except asyncio.TimeoutError:
                return LLMResponse(text="", model=model, success=False, error="Request timed out")
            except aiohttp.ClientError as e:
                return LLMResponse(
                    text="", model=model, success=False, error=f"Anthropic API error: {str(e)}"
                )
            except Exception as e:
                return LLMResponse(
                    text="", model=model, success=False, error=f"Anthropic error: {str(e)}"
                )

        # Should never reach here, but just in case
        return LLMResponse(text="", model=model, success=False, error="Max retries exceeded")

    def _deepseek_generate(
        self,
        prompt: str,
        model: str,
        system: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        json_mode: bool,
    ) -> LLMResponse:
        """Generate using DeepSeek API (OpenAI-compatible) with automatic retry on rate limits.

        Args:
            prompt: User prompt
            model: Model name
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            json_mode: Force JSON output

        Returns:
            LLMResponse
        """
        import time

        # Check for API key upfront
        if not Config.DEEPSEEK_API_KEY:
            return LLMResponse(
                text="",
                model=model,
                success=False,
                error="DEEPSEEK_API_KEY not set. Get one at https://platform.deepseek.com",
            )

        # Check cache first
        cache_key = self._generate_cache_key(
            provider="deepseek",
            model=model,
            prompt=prompt,
            system=system,
            temperature=temperature,
            json_mode=json_mode,
        )

        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response

        max_retries = 3
        base_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                url = "https://api.deepseek.com/v1/chat/completions"

                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})

                # Reasoner model doesn't support temperature or json_mode
                is_reasoner = model == "deepseek-reasoner"

                payload = {
                    "model": model,
                    "messages": messages,
                }

                if not is_reasoner:
                    payload["temperature"] = temperature
                    if json_mode:
                        payload["response_format"] = {"type": "json_object"}

                if max_tokens:
                    payload["max_tokens"] = max_tokens

                headers = {
                    "Authorization": f"Bearer {Config.DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json",
                }

                response = requests.post(
                    url, json=payload, headers=headers, timeout=60 if not is_reasoner else 300
                )
                response.raise_for_status()

                result = response.json()
                message = result["choices"][0]["message"]
                text = message["content"]

                # Capture reasoning for reasoner model
                metadata = {
                    "usage": result.get("usage"),
                    "finish_reason": result["choices"][0].get("finish_reason"),
                }
                if is_reasoner and "reasoning_content" in message:
                    metadata["reasoning"] = message["reasoning_content"]

                llm_response = LLMResponse(text=text, model=model, success=True, metadata=metadata)

                # Cache successful response
                self._save_to_cache(cache_key, llm_response)

                return llm_response

            except requests.exceptions.HTTPError as e:
                error_msg = f"DeepSeek API error: {e}"
                if e.response.status_code == 401:
                    error_msg = "Invalid DEEPSEEK_API_KEY. Check your API key."
                    # Don't retry on auth errors
                    return LLMResponse(text="", model=model, success=False, error=error_msg)
                elif e.response.status_code == 429:
                    # Rate limit - retry with backoff
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)  # Exponential backoff
                        print(
                            f"  Rate limit hit, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        error_msg = "Rate limit exceeded. All retries exhausted."
                elif e.response.status_code == 400:
                    try:
                        error_detail = e.response.json()
                        error_msg = f"DeepSeek API error: {error_detail}"
                    except:
                        error_msg = f"DeepSeek API error: {e} - {e.response.text}"

                # Return error if not retrying
                return LLMResponse(text="", model=model, success=False, error=error_msg)
            except Exception as e:
                return LLMResponse(
                    text="", model=model, success=False, error=f"DeepSeek error: {str(e)}"
                )

        # Should never reach here, but just in case
        return LLMResponse(text="", model=model, success=False, error="Max retries exceeded")

    async def _deepseek_generate_async(
        self,
        prompt: str,
        model: str,
        system: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        json_mode: bool,
    ) -> LLMResponse:
        """Generate using DeepSeek API asynchronously with automatic retry on rate limits.

        Args:
            prompt: User prompt
            model: Model name
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            json_mode: Force JSON output

        Returns:
            LLMResponse
        """
        # Check for API key upfront
        if not Config.DEEPSEEK_API_KEY:
            return LLMResponse(
                text="",
                model=model,
                success=False,
                error="DEEPSEEK_API_KEY not set. Get one at https://platform.deepseek.com",
            )

        # Check cache first
        cache_key = self._generate_cache_key(
            provider="deepseek",
            model=model,
            prompt=prompt,
            system=system,
            temperature=temperature,
            json_mode=json_mode,
        )

        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response

        if not self._session:
            return LLMResponse(
                text="",
                model=model,
                success=False,
                error="Async session not initialized. Use 'async with LLMManager()' context manager.",
            )

        max_retries = 3
        base_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                url = "https://api.deepseek.com/v1/chat/completions"

                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})

                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }

                if max_tokens:
                    payload["max_tokens"] = max_tokens

                if json_mode:
                    payload["response_format"] = {"type": "json_object"}

                headers = {
                    "Authorization": f"Bearer {Config.DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json",
                }

                timeout = aiohttp.ClientTimeout(total=120)
                async with self._session.post(
                    url, json=payload, headers=headers, timeout=timeout
                ) as response:
                    if response.status == 401:
                        error_msg = "Invalid DEEPSEEK_API_KEY. Check your API key."
                        return LLMResponse(text="", model=model, success=False, error=error_msg)
                    elif response.status == 429:
                        # Rate limit - retry with backoff
                        if attempt < max_retries - 1:
                            delay = base_delay * (2**attempt)  # Exponential backoff
                            print(
                                f"  Rate limit hit, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})"
                            )
                            await asyncio.sleep(delay)
                            continue
                        else:
                            error_msg = "Rate limit exceeded. All retries exhausted."
                            return LLMResponse(text="", model=model, success=False, error=error_msg)
                    elif response.status == 400:
                        try:
                            error_detail = await response.json()
                            error_msg = f"DeepSeek API error: {error_detail}"
                        except:
                            text = await response.text()
                            error_msg = f"DeepSeek API error: {response.status} - {text}"
                        return LLMResponse(text="", model=model, success=False, error=error_msg)

                    response.raise_for_status()
                    result = await response.json()
                    text = result["choices"][0]["message"]["content"]

                    llm_response = LLMResponse(
                        text=text,
                        model=model,
                        success=True,
                        metadata={
                            "usage": result.get("usage"),
                            "finish_reason": result["choices"][0].get("finish_reason"),
                        },
                    )

                    # Cache successful response
                    self._save_to_cache(cache_key, llm_response)

                    return llm_response

            except asyncio.TimeoutError:
                return LLMResponse(text="", model=model, success=False, error="Request timed out")
            except aiohttp.ClientError as e:
                return LLMResponse(
                    text="", model=model, success=False, error=f"DeepSeek API error: {str(e)}"
                )
            except Exception as e:
                return LLMResponse(
                    text="", model=model, success=False, error=f"DeepSeek error: {str(e)}"
                )

        # Should never reach here, but just in case
        return LLMResponse(text="", model=model, success=False, error="Max retries exceeded")

    def _openrouter_generate(
        self,
        prompt: str,
        model: str,
        system: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        json_mode: bool,
    ) -> LLMResponse:
        """Generate using OpenRouter API (OpenAI-compatible) with automatic retry on rate limits.

        Args:
            prompt: User prompt
            model: Model name (e.g., "deepseek/deepseek-chat", "google/gemini-2.0-flash-001")
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            json_mode: Force JSON output

        Returns:
            LLMResponse
        """
        import time

        # Check for API key upfront
        if not Config.OPENROUTER_API_KEY:
            return LLMResponse(
                text="",
                model=model,
                success=False,
                error="OPENROUTER_API_KEY not set. Get one at https://openrouter.ai/keys",
            )

        # Check cache first
        cache_key = self._generate_cache_key(
            provider="openrouter",
            model=model,
            prompt=prompt,
            system=system,
            temperature=temperature,
            json_mode=json_mode,
        )

        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response

        max_retries = 3
        base_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                url = "https://openrouter.ai/api/v1/chat/completions"

                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})

                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }

                if max_tokens:
                    payload["max_tokens"] = max_tokens

                if json_mode:
                    payload["response_format"] = {"type": "json_object"}

                headers = {
                    "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://examina.io",
                    "X-Title": "Examina",
                }

                response = requests.post(url, json=payload, headers=headers, timeout=120)
                response.raise_for_status()

                result = response.json()
                text = result["choices"][0]["message"]["content"]

                llm_response = LLMResponse(
                    text=text,
                    model=model,
                    success=True,
                    metadata={
                        "usage": result.get("usage"),
                        "finish_reason": result["choices"][0].get("finish_reason"),
                    },
                )

                # Cache successful response
                self._save_to_cache(cache_key, llm_response)

                return llm_response

            except requests.exceptions.HTTPError as e:
                error_msg = f"OpenRouter API error: {e}"
                if e.response.status_code == 401:
                    error_msg = "Invalid OPENROUTER_API_KEY. Check your API key."
                    # Don't retry on auth errors
                    return LLMResponse(text="", model=model, success=False, error=error_msg)
                elif e.response.status_code == 429:
                    # Rate limit - retry with backoff
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)  # Exponential backoff
                        print(
                            f"  Rate limit hit, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        error_msg = "Rate limit exceeded. All retries exhausted."
                elif e.response.status_code == 400:
                    try:
                        error_detail = e.response.json()
                        error_msg = f"OpenRouter API error: {error_detail}"
                    except:
                        error_msg = f"OpenRouter API error: {e} - {e.response.text}"

                # Return error if not retrying
                return LLMResponse(text="", model=model, success=False, error=error_msg)
            except Exception as e:
                return LLMResponse(
                    text="", model=model, success=False, error=f"OpenRouter error: {str(e)}"
                )

        # Should never reach here, but just in case
        return LLMResponse(text="", model=model, success=False, error="Max retries exceeded")

    async def _openrouter_generate_async(
        self,
        prompt: str,
        model: str,
        system: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        json_mode: bool,
    ) -> LLMResponse:
        """Generate using OpenRouter API asynchronously with automatic retry on rate limits.

        Args:
            prompt: User prompt
            model: Model name
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            json_mode: Force JSON output

        Returns:
            LLMResponse
        """
        # Check for API key upfront
        if not Config.OPENROUTER_API_KEY:
            return LLMResponse(
                text="",
                model=model,
                success=False,
                error="OPENROUTER_API_KEY not set. Get one at https://openrouter.ai/keys",
            )

        # Check cache first
        cache_key = self._generate_cache_key(
            provider="openrouter",
            model=model,
            prompt=prompt,
            system=system,
            temperature=temperature,
            json_mode=json_mode,
        )

        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response

        if not self._session:
            return LLMResponse(
                text="",
                model=model,
                success=False,
                error="Async session not initialized. Use 'async with LLMManager()' context manager.",
            )

        max_retries = 3
        base_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                url = "https://openrouter.ai/api/v1/chat/completions"

                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})

                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }

                if max_tokens:
                    payload["max_tokens"] = max_tokens

                if json_mode:
                    payload["response_format"] = {"type": "json_object"}

                headers = {
                    "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://examina.io",
                    "X-Title": "Examina",
                }

                timeout = aiohttp.ClientTimeout(total=120)
                async with self._session.post(
                    url, json=payload, headers=headers, timeout=timeout
                ) as response:
                    if response.status == 401:
                        error_msg = "Invalid OPENROUTER_API_KEY. Check your API key."
                        return LLMResponse(text="", model=model, success=False, error=error_msg)
                    elif response.status == 429:
                        # Rate limit - retry with backoff
                        if attempt < max_retries - 1:
                            delay = base_delay * (2**attempt)  # Exponential backoff
                            print(
                                f"  Rate limit hit, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})"
                            )
                            await asyncio.sleep(delay)
                            continue
                        else:
                            error_msg = "Rate limit exceeded. All retries exhausted."
                            return LLMResponse(text="", model=model, success=False, error=error_msg)
                    elif response.status == 400:
                        try:
                            error_detail = await response.json()
                            error_msg = f"OpenRouter API error: {error_detail}"
                        except:
                            text = await response.text()
                            error_msg = f"OpenRouter API error: {response.status} - {text}"
                        return LLMResponse(text="", model=model, success=False, error=error_msg)

                    response.raise_for_status()
                    result = await response.json()
                    text = result["choices"][0]["message"]["content"]

                    llm_response = LLMResponse(
                        text=text,
                        model=model,
                        success=True,
                        metadata={
                            "usage": result.get("usage"),
                            "finish_reason": result["choices"][0].get("finish_reason"),
                        },
                    )

                    # Cache successful response
                    self._save_to_cache(cache_key, llm_response)

                    return llm_response

            except asyncio.TimeoutError:
                return LLMResponse(text="", model=model, success=False, error="Request timed out")
            except aiohttp.ClientError as e:
                return LLMResponse(
                    text="", model=model, success=False, error=f"OpenRouter API error: {str(e)}"
                )
            except Exception as e:
                return LLMResponse(
                    text="", model=model, success=False, error=f"OpenRouter error: {str(e)}"
                )

        # Should never reach here, but just in case
        return LLMResponse(text="", model=model, success=False, error="Max retries exceeded")

    def generate_with_image(
        self,
        prompt: str,
        image_bytes: bytes,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate text from LLM with image input (Vision).

        Args:
            prompt: User prompt describing what to do with the image
            image_bytes: Image data as bytes (PNG, JPEG, etc.)
            model: Vision model to use (defaults to provider-specific vision model)
            system: System prompt
            temperature: Sampling temperature (lower for OCR tasks)
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse with generated text
        """
        # Only OpenRouter and DeepSeek support vision
        if self.provider not in ("openrouter", "deepseek"):
            return LLMResponse(
                text="",
                model=model or "unknown",
                success=False,
                error=f"Vision not supported for provider '{self.provider}'. Use 'openrouter' or 'deepseek' provider.",
            )

        # Select appropriate vision model
        if self.provider == "openrouter":
            model = model or Config.OPENROUTER_VISION_MODEL
        else:
            model = model or Config.DEEPSEEK_VISION_MODEL

        # Apply rate limiting before making request
        wait_time = self.rate_limiter.wait_if_needed(self.provider)
        if wait_time > 0:
            print(
                f"  [RATE LIMIT] Waiting {wait_time:.1f}s for '{self.provider}' (rate limit protection)"
            )

        if self.provider == "openrouter":
            response = self._openrouter_generate_with_image(
                prompt, image_bytes, model, system, temperature, max_tokens
            )
        else:
            response = self._deepseek_generate_with_image(
                prompt, image_bytes, model, system, temperature, max_tokens
            )

        # Record usage if request was successful
        if response.success:
            tokens_used = 0
            if response.metadata:
                usage = response.metadata.get("usage", {})
                if isinstance(usage, dict):
                    tokens_used = usage.get("total_tokens", 0)
                    if tokens_used == 0:
                        tokens_used = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

            self.rate_limiter.record_request(self.provider, tokens_used=tokens_used)

        return response

    def _deepseek_generate_with_image(
        self,
        prompt: str,
        image_bytes: bytes,
        model: str,
        system: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
    ) -> LLMResponse:
        """Generate using DeepSeek Vision API with image input.

        Args:
            prompt: User prompt
            image_bytes: Image data as bytes
            model: Model name
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            LLMResponse
        """
        import base64
        import time

        # Check for API key upfront
        if not Config.DEEPSEEK_API_KEY:
            return LLMResponse(
                text="",
                model=model,
                success=False,
                error="DEEPSEEK_API_KEY not set. Get one at https://platform.deepseek.com",
            )

        # Encode image to base64
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Detect image type from bytes
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            media_type = "image/png"
        elif image_bytes[:2] == b'\xff\xd8':
            media_type = "image/jpeg"
        else:
            media_type = "image/png"  # Default to PNG

        max_retries = 3
        base_delay = 2

        for attempt in range(max_retries):
            try:
                url = "https://api.deepseek.com/v1/chat/completions"

                # Build messages with image content
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})

                # User message with image and text
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                })

                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }

                if max_tokens:
                    payload["max_tokens"] = max_tokens

                headers = {
                    "Authorization": f"Bearer {Config.DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json",
                }

                response = requests.post(url, json=payload, headers=headers, timeout=120)
                response.raise_for_status()

                result = response.json()
                text = result["choices"][0]["message"]["content"]

                return LLMResponse(
                    text=text,
                    model=model,
                    success=True,
                    metadata={
                        "usage": result.get("usage"),
                        "finish_reason": result["choices"][0].get("finish_reason"),
                    },
                )

            except requests.exceptions.HTTPError as e:
                error_msg = f"DeepSeek Vision API error: {e}"
                if e.response.status_code == 401:
                    error_msg = "Invalid DEEPSEEK_API_KEY. Check your API key."
                    return LLMResponse(text="", model=model, success=False, error=error_msg)
                elif e.response.status_code == 429:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)
                        print(f"  Rate limit hit, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        error_msg = "Rate limit exceeded. All retries exhausted."
                elif e.response.status_code == 400:
                    try:
                        error_detail = e.response.json()
                        error_msg = f"DeepSeek Vision API error: {error_detail}"
                    except:
                        error_msg = f"DeepSeek Vision API error: {e} - {e.response.text}"

                return LLMResponse(text="", model=model, success=False, error=error_msg)
            except Exception as e:
                return LLMResponse(
                    text="", model=model, success=False, error=f"DeepSeek Vision error: {str(e)}"
                )

        return LLMResponse(text="", model=model, success=False, error="Max retries exceeded")

    def _openrouter_generate_with_image(
        self,
        prompt: str,
        image_bytes: bytes,
        model: str,
        system: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
    ) -> LLMResponse:
        """Generate using OpenRouter API with image input (via vision models like Gemini).

        Args:
            prompt: User prompt
            image_bytes: Image data as bytes
            model: Vision model name (e.g., "google/gemini-2.0-flash-001")
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            LLMResponse
        """
        import base64
        import time

        # Check for API key upfront
        if not Config.OPENROUTER_API_KEY:
            return LLMResponse(
                text="",
                model=model,
                success=False,
                error="OPENROUTER_API_KEY not set. Get one at https://openrouter.ai/keys",
            )

        # Encode image to base64
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Detect image type from bytes
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            media_type = "image/png"
        elif image_bytes[:2] == b'\xff\xd8':
            media_type = "image/jpeg"
        else:
            media_type = "image/png"  # Default to PNG

        max_retries = 3
        base_delay = 2

        for attempt in range(max_retries):
            try:
                url = "https://openrouter.ai/api/v1/chat/completions"

                # Build messages with image content (OpenAI-compatible format)
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})

                # User message with image and text
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                })

                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }

                if max_tokens:
                    payload["max_tokens"] = max_tokens

                headers = {
                    "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://examina.io",
                    "X-Title": "Examina",
                }

                response = requests.post(url, json=payload, headers=headers, timeout=120)
                response.raise_for_status()

                result = response.json()
                text = result["choices"][0]["message"]["content"]

                return LLMResponse(
                    text=text,
                    model=model,
                    success=True,
                    metadata={
                        "usage": result.get("usage"),
                        "finish_reason": result["choices"][0].get("finish_reason"),
                    },
                )

            except requests.exceptions.HTTPError as e:
                error_msg = f"OpenRouter Vision API error: {e}"
                if e.response.status_code == 401:
                    error_msg = "Invalid OPENROUTER_API_KEY. Check your API key."
                    return LLMResponse(text="", model=model, success=False, error=error_msg)
                elif e.response.status_code == 429:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)
                        print(f"  Rate limit hit, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        error_msg = "Rate limit exceeded. All retries exhausted."
                elif e.response.status_code == 400:
                    try:
                        error_detail = e.response.json()
                        error_msg = f"OpenRouter Vision API error: {error_detail}"
                    except:
                        error_msg = f"OpenRouter Vision API error: {e} - {e.response.text}"

                return LLMResponse(text="", model=model, success=False, error=error_msg)
            except Exception as e:
                return LLMResponse(
                    text="", model=model, success=False, error=f"OpenRouter Vision error: {str(e)}"
                )

        return LLMResponse(text="", model=model, success=False, error="Max retries exceeded")

    def generate_image(
        self,
        prompt: str,
        model: Optional[str] = None,
    ) -> LLMResponse:
        """Generate an image from a text prompt using OpenRouter.

        Args:
            prompt: Text description of the image to generate
            model: Image model to use (defaults to OPENROUTER_IMAGE_MODEL)

        Returns:
            LLMResponse with image URL in text field
        """
        import time

        # Check for API key upfront
        if not Config.OPENROUTER_API_KEY:
            return LLMResponse(
                text="",
                model=model or "unknown",
                success=False,
                error="OPENROUTER_API_KEY not set. Get one at https://openrouter.ai/keys",
            )

        model = model or Config.OPENROUTER_IMAGE_MODEL

        max_retries = 3
        base_delay = 2

        for attempt in range(max_retries):
            try:
                url = "https://openrouter.ai/api/v1/chat/completions"

                # Request image generation via modalities parameter
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "modalities": ["image", "text"],
                }

                headers = {
                    "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://examina.io",
                    "X-Title": "Examina",
                }

                response = requests.post(url, json=payload, headers=headers, timeout=120)
                response.raise_for_status()

                result = response.json()
                message = result["choices"][0]["message"]

                # Extract image URL from response
                # OpenRouter returns images in content array with type "image_url"
                image_url = None
                text_content = ""

                if isinstance(message.get("content"), list):
                    for item in message["content"]:
                        if item.get("type") == "image_url":
                            image_url = item.get("image_url", {}).get("url")
                        elif item.get("type") == "text":
                            text_content = item.get("text", "")
                else:
                    text_content = message.get("content", "")

                return LLMResponse(
                    text=image_url or text_content,
                    model=model,
                    success=image_url is not None,
                    error=None if image_url else "No image generated",
                    metadata={
                        "usage": result.get("usage"),
                        "image_url": image_url,
                        "text": text_content,
                    },
                )

            except requests.exceptions.HTTPError as e:
                error_msg = f"OpenRouter Image API error: {e}"
                if e.response.status_code == 401:
                    error_msg = "Invalid OPENROUTER_API_KEY. Check your API key."
                    return LLMResponse(text="", model=model, success=False, error=error_msg)
                elif e.response.status_code == 429:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)
                        print(f"  Rate limit hit, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        error_msg = "Rate limit exceeded. All retries exhausted."
                elif e.response.status_code == 400:
                    try:
                        error_detail = e.response.json()
                        error_msg = f"OpenRouter Image API error: {error_detail}"
                    except:
                        error_msg = f"OpenRouter Image API error: {e} - {e.response.text}"

                return LLMResponse(text="", model=model, success=False, error=error_msg)
            except Exception as e:
                return LLMResponse(
                    text="", model=model, success=False, error=f"OpenRouter Image error: {str(e)}"
                )

        return LLMResponse(text="", model=model, success=False, error="Max retries exceeded")

    def embed(self, text: str, model: Optional[str] = None) -> Optional[List[float]]:
        """Generate embeddings for text.

        Args:
            text: Text to embed
            model: Embedding model (defaults to embed_model)

        Returns:
            List of floats (embedding vector) or None on error
        """
        model = model or self.embed_model

        if self.provider == "ollama":
            return self._ollama_embed(text, model)
        else:
            return None

    def _ollama_embed(self, text: str, model: str) -> Optional[List[float]]:
        """Generate embeddings using Ollama.

        Args:
            text: Text to embed
            model: Model name

        Returns:
            Embedding vector or None
        """
        try:
            url = f"{self.base_url}/api/embeddings"

            payload = {"model": model, "prompt": text}

            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            return result.get("embedding")

        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def check_model_available(self, model: str) -> bool:
        """Check if a model is available locally.

        Args:
            model: Model name

        Returns:
            True if model is available
        """
        if self.provider == "ollama":
            try:
                url = f"{self.base_url}/api/tags"
                response = requests.get(url, timeout=10)
                response.raise_for_status()

                models = response.json().get("models", [])
                available_models = [m["name"] for m in models]

                return model in available_models

            except Exception:
                return False
        return False

    def list_available_models(self) -> List[str]:
        """List all available models.

        Returns:
            List of model names
        """
        if self.provider == "ollama":
            try:
                url = f"{self.base_url}/api/tags"
                response = requests.get(url, timeout=10)
                response.raise_for_status()

                models = response.json().get("models", [])
                return [m["name"] for m in models]

            except Exception:
                return []
        return []

    def parse_json_response(self, response: LLMResponse) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response.

        Args:
            response: LLM response

        Returns:
            Parsed JSON dict or None on error
        """
        if not response.success:
            return None

        try:
            # Try to parse the entire response
            return json.loads(response.text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            text = response.text
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                json_str = text[start:end].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass

            # Try to find JSON object in text
            import re

            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            return None

    @staticmethod
    def is_provider_available(provider: str) -> bool:
        """Check if a provider is available (has API key configured).

        This is a static utility method for provider availability checking.
        Used by ProviderRouter to determine if a provider can be used.

        Args:
            provider: Provider name (e.g., "anthropic", "groq", "ollama", "deepseek")

        Returns:
            True if provider is available (API key exists or not required)
        """
        # Ollama doesn't need API key (local)
        if provider == "ollama":
            return True

        # Check for API keys
        if provider == "anthropic":
            return Config.ANTHROPIC_API_KEY is not None and len(Config.ANTHROPIC_API_KEY) > 0
        elif provider == "groq":
            return Config.GROQ_API_KEY is not None and len(Config.GROQ_API_KEY) > 0
        elif provider == "openai":
            return Config.OPENAI_API_KEY is not None and len(Config.OPENAI_API_KEY) > 0
        elif provider == "deepseek":
            return Config.DEEPSEEK_API_KEY is not None and len(Config.DEEPSEEK_API_KEY) > 0
        elif provider == "openrouter":
            return Config.OPENROUTER_API_KEY is not None and len(Config.OPENROUTER_API_KEY) > 0

        # Unknown provider
        return False
