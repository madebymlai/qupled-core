"""
LLM Manager for Examina.
Handles interactions with Ollama and other LLM providers.
"""

import json
import requests
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from config import Config


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

    def __init__(self, provider: str = "ollama", base_url: Optional[str] = None):
        """Initialize LLM manager.

        Args:
            provider: LLM provider ("ollama", "anthropic", "openai", "groq")
            base_url: Base URL for API (for Ollama)
        """
        self.provider = provider
        self.base_url = base_url or Config.OLLAMA_BASE_URL

        # Model selection
        if provider == "groq":
            self.primary_model = Config.GROQ_MODEL
            self.fast_model = Config.GROQ_MODEL
            self.embed_model = Config.LLM_EMBED_MODEL  # Still use Ollama for embeddings
        elif provider == "anthropic":
            self.primary_model = Config.ANTHROPIC_MODEL
            self.fast_model = Config.ANTHROPIC_MODEL
            self.embed_model = Config.LLM_EMBED_MODEL  # Still use Ollama for embeddings
        else:
            self.primary_model = Config.LLM_PRIMARY_MODEL  # Heavy reasoning
            self.fast_model = Config.LLM_FAST_MODEL  # Quick tasks
            self.embed_model = Config.LLM_EMBED_MODEL  # Embeddings

    def generate(self, prompt: str, model: Optional[str] = None,
                 system: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 json_mode: bool = False) -> LLMResponse:
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

        if self.provider == "ollama":
            return self._ollama_generate(
                prompt, model, system, temperature, max_tokens, json_mode
            )
        elif self.provider == "groq":
            return self._groq_generate(
                prompt, model, system, temperature, max_tokens, json_mode
            )
        elif self.provider == "anthropic":
            return self._anthropic_generate(
                prompt, model, system, temperature, max_tokens, json_mode
            )
        else:
            return LLMResponse(
                text="",
                model=model,
                success=False,
                error=f"Provider {self.provider} not implemented yet"
            )

    def _ollama_generate(self, prompt: str, model: str,
                        system: Optional[str], temperature: float,
                        max_tokens: Optional[int], json_mode: bool) -> LLMResponse:
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
                }
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
                }
            )

        except requests.exceptions.ConnectionError:
            return LLMResponse(
                text="",
                model=model,
                success=False,
                error="Cannot connect to Ollama. Is it running? (ollama serve)"
            )
        except requests.exceptions.Timeout:
            return LLMResponse(
                text="",
                model=model,
                success=False,
                error="Request timed out. Model might be too slow."
            )
        except Exception as e:
            return LLMResponse(
                text="",
                model=model,
                success=False,
                error=f"Ollama error: {str(e)}"
            )

    def _groq_generate(self, prompt: str, model: str,
                      system: Optional[str], temperature: float,
                      max_tokens: Optional[int], json_mode: bool) -> LLMResponse:
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
                error="GROQ_API_KEY not set. Get one at https://console.groq.com"
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
                    "Content-Type": "application/json"
                }

                response = requests.post(url, json=payload, headers=headers, timeout=60)
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
                    }
                )

            except requests.exceptions.HTTPError as e:
                error_msg = f"Groq API error: {e}"
                if e.response.status_code == 401:
                    error_msg = "Invalid GROQ_API_KEY. Check your API key."
                    # Don't retry on auth errors
                    return LLMResponse(text="", model=model, success=False, error=error_msg)
                elif e.response.status_code == 429:
                    # Rate limit - retry with backoff
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        print(f"  Rate limit hit, retrying in {delay}s... (attempt {attempt+1}/{max_retries})")
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
                return LLMResponse(
                    text="",
                    model=model,
                    success=False,
                    error=error_msg
                )
            except Exception as e:
                return LLMResponse(
                    text="",
                    model=model,
                    success=False,
                    error=f"Groq error: {str(e)}"
                )

        # Should never reach here, but just in case
        return LLMResponse(
            text="",
            model=model,
            success=False,
            error="Max retries exceeded"
        )

    def _anthropic_generate(self, prompt: str, model: str,
                           system: Optional[str], temperature: float,
                           max_tokens: Optional[int], json_mode: bool) -> LLMResponse:
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
                error="ANTHROPIC_API_KEY not set. Get one at https://console.anthropic.com"
            )

        max_retries = 3
        base_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                url = "https://api.anthropic.com/v1/messages"

                headers = {
                    "x-api-key": Config.ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
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

                return LLMResponse(
                    text=text,
                    model=model,
                    success=True,
                    metadata={
                        "usage": result.get("usage"),
                        "stop_reason": result.get("stop_reason"),
                    }
                )

            except requests.exceptions.HTTPError as e:
                error_msg = f"Anthropic API error: {e}"
                if e.response.status_code == 401:
                    error_msg = "Invalid ANTHROPIC_API_KEY. Check your API key."
                    # Don't retry on auth errors
                    return LLMResponse(text="", model=model, success=False, error=error_msg)
                elif e.response.status_code == 429:
                    # Rate limit - retry with backoff
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        print(f"  Rate limit hit, retrying in {delay}s... (attempt {attempt+1}/{max_retries})")
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
                return LLMResponse(
                    text="",
                    model=model,
                    success=False,
                    error=error_msg
                )
            except Exception as e:
                return LLMResponse(
                    text="",
                    model=model,
                    success=False,
                    error=f"Anthropic error: {str(e)}"
                )

        # Should never reach here, but just in case
        return LLMResponse(
            text="",
            model=model,
            success=False,
            error="Max retries exceeded"
        )

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

            payload = {
                "model": model,
                "prompt": text
            }

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
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            return None
