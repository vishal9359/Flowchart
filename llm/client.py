"""
Local LLM HTTP client.

Compatible with Ollama's /api/generate endpoint and OpenAI-compatible servers.

Key parameter: num_ctx
----------------------
Ollama defaults num_ctx (the model's context window) to 2048 tokens for many
models.  Our prompts are 2000–2500 tokens for medium-sized functions, which
exceeds that default.  When the prompt exceeds num_ctx Ollama returns an empty
response string immediately — which maps to None in generate() and causes the
"no LLM response" warning.

The fix: always pass num_ctx explicitly in the Ollama options so the model
loads with a sufficient context window.  Default is 8192 tokens; configurable
via --llm-num-ctx on the CLI.
"""

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class LlmClient:
    """
    Thin wrapper around a local LLM HTTP API.

    Supports:
      - Ollama:  POST /api/generate   (default)
      - OpenAI-compatible: POST /v1/chat/completions  (use_openai_format=True)
    """

    def __init__(self, url: str, model: str,
                 timeout: int = 120,
                 temperature: float = 0.1,
                 num_ctx: int = 8192,
                 use_openai_format: bool = False) -> None:
        self._url = url
        self._model = model
        self._timeout = timeout
        self._temperature = temperature
        self._num_ctx = num_ctx
        self._openai = use_openai_format

    def generate(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        Call the LLM and return the raw text response.
        Returns None on any failure (timeout, HTTP error, empty response, etc.).
        """
        try:
            if self._openai:
                return self._call_openai(system_prompt, user_prompt)
            return self._call_ollama(system_prompt, user_prompt)
        except requests.Timeout:
            logger.error("LLM request timed out after %ds", self._timeout)
        except requests.ConnectionError as exc:
            logger.error("LLM connection error: %s", exc)
        except requests.HTTPError as exc:
            logger.error("LLM HTTP error %s: %s",
                         exc.response.status_code, exc.response.text[:200])
        except Exception as exc:
            logger.error("Unexpected LLM error: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Ollama format
    # ------------------------------------------------------------------

    def _call_ollama(self, system: str, user: str) -> Optional[str]:
        payload = {
            "model": self._model,
            "system": system,
            "prompt": user,
            "stream": False,
            "options": {
                # num_ctx: explicitly set the context window size.
                # Without this, Ollama uses the model's baked-in default
                # which is often 2048 tokens.  Prompts that exceed num_ctx
                # cause Ollama to return an empty response string immediately.
                "num_ctx": self._num_ctx,
                "temperature": self._temperature,
                "top_p": 0.9,
                "num_predict": 2048,
            },
        }
        resp = requests.post(self._url, json=payload, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()

        response_text = data.get("response", "").strip()
        if not response_text:
            # Log the full server response at DEBUG level to aid diagnosis
            logger.debug("Ollama returned empty response. Server data: %s",
                         str(data)[:300])
        return response_text or None

    # ------------------------------------------------------------------
    # OpenAI-compatible format
    # ------------------------------------------------------------------

    def _call_openai(self, system: str, user: str) -> Optional[str]:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self._temperature,
            "max_tokens": 2048,
        }
        resp = requests.post(self._url, json=payload, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "").strip() or None
        return None
