"""
Local LLM HTTP client.

Compatible with Ollama's /api/generate endpoint.
For OpenAI-compatible servers, set llm_url to their chat/completions endpoint
and set use_openai_format=True in the constructor.
"""

import json
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
                 use_openai_format: bool = False) -> None:
        self._url = url
        self._model = model
        self._timeout = timeout
        self._temperature = temperature
        self._openai = use_openai_format

    def generate(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        Call the LLM and return the raw text response.
        Returns None on any failure (timeout, HTTP error, etc.).
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
                "temperature": self._temperature,
                "top_p": 0.9,
                "num_predict": 2048,
            },
        }
        resp = requests.post(self._url, json=payload, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip() or None

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
