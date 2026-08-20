"""
utils/llm_client.py

Thin wrapper around NVIDIA's OpenAI-compatible API so every agent calls the
model the same way. Swap MODEL / provider here without touching agent code.

Env vars expected:
    NVIDIA_API_KEY
    EDULEAP_MODEL (optional)
    EDULEAP_FALLBACK_MODEL (optional)
"""

import os
import json
import logging
from typing import Optional

from openai import OpenAI

logger = logging.getLogger("eduleap.llm")

NVIDIA_BASE_URL = os.getenv(
    "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
)
MODEL = os.getenv("EDULEAP_MODEL", "meta/llama-3.3-70b-instruct")
FALLBACK_MODEL = os.getenv(
    "EDULEAP_FALLBACK_MODEL", "meta/llama-3.1-70b-instruct"
)

_api_key = os.getenv("NVIDIA_API_KEY")
if not _api_key:
    raise RuntimeError(
        "NVIDIA_API_KEY is not set. Add your NVIDIA API key to the environment."
    )

_client = OpenAI(
    api_key=_api_key,
    base_url=NVIDIA_BASE_URL,
)


def call_llm(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 800,
    temperature: float = 0.4,
    json_mode: bool = False,
) -> str:
    """
    Single entry point for all LLM calls in EduLeap.
    Retries once against a fallback model on failure.
    Returns raw text. If json_mode=True, strips Markdown JSON fences before returning.
    """
    for model in (MODEL, FALLBACK_MODEL):
        try:
            response = _client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            text = (response.choices[0].message.content or "").strip()

            if json_mode:
                text = text.replace("```json", "").replace("```", "").strip()

            return text

        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM call failed on model=%s: %s", model, exc)
            continue

    raise RuntimeError("EduLeap: all configured NVIDIA LLM models failed for this request.")


def call_llm_json(system_prompt: str, user_prompt: str, max_tokens: int = 800) -> dict:
    """Convenience wrapper that parses the response as JSON, with a safe fallback."""
    raw = call_llm(system_prompt, user_prompt, max_tokens=max_tokens, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM JSON output: %s", raw[:300])
        return {"error": "parse_failure", "raw": raw}
