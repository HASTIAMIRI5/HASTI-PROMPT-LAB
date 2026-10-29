from __future__ import annotations

import os

from .models import PromptVariant


class PromptProviderError(RuntimeError):
    """Raised when a provider is unavailable or misconfigured."""


def _get_api_key(env_name: str, override: str | None = None) -> str:
    value = override or os.getenv(env_name)
    if not value:
        raise PromptProviderError(f"Missing API key for {env_name}. Set it in your environment or .env file.")
    return value


def call_provider(
    prompt: PromptVariant,
    user_input: str,
    *,
    provider_name: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> str:
    """Call a supported provider and return the model response text."""
    provider_name = (provider_name or prompt.provider or "openai").lower()
    selected_model = model or prompt.model

    if provider_name == "openai":
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise PromptProviderError("The openai package is not installed. Install it with pip install openai.") from exc

        client = OpenAI(api_key=_get_api_key("OPENAI_API_KEY", api_key))
        response = client.chat.completions.create(
            model=selected_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": user_input},
            ],
            temperature=prompt.temperature,
            max_tokens=prompt.max_tokens,
        )
        return response.choices[0].message.content or ""

    if provider_name == "anthropic":
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover
            raise PromptProviderError("The anthropic package is not installed. Install it with pip install anthropic.") from exc

        client = Anthropic(api_key=_get_api_key("ANTHROPIC_API_KEY", api_key))
        response = client.messages.create(
            model=selected_model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"),
            max_tokens=prompt.max_tokens,
            temperature=prompt.temperature,
            system=prompt.system_prompt,
            messages=[{"role": "user", "content": user_input}],
        )
        text_parts: list[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
        return "\n".join(text_parts).strip()

    raise PromptProviderError(f"Unsupported provider: {provider_name}. Supported providers: openai, anthropic.")


def provider_defaults() -> dict[str, str]:
    """Return the default model selections used by the lab."""
    return {
        "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"),
    }
