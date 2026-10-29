from __future__ import annotations

from typing import Iterable

from .models import PromptVariant


def format_prompt(system_prompt: str, user_prompt: str, context: str | None = None) -> str:
    """Turn a system + user prompt into a clean final instruction payload."""
    sections = [
        "SYSTEM:",
        system_prompt.strip(),
        "",
        "USER:",
        user_prompt.strip(),
    ]

    if context and context.strip():
        sections.extend(["", "CONTEXT:", context.strip()])

    return "\n".join(sections).strip()


def build_prompt_variants(
    base_system_prompt: str,
    base_user_prompt: str,
    variants: Iterable[str],
    context: str | None = None,
) -> list[PromptVariant]:
    """Create variants by appending addition prompts or instructions."""
    generated: list[PromptVariant] = []
    for idx, modifier in enumerate(variants, start=1):
        user = f"{base_user_prompt}\n\n{modifier}".strip()
        generated.append(
            PromptVariant(
                name=f"variant_{idx}",
                system_prompt=base_system_prompt,
                user_prompt=format_prompt(base_system_prompt, user, context=context),
            )
        )
    return generated
