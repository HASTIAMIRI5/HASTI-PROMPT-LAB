from __future__ import annotations

from pydantic import BaseModel, Field


class PromptVariant(BaseModel):
    """A single prompt configuration for testing."""

    name: str
    system_prompt: str = Field(default="You are a helpful assistant.")
    user_prompt: str
    temperature: float = 0.1
    max_tokens: int = 256


class ExperimentCase(BaseModel):
    """A single evaluation example."""

    id: str
    input: str
    expected_output: str
    category: str = "general"
    tags: list[str] = Field(default_factory=list)


class EvaluationConfig(BaseModel):
    """Global evaluation settings."""

    pass_threshold: float = 0.8
    weight_accuracy: float = 0.7
    weight_clarity: float = 0.3


class EvaluationResult(BaseModel):
    """Result for one prompt evaluated on a dataset."""

    variant_name: str
    average_score: float
    accuracy: float
    clarity: float
    passed: bool
    details: list[dict[str, float | str]] = Field(default_factory=list)
