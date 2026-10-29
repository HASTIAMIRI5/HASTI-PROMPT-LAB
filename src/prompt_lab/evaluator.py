from __future__ import annotations

import json
import uuid
from typing import Callable, Iterable

from .db import ExperimentStore
from .models import EvaluationResult, ExperimentCase, PromptVariant
from .providers import call_provider


def normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def exactness_score(prediction: str, expected: str) -> float:
    normalized_prediction = normalize_text(prediction)
    normalized_expected = normalize_text(expected)
    if not normalized_expected:
        return 1.0
    return 1.0 if normalized_prediction == normalized_expected else 0.0


def keyword_coverage_score(prediction: str, expected: str) -> float:
    expected_words = set(normalize_text(expected).split())
    if not expected_words:
        return 1.0
    prediction_words = set(normalize_text(prediction).split())
    if not prediction_words:
        return 0.0
    overlap = expected_words & prediction_words
    return len(overlap) / len(expected_words)


def evaluate_case(case: ExperimentCase, prediction: str) -> dict[str, float | str]:
    accuracy = exactness_score(prediction, case.expected_output)
    clarity = keyword_coverage_score(prediction, case.expected_output)
    return {
        "id": case.id,
        "accuracy": round(accuracy, 4),
        "clarity": round(clarity, 4),
        "category": case.category,
    }


class ExperimentRunner:
    """Runs prompt experiments across a dataset and evaluates each output."""

    def __init__(
        self,
        model_name: str = "demo-model",
        response_builder: Callable[[PromptVariant, str], str] | None = None,
        *,
        store: ExperimentStore | None = None,
    ) -> None:
        self.model_name = model_name
        self.response_builder = response_builder or self._default_response_builder
        self.store = store

    @staticmethod
    def _default_response_builder(prompt: PromptVariant, user_input: str) -> str:
        cleaned_input = user_input.strip()
        if not cleaned_input:
            return ""
        return f"{prompt.system_prompt}\n\n{cleaned_input}"

    @staticmethod
    def load_cases(path: str) -> list[ExperimentCase]:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, dict) and "cases" in payload:
            payload = payload["cases"]

        return [ExperimentCase(**item) for item in payload]

    def run(self, prompt: PromptVariant, cases: Iterable[ExperimentCase]) -> EvaluationResult:
        case_list = list(cases)
        total_accuracy = 0.0
        total_clarity = 0.0
        details: list[dict[str, float | str]] = []

        for case in case_list:
            prediction = self.response_builder(prompt, case.input)
            metrics = evaluate_case(case, prediction)
            details.append(metrics)
            total_accuracy += float(metrics["accuracy"])
            total_clarity += float(metrics["clarity"])

        accuracy = total_accuracy / len(case_list) if case_list else 0.0
        clarity = total_clarity / len(case_list) if case_list else 0.0
        average_score = (accuracy * 0.7) + (clarity * 0.3)

        result = EvaluationResult(
            variant_name=prompt.name,
            average_score=round(average_score, 4),
            accuracy=round(accuracy, 4),
            clarity=round(clarity, 4),
            passed=average_score >= 0.8,
            details=details,
        )

        if self.store is not None:
            run_id = f"{prompt.name}-{uuid.uuid4().hex[:8]}"
            self.store.save_run(
                run_id=run_id,
                variant_name=prompt.name,
                provider=prompt.provider,
                model=prompt.model or self.model_name,
                dataset_name=None,
                metrics=result.model_dump(),
            )

        return result


def evaluate_prompt_variant(
    prompt: PromptVariant,
    cases: Iterable[ExperimentCase],
    response_builder: Callable[[PromptVariant, str], str] | None = None,
    *,
    store: ExperimentStore | None = None,
) -> EvaluationResult:
    runner = ExperimentRunner(response_builder=response_builder, store=store, model_name=prompt.model or "demo-model")
    return runner.run(prompt, cases)


def run_provider_experiment(
    prompt: PromptVariant,
    cases: Iterable[ExperimentCase],
    *,
    provider_name: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    store: ExperimentStore | None = None,
) -> EvaluationResult:
    """Run a prompt against a real provider and score the outputs."""
    selected_provider = provider_name or prompt.provider or "openai"
    selected_model = model or prompt.model

    def response_builder(current_prompt: PromptVariant, user_input: str) -> str:
        return call_provider(
            current_prompt,
            user_input,
            provider_name=selected_provider,
            api_key=api_key,
            model=selected_model,
        )

    return evaluate_prompt_variant(
        prompt,
        cases,
        response_builder=response_builder,
        store=store,
    )
