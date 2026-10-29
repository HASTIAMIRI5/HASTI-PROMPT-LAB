from __future__ import annotations

import argparse
import json
from typing import Any

from dotenv import load_dotenv

from prompt_lab.db import ExperimentStore
from prompt_lab.evaluator import ExperimentRunner, run_provider_experiment
from prompt_lab.models import PromptVariant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run prompt experiments in the Hasti Prompt Lab.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a prompt variant against a dataset.")
    run_parser.add_argument("--system", default="You are a helpful assistant.", help="System prompt.")
    run_parser.add_argument("--input", default="Summarize the benefits of clean, testable code.", help="User prompt input.")
    run_parser.add_argument("--dataset", type=str, help="Path to a JSON dataset with cases.")
    run_parser.add_argument("--name", default="baseline", help="Prompt variant name.")
    run_parser.add_argument("--provider", default="openai", help="Provider: openai or anthropic.")
    run_parser.add_argument("--model", default=None, help="Model name override.")
    run_parser.add_argument("--db", default="data/prompt_lab.db", help="SQLite database path for experiment storage.")
    run_parser.add_argument("--save", action="store_true", help="Persist the run in SQLite.")

    list_parser = subparsers.add_parser("list-runs", help="Display saved runs.")
    list_parser.add_argument("--db", default="data/prompt_lab.db", help="SQLite database path.")

    return parser


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list-runs":
        store = ExperimentStore(args.db)
        print(json.dumps(store.list_runs(), indent=2))
        return

    if not args.dataset:
        parser.error("--dataset is required for the run command.")

    store = ExperimentStore(args.db) if args.save else None
    prompt = PromptVariant(
        name=args.name,
        system_prompt=args.system,
        user_prompt=args.input,
        provider=args.provider,
        model=args.model,
    )

    cases = ExperimentRunner.load_cases(args.dataset)

    try:
        result = run_provider_experiment(
            prompt,
            cases,
            provider_name=args.provider,
            model=args.model,
            store=store,
        )
    except Exception as exc:  # pragma: no cover - friendly feedback for users
        print(f"Prompt execution failed: {exc}")
        raise SystemExit(1) from exc

    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
