# hasti-prompt-lab

AI prompt engineering sandbox for experiments and evaluation.

This version is a working prompt lab with provider integrations, backend storage, and a simple dashboard for comparing experiments.

## What is included

- typed prompt and experiment models
- real OpenAI and Anthropic provider adapters
- SQLite-backed experiment tracking
- reusable scoring and evaluation logic
- CLI execution for prompt experiments
- Streamlit dashboard for saved runs
- tests covering the core project behavior

## Project structure

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── examples/
│   └── demo_cases.json
├── src/
│   └── prompt_lab/
│       ├── __init__.py
│       ├── cli.py
│       ├── db.py
│       ├── evaluator.py
│       ├── models.py
│       ├── prompt_builder.py
│       ├── providers.py
│       └── __init__.py
├── streamlit_app.py
└── tests/
    └── test_prompt_lab.py
```

## Quick start

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2) Set your env vars

```bash
cp .env.example .env
```

Then set a provider key:

```bash
export OPENAI_API_KEY="your-key"
```

### 3) Run an experiment

```bash
hasti-prompt-lab run \
  --dataset examples/demo_cases.json \
  --provider openai \
  --name baseline \
  --system "You are a helpful assistant." \
  --input "Explain why good documentation matters." \
  --save
```

### 4) Launch the dashboard

```bash
streamlit run streamlit_app.py
```

## Example Python usage

```python
from prompt_lab.models import PromptVariant
from prompt_lab.evaluator import run_provider_experiment
from prompt_lab.db import ExperimentStore

variant = PromptVariant(
    name="baseline",
    system_prompt="You are a helpful assistant.",
    user_prompt="Explain why tests matter in software teams.",
    provider="openai",
)

cases = [
    {
        "id": "case-1",
        "input": "Why are tests valuable?",
        "expected_output": "Tests reduce regressions and increase confidence in changes.",
        "category": "quality",
    }
]

result = run_provider_experiment(
    variant,
    cases,
    provider_name="openai",
    store=ExperimentStore("data/prompt_lab.db"),
)
print(result.model_dump())
```

## License

MIT
