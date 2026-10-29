from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from prompt_lab.db import ExperimentStore


def load_runs(db_path: str = "data/prompt_lab.db") -> pd.DataFrame:
    store = ExperimentStore(db_path)
    rows = store.list_runs()
    if not rows:
        return pd.DataFrame(
            columns=[
                "run_id",
                "variant_name",
                "provider",
                "model",
                "dataset_name",
                "created_at",
                "average_score",
                "accuracy",
                "clarity",
                "passed",
            ]
        )

    payload: list[dict[str, object]] = []
    for row in rows:
        metrics = row["metrics"]
        if isinstance(metrics, str):
            metrics = json.loads(metrics)
        payload.append(
            {
                "run_id": row["run_id"],
                "variant_name": row["variant_name"],
                "provider": row["provider"],
                "model": row["model"],
                "dataset_name": row["dataset_name"],
                "created_at": row["created_at"],
                "average_score": metrics.get("average_score", 0.0),
                "accuracy": metrics.get("accuracy", 0.0),
                "clarity": metrics.get("clarity", 0.0),
                "passed": metrics.get("passed", False),
            }
        )
    return pd.DataFrame(payload)


def main() -> None:
    st.title("Hasti Prompt Lab")
    st.caption("Compare prompt variants across real LLM providers and saved evaluations.")

    db_path = st.sidebar.text_input("SQLite database path", value="data/prompt_lab.db")
    runs = load_runs(db_path)

    if runs.empty:
        st.info("No saved runs yet. Run a prompt experiment with the CLI to populate the database.")
        return

    st.dataframe(runs, use_container_width=True)

    selected = st.selectbox("Choose a run", runs["run_id"].tolist())
    row = runs[runs["run_id"] == selected].iloc[0]
    st.metric("Average score", f"{row['average_score']:.3f}")
    st.metric("Accuracy", f"{row['accuracy']:.3f}")
    st.metric("Clarity", f"{row['clarity']:.3f}")
    st.write(
        {
            "provider": row["provider"],
            "model": row["model"],
            "dataset": row["dataset_name"],
            "created_at": row["created_at"],
            "passed": row["passed"],
        }
    )


if __name__ == "__main__":
    main()
