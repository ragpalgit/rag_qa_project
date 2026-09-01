"""
run_rag_tests.py
Step 10-11 in the flow: Test Runner.

For each test case:
  1. Read Question
  2. Run rag_graph (RAG)
  3. Get Retrieved Contexts
  4. Get LLM Response
  5. Collect Reference

Produces an Evaluation Dataset (list of dicts) saved to disk for RAGAS.
"""
import json
import time
from pathlib import Path

from rag_graph import rag_app

TEST_CASES_PATH = Path("data/rag_test_cases.json")
EVAL_DATASET_PATH = Path("data/eval_dataset.json")


def load_test_cases(path: Path = TEST_CASES_PATH) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def run_tests(test_cases: list[dict], sleep_between: float = 1.0) -> list[dict]:
    """Run each test case through the RAG graph and build the evaluation dataset."""
    eval_rows = []

    for i, case in enumerate(test_cases, 1):
        question = case["user_input"]
        reference = case["reference"]

        print(f"[run_rag_tests] ({i}/{len(test_cases)}) Running: {question!r}")

        result = rag_app.invoke({"question": question})

        eval_rows.append({
            "user_input": question,
            "retrieved_contexts": result["context"],
            "response": result["answer"],
            "reference": reference,
        })

        # Small delay to be polite to free-tier rate limits on OpenRouter
        if sleep_between and i < len(test_cases):
            time.sleep(sleep_between)

    return eval_rows


def save_eval_dataset(eval_rows: list[dict], path: Path = EVAL_DATASET_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(eval_rows, f, indent=2)
    print(f"[run_rag_tests] Saved evaluation dataset to {path}")


if __name__ == "__main__":
    cases = load_test_cases()
    rows = run_tests(cases)
    save_eval_dataset(rows)

    print("\n=== Sample result ===")
    print("Q:", rows[0]["user_input"])
    print("A:", rows[0]["response"])
    print("Contexts retrieved:", len(rows[0]["retrieved_contexts"]))
