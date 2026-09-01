"""
evaluate_ragas.py
Step 12-13 in the flow: RAGAS Evaluator.

Loads the evaluation dataset built by run_rag_tests.py and scores it with RAGAS
using the free OpenRouter LLM as the evaluator, and local HuggingFace embeddings.

Metrics:
  - Faithfulness       -> Is the answer supported by the retrieved context?
  - Answer Relevance   -> Does the answer address the question?
  - Context Precision  -> How much of the retrieved context is relevant?
  - Context Recall     -> How much of the required info was retrieved?
"""
import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from llm import get_llm
from embeddings import get_embeddings

EVAL_DATASET_PATH = Path("data/eval_dataset.json")
RESULTS_PATH = Path("data/ragas_results.json")


def load_eval_dataset(path: Path = EVAL_DATASET_PATH) -> Dataset:
    with open(path, "r") as f:
        rows = json.load(f)
    return Dataset.from_list(rows)


def run_evaluation(dataset: Dataset):
    # Wrap our OpenRouter LLM + local embeddings so RAGAS can use them as the "judge"
    evaluator_llm = LangchainLLMWrapper(get_llm(temperature=0.0))
    evaluator_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    return results


def save_results(results, path: Path = RESULTS_PATH):
    df = results.to_pandas()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", indent=2)
    print(f"[evaluate_ragas] Saved detailed results to {path}")
    return df


if __name__ == "__main__":
    print("[evaluate_ragas] Loading evaluation dataset...")
    dataset = load_eval_dataset()

    print("[evaluate_ragas] Running RAGAS evaluation (this calls the free LLM per metric)...")
    results = run_evaluation(dataset)

    print("\n=== RAGAS Scores (averaged across test cases) ===")
    print(results)

    df = save_results(results)
    print("\n=== Per-question breakdown ===")
    cols = [c for c in ["user_input", "faithfulness", "answer_relevancy",
                         "context_precision", "context_recall"] if c in df.columns]
    print(df[cols].to_string(index=False))
