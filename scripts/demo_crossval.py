"""Exhaustive question-set search by cross-validation.

The backend that answers the questions is chosen at runtime:

    python scripts/demo_crossval.py                  # Gemini (the default)
    python scripts/demo_crossval.py --backend local  # fine-tuned models
"""

import argparse
from typing import Type

import numpy as np

import training.crossvalidate_pastel as cvp
from local_models.questions import QUESTIONS
from pastel import BACKENDS, DEFAULT_BACKEND, PastelLocal, PastelModel, get_backend
from training.db_manager import DatabaseManager

TRAINING_DATA_PATH = "data/example_training_data.jsonl"


def report_score_ranges(data_filename: str) -> None:
    """Quick report of min/max/mean scores from a data set.
    The cross-validation module evaluate_model() function includes a
    threshold that should be somewhere in the middle of the range."""

    examples = cvp.load_examples(data_filename)
    true_scores = np.array([ex[1] for ex in examples])
    print(
        f"Range of target scores:  {np.min(true_scores):.3f} -- {np.max(true_scores):.3f}. "
    )
    print(f"Mean: {np.mean(true_scores):.3f} (sd. {np.std(true_scores):.2f})")


def get_questions(backend: Type[PastelModel]) -> list[str]:
    """The pool of questions to search over.

    The local backend is limited to the questions it has fine-tuned models for.
    Otherwise take every question in the cache database, i.e. every question
    that has been tried out and not deleted.
    """
    if issubclass(backend, PastelLocal):
        return list(QUESTIONS)
    db = DatabaseManager()
    return db.get_unique_questions()


def demo(backend: Type[PastelModel]) -> None:
    """Take a pool of questions, then try every combination (or at least many
    combinations) of them, build the corresponding regression model and
    calculate its f1 score.
    After the first few iterations, all the backend's responses should be in
    the cache, so it's just building/evaluating linear regression models which
    is quite fast. (Though millions of combinations will still take hours!)
    """
    all_questions = get_questions(backend)

    report_score_ranges(TRAINING_DATA_PATH)

    results = cvp.evaluate_question_combinations(
        questions=all_questions,
        data_filename=TRAINING_DATA_PATH,
        min_questions=8,
        max_questions=10,
        n_trials=2,
        backend=backend,
    )

    # Find the best performing combination
    best_combination = max(results.items(), key=lambda x: x[1]["test"]["mean"]["f1"])
    print("\nBEST set of questions:")
    _ = [print(" * ", q) for q in best_combination[0]]
    best_eval_scores = best_combination[1]["test"]
    for metric in best_eval_scores["mean"].keys():
        print(
            f"- {metric}: {best_eval_scores['mean'][metric]:.4f} ± {best_eval_scores['std'][metric]:.4f}"
        )

    # Also find the worst combination, just to demonstrate what "bad" looks like here:
    worst_combination = min(results.items(), key=lambda x: x[1]["test"]["mean"]["f1"])
    print("\nWORST set of questions:")
    _ = [print(" * ", q) for q in worst_combination[0]]
    worst_eval_scores = worst_combination[1]["test"]
    for metric in worst_eval_scores["mean"].keys():
        print(
            f"- {metric}: {worst_eval_scores['mean'][metric]:.4f} ± {worst_eval_scores['std'][metric]:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=sorted(BACKENDS),
        default=None,
        help="Which backend answers the questions. Defaults to the "
        f"PASTEL_BACKEND environment variable, or {DEFAULT_BACKEND}.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    demo(get_backend(parse_args().backend))
