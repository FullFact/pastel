import numpy as np

import training.crossvalidate_pastel as cvp
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


def demo() -> None:
    """Take all the questions that exist in the cache database.
    (This is all questions that have been tried out and not deleted.)
    Then try every combination (or at least many combinations) of questions,
    build the corresponding regression model and calculate its f1 score.
    After the first few iterations, all the Gemini responses should be in the
    cache, so it's just building/evaluating linear regression models which is
    quite fast. (Though millions of combinations will still take hours!)
    """
    # Load questions from database
    db = DatabaseManager()
    all_questions = db.get_unique_questions()

    report_score_ranges(TRAINING_DATA_PATH)

    results = cvp.evaluate_question_combinations(
        questions=all_questions,
        data_filename=TRAINING_DATA_PATH,
        min_questions=8,
        max_questions=10,
        n_trials=2,
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


if __name__ == "__main__":
    demo()
