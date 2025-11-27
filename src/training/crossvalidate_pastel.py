"""
Cross-validation utilities for evaluating Pastel models.
This can be used to compare millions of different combinations of questions in
order to find a good set. The combinatorial explosion means that even with
just 10-20 questions, this may take hours to run.
"""

import asyncio
import json
import time
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

from pastel.models import Sentence
from pastel.optimise_weights import lin_reg
from pastel.pastel import EXAMPLES_TYPE, FEATURE_TYPE, BiasType, Pastel
from training.cached_pastel import CachedPastel


def load_examples(filename: str) -> List[EXAMPLES_TYPE]:
    """
    Load examples from JSONL file. Each row should be a sentence followed by its score.

    Args:
        filename: Path to JSONL file containing sentences, claim types and scores

    Returns:
        List of (sentence, score) tuples
    """
    examples = []
    with open(filename, "rt", encoding="utf-8") as fin:
        # reader = csv.reader(fin, quoting=csv.QUOTE_ALL)
        for row in fin:
            obj = json.loads(row)
            # sentence = row[0]
            # score = float(row[1])
            examples.append(
                (
                    Sentence(str(obj["sentence_text"]), tuple(obj["claim_types"])),
                    float(obj["score"]),
                ),
            )
    return examples


def prepare_crossvalidation_data(
    pastel: Pastel, data_filename: str, random_seed: int | None
) -> Tuple[List[EXAMPLES_TYPE], List[EXAMPLES_TYPE]]:
    """
    Prepare data for cross-validation by:
    1. Loading all examples from the data file
    2. Caching responses for all sentences using CachedPastel
    3. Splitting data into train and test sets

    Note that the cache is a local persistent database, so generating the cached
    responses in step 2 will call Gemini multiple times but only the first time
    this function is called (or when new questions are added).

    Args:
        pastel: Pastel model to evaluate
        data_filename: Path to CSV file containing sentences and scores
        random_seed: Random seed for reproducible splits.

    Returns:
        Tuple of (train_examples, test_examples) where each is a list of (sentence, score) tuples
    """
    # Load all examples
    examples = load_examples(data_filename)

    # Cache responses for all sentences as needed
    cached_model = CachedPastel.from_pastel(pastel)
    all_sentences = [ex[0] for ex in examples]
    _ = asyncio.run(cached_model.get_answers_to_questions(all_sentences))

    # Split into train and test sets
    train_examples, test_examples = train_test_split(
        examples, test_size=0.5, random_state=random_seed
    )

    return train_examples, test_examples


def evaluate_model(
    model: Pastel, examples: List[EXAMPLES_TYPE], threshold: float = 3.0
) -> Dict[str, float]:
    """
    Evaluate model performance on a set of examples using classification metrics.
    Scores are converted to binary classifications using the threshold value.

    Args:
        model: Trained Pastel model to evaluate
        examples: List of (sentence, score) tuples
        threshold: Score threshold for binary classification
            Scores >= threshold are considered positive class
            Scores < threshold are considered negative class

    Returns:
        Dictionary containing classification metrics: precision, recall & f1
    """
    # Get model predictions
    sentences = [ex[0] for ex in examples]
    cached_model = CachedPastel.from_pastel(model)
    predictions = asyncio.run(cached_model.get_answers_to_questions(sentences))
    answers = list(predictions.values())
    predicted_scores = cached_model.get_scores_from_answers(answers)

    # Note: predicted scores may be 'missing' a few rows if Gemini fails to respond
    # so now we need to keep just the training examples where we have predictions
    train_score_lookup = {ex[0]: ex[1] for ex in examples}
    train_scores_w_answers = np.array(
        [train_score_lookup[sent] for sent in predictions.keys()]
    )

    # Convert continuous scores to binary classifications for evaluation purposes
    y_true = (train_scores_w_answers >= threshold).astype(int)
    y_pred = (predicted_scores >= threshold).astype(int)

    # Calculate classification metrics
    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    return metrics


def run_crossvalidation(
    pastel: Pastel,
    data_filename: str,
    n_trials: int = 5,
    random_seed: int | None = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Run complete cross-validation multiple times and return statistics:
    1. Split data into train/test sets
    2. Train model on training data
    3. Evaluate on both training and test sets
    4. Repeat for n_trials with different random splits
    5. Calculate mean and standard deviation of metrics

    Args:
        pastel: Initial Pastel model (weights will be updated during training)
        data_filename: Path to CSV file containing sentences and scores
        n_trials: Number of trials to run with different random splits (default: 5)
        random_seed: Base random seed for reproducible splits (default: 42)

    Returns:
        Tuple of (train_metrics, test_metrics) dictionaries containing:
        - mean: Mean value for each metric across trials
        - std: Standard deviation for each metric across trials
    """
    # Initialize arrays to store metrics for each trial
    train_metrics_list = []
    test_metrics_list = []

    print(f"\nRunning {n_trials} cross-validation trials...", end=" ")

    for trial in range(n_trials):
        # Use a different random seed for each trial
        trial_seed = random_seed + trial if random_seed else None

        # Prepare data with this trial's random seed
        train_examples, test_examples = prepare_crossvalidation_data(
            pastel, data_filename, random_seed=trial_seed
        )

        # Train model on training data
        train_sentences = [ex[0] for ex in train_examples]

        # Create a new model for training to avoid modifying the input model
        train_model = Pastel(pastel.model)
        cached_train_model = CachedPastel.from_pastel(train_model)

        # Get cached responses and learn weights
        responses = asyncio.run(
            cached_train_model.get_answers_to_questions(train_sentences)
        )
        scores = np.array(cached_train_model.quantify_answers(list(responses.values())))

        # Note: scores may be 'missing' a few rows if Gemini fails to respond
        # so now we need to remove the corresponding training examples

        score_lookup = {x[0]: x[1] for x in train_examples}
        train_scores_w_answers = [
            score_lookup[sentence]
            for sentence in responses.keys()
            if sentence in score_lookup
        ]
        new_weights = lin_reg(scores, np.array(train_scores_w_answers))
        train_model.model = {
            feat: weight for feat, weight in zip(train_model.model.keys(), new_weights)
        }

        # Evaluate on both sets
        train_metrics = evaluate_model(train_model, train_examples)
        test_metrics = evaluate_model(train_model, test_examples)

        train_metrics_list.append(train_metrics)
        test_metrics_list.append(test_metrics)

    # Calculate statistics across trials
    def calculate_stats(
        metrics_list: List[Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {"mean": {}, "std": {}}
        for metric in metrics_list[0].keys():
            values = [m[metric] for m in metrics_list]
            stats["mean"][metric] = float(np.mean(values))
            stats["std"][metric] = float(np.std(values))
        return stats

    train_stats = calculate_stats(train_metrics_list)
    test_stats = calculate_stats(test_metrics_list)

    # Print results
    print("Mean test results from {n_trials} trials:")
    for metric in test_stats["mean"]:
        print(
            f"- {metric}: {test_stats['mean'][metric]:.4f} ± {test_stats['std'][metric]:.4f}",
            end="\t",
        )
    print()

    return train_stats, test_stats


def evaluate_question_combinations(
    questions: List[str],
    data_filename: str,
    min_questions: int = 1,
    max_questions: int = 20,
    n_trials: int = 3,
    random_seed: int | None = None,
) -> Dict[Tuple[str, ...], Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Evaluate all possible combinations of questions by running cross-validation on each subset.
    Each combination of questions is used to build and evaluated a Pastel model.
    The evaluation data will be split randomly into train and evaluation parts; this is repeated
    for 'n_trials' experiments on each combination of questions.

    Args:
        questions: List of questions to evaluate combinations of
        data_filename: Path to CSV file containing sentences and scores
        min_questions: Minimum number of questions in each combination
        max_questions: Maximum number of questions in each combination
        n_trials: Number of cross-validation trials for each combination
        random_seed: Base random seed for reproducible splits

    Returns:
        Dictionary mapping question combinations to their performance metrics:
        {
            ('question1', 'question2'): {
                'train': {
                    'mean': {'precision': 0.8, 'recall': 0.7, 'f1': 0.75},
                    'std': {'precision': 0.1, 'recall': 0.1, 'f1': 0.1}
                },
                'test': {
                    'mean': {'precision': 0.75, 'recall': 0.65, 'f1': 0.7},
                    'std': {'precision': 0.12, 'recall': 0.11, 'f1': 0.11}
                }
            },
            ...
        }
    """
    if max_questions > len(questions):
        max_questions = len(questions)

    results: Dict[Tuple[str, ...], Dict[str, Dict[str, Dict[str, float]]]] = {}
    total_combinations = sum(
        1
        for i in range(min_questions, max_questions + 1)
        for _ in combinations(questions, i)
    )

    print(
        f"\nEvaluating {total_combinations:,} combinations of between {min_questions} and {max_questions} questions, out of {len(questions)} questions in total..."
    )
    combination_count = 0

    start_time = time.time()
    last_update_time = start_time

    # Generate and evaluate all combinations
    for n_questions in range(min_questions, max_questions + 1):
        for question_subset in combinations(questions, n_questions):
            combination_count += 1
            # combination_start_time = time.time()

            # Create a new model with this subset of questions
            q_model: dict[FEATURE_TYPE, float] = {q: 0.0 for q in question_subset}
            q_model[BiasType.BIAS] = 0.0
            model = Pastel(q_model)

            # Run cross-validation
            train_stats, test_stats = run_crossvalidation(
                model, data_filename, n_trials=n_trials, random_seed=random_seed
            )

            # Store results
            results[question_subset] = {"train": train_stats, "test": test_stats}

            # Calculate timing information
            # combination_time = time.time() - combination_start_time
            elapsed_time = time.time() - start_time
            avg_time_per_combination = elapsed_time / combination_count
            remaining_combinations = total_combinations - combination_count
            estimated_time_remaining = remaining_combinations * avg_time_per_combination

            # Print progress update periodically
            current_time = time.time()
            if current_time - last_update_time >= 60 or combination_count % 10 == 0:
                last_update_time = current_time
                print(
                    f"\nProgress update: (after {combination_count} combinations out of {total_combinations})"
                )
                print(
                    f"Average time per combination: {avg_time_per_combination:.1f} seconds"
                )
                print(f"Total elapsed time: {elapsed_time/60:.1f} minutes")
                print(
                    f"Estimated time remaining: {estimated_time_remaining/60:.1f} minutes"
                )

                # Print best f1 score so far and corresponding questions
                best_f1 = max(
                    (result_dict["test"]["mean"]["f1"], qs)
                    for qs, result_dict in results.items()
                )
                print(
                    f"\nBest test F1 so far: {best_f1[0]:.4f} "
                    f"(using {len(best_f1[1])} questions)"
                )
                _ = [print("   * ", q) for q in best_f1[1]]
                # Need to add bias term to new model, as it's currently just a list of questions
                pastel_to_save = Pastel.from_feature_list(list(best_f1[1]))
                pastel_to_save.save_model(f"best_so_far{len(best_f1[1])}.json")

    # Print final timing summary
    total_time = time.time() - start_time
    print("\nEvaluation complete!")
    print(
        f"Total time: {total_time/60:.1f} minutes to process {combination_count} models"
    )
    print(f"Average time per combination: {total_time/combination_count:.3f} seconds")

    return results
