# Beam search for feature selection
# Start small (single-feature models) and then gradually grow them
# by adding more features to create multiple sets of features.
# At each stage, only keep the best few sets of features, because
# otherwise the problem space grows exponentially.

import asyncio
from typing import TypeAlias, cast

import numpy as np
from sklearn.model_selection import train_test_split  # type: ignore

from pastel.models import FEATURE_TYPE, BiasType
from pastel.optimise_weights import lin_reg
from pastel.pastel import EXAMPLES_TYPE, Pastel
from training.cached_pastel import CachedPastel
from training.crossvalidate_pastel import (
    evaluate_model,
    load_examples,
)

# One split of training data & test data
SplitData: TypeAlias = tuple[list[EXAMPLES_TYPE], list[EXAMPLES_TYPE]]


def load_data(
    num_splits: int = 1, data_filename: str = "data/example_training_data.jsonl"
) -> list[SplitData]:
    """Load labelled data set & split into train and test sets"""
    all_examples = load_examples(data_filename)
    all_splits = []
    for _ in range(num_splits):
        train_examples, test_examples = train_test_split(all_examples, test_size=0.5)
        all_splits.append((train_examples, test_examples))
    return all_splits


def add_one(
    current_features: frozenset[str], all_features: list[str]
) -> list[frozenset[str]]:
    """Take the current set and create a list of new sets, identical
    except each has one new, different feature added."""
    new_candidates = []
    for f in all_features:
        S = set(current_features)
        if f not in S:
            S.add(f)
            new_candidates.append(frozenset(S))
    return new_candidates


def final_pass(
    good_pool: list[frozenset[str]], all_splits: list[SplitData]
) -> tuple[Pastel | None, float]:
    """Take a shortlist of 'good' feature sets and do a final evaluation"""
    highest_score = -1.0
    best_model = None
    for candidate in good_pool:
        metrics, train_model = evaluate_pastel_set(candidate, all_splits, threshold=3.0)
        if metrics["f1"] > highest_score:
            highest_score = metrics["f1"]
            best_model = train_model
    return best_model, highest_score


def run_beam_search(
    all_features: list[str], beta: int = 3, max_iter: int | None = None
) -> tuple[Pastel | None, float]:
    """Main feature selection algorithm. Systematically add more and
    more features, but only keep the best 'beta' models at each iteration.
    See https://en.wikipedia.org/wiki/Beam_search for background.
    Each iteration adds one new feature, so max_iter is also the maximum number
    of features to be considered. If set to None, defaults to 'try all features'."""

    current_candidates: dict[frozenset[str], float] = {frozenset(): 0.0}
    evaluated_sets = []
    good_pool = []
    if not max_iter:
        max_iter = len(all_features)
    all_splits = load_data(num_splits=3)
    # train_examples, test_examples = all_splits[0]
    for i in range(0, max_iter):
        print(f"\nIteration {i}")
        # At each iteration, we take the current best few models and
        # consider adding each available feature to each of them
        scored_candidates = {}
        for candidate in current_candidates:
            new_candidates = add_one(candidate, all_features)
            if len(new_candidates) == 0:
                # We've run out of features to add
                break
            for nc in new_candidates:
                if nc not in evaluated_sets:
                    # only evaluate previously unseen sets of features
                    metrics, _ = evaluate_pastel_set(nc, all_splits, threshold=3.0)
                    scored_candidates[nc] = metrics["f1"]
                    evaluated_sets.append(nc)

        # Find the f1 score for the 'beta'-th best model:
        # Note that early on, all candidates may have f1=0, so keep them all
        all_f1 = sorted(scored_candidates.values(), reverse=True)
        if len(all_f1) > beta:
            f1_threshold = all_f1[beta - 1]
        elif len(all_f1) == 0:
            f1_threshold = 0
        else:
            f1_threshold = all_f1[-1]

        current_candidates = {
            fs: f1 for fs, f1 in scored_candidates.items() if f1 >= f1_threshold
        }
        print(f"Current candidates ({len(current_candidates)}):")
        for fs, f1 in current_candidates.items():
            print(f"F1: {f1:.4f} {' / '.join([str(f) for f in fs])}")
            good_pool.append(fs)
    print(
        f"Feature sets compared: {len(evaluated_sets)}; starting final pass of {len(good_pool)}."
    )
    # Process the store of best candidates and evaluate them to find the final best one
    best_features, best_f1 = final_pass(good_pool, all_splits)
    return best_features, best_f1


def train_model_from_examples(
    train_model: Pastel, train_examples: list[EXAMPLES_TYPE]
) -> Pastel:
    """Optimise weights of a model using the training set of sentences"""
    train_sentences = [ex[0] for ex in train_examples]
    # Get (maybe cached) responses to questions from genAI
    responses = asyncio.run(train_model.get_answers_to_questions(train_sentences))
    # Update each response with function responses too.
    for ts in train_sentences:
        responses_of_functions: dict[FEATURE_TYPE, float] = {
            cast(FEATURE_TYPE, f): float(f(ts)) for f in train_model.get_functions()
        }
        responses[ts].update(responses_of_functions)

    scores = train_model.quantify_answers(list(responses.values()))

    # Note: scores may be 'missing' a few rows if Gemini fails to respond
    # so now we need to remove the corresponding training examples
    score_lookup = {x[0]: x[1] for x in train_examples}
    train_scores_w_answers = [
        score_lookup[sentence]
        for sentence in responses.keys()
        if sentence in score_lookup
    ]
    # and now actually learn the weights and return the new model
    new_weights = lin_reg(scores, np.array(train_scores_w_answers))
    new_model = {
        feat: float(weight)
        for feat, weight in zip(train_model.model.keys(), new_weights)
    }
    new_pastel = Pastel(new_model)
    return new_pastel


def evaluate_pastel_set(
    question_subset: frozenset[str],
    all_splits: list[SplitData],
    threshold: float,
) -> tuple[dict[str, float], Pastel]:
    """Create a Pastel model from a set of features (which is a set of questions) and
    evaluate it.
    That model will predict answers & get scores for test set of sentences.
    If the threshold is high and the model not great, then F1 will be zero, which makes it
    hard to compare models (e.g. in the early stages of beam search).
    So to allow for a gradient descent, we can use smaller thresholds during the
    earlier stages of learning."""
    q_model: dict[FEATURE_TYPE, float] = {q: 0.0 for q in question_subset}
    q_model[BiasType.BIAS] = 0.0
    train_model = Pastel(q_model)
    cached_train_model = CachedPastel.from_pastel(train_model)
    all_metrics = []

    for train_examples, test_examples in all_splits:
        trained_model = train_model_from_examples(cached_train_model, train_examples)
        metrics = evaluate_model(trained_model, test_examples, threshold=threshold)
        all_metrics.append(metrics)

    # Calculate the average value of each key in the metrics list
    mean_metrics = {}
    for key in all_metrics[0].keys():
        mean_metrics[key] = sum(d[key] for d in all_metrics) / len(all_metrics)

    print("\nF1 scores:")
    _ = [print(m["f1"], end="\t") for m in all_metrics]
    print(mean_metrics["f1"])

    # combine train & test data to optimise best model with this feature set
    all_examples = all_splits[0][0] + all_splits[0][1]
    final_trained_model = train_model_from_examples(cached_train_model, all_examples)

    return mean_metrics, final_trained_model
