# First attempt at asking a series of yes/no questions for checkworthiness etc., inspired by Sheffield's PASTEL model
# See paper: https://arxiv.org/abs/2309.07601v3 "Weakly Supervised Veracity Classification with LLM-Predicted Credibility Signals"

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Self, Sequence, Tuple, TypeAlias

import numpy as np
import numpy.typing as npt

from pastel import pastel_functions
from pastel.models import FEATURE_TYPE, BiasType, ScoreAndAnswers, Sentence

_logger = logging.getLogger(__name__)

EXAMPLES_TYPE = Tuple[Sentence, float]
ARRAY_TYPE: TypeAlias = npt.NDArray[np.float64]

# The key the bias term is stored under in a saved model file.
BIAS_KEY = "bias"


def feature_as_string(feature: FEATURE_TYPE) -> str:
    """The name a feature is saved and displayed under: the bias term is
    "bias", a function is its own name and a question is itself.
    load_model() reverses this."""
    if isinstance(feature, BiasType):
        return BIAS_KEY
    if callable(feature):
        return feature.__name__
    return str(feature)


class PastelModel(ABC):
    """
    Uses list of yes/no questions and functions to analyse a piece of text.
    Each of these features has an associated weight which is used to generate
    the final score for the text.
    The main model is a dict mapping features to weights.

    Subclasses of this abstract class must implement
    their own version of `get_answers_to_questions`.
    """

    def __init__(self, model: dict[FEATURE_TYPE, float]) -> None:
        """
        Create a new Pastel object from a list of questions and functions.
        A Pastel model is dict of features to weights. Exactly one
        entry should be BiasType.BIAS; zero or more may be features
        that are questions (ie strings) and zero or more may be
        are callable functions defined in the pastel_functions module.
        """
        self.model = model

        # assert bias term exists
        assert isinstance(self.get_bias(), float)

    def display_model(self) -> None:
        """Print the model's features and weights in a readable format."""
        print(f"{type(self).__name__} model:")
        for feature, weight in self.model.items():
            print(f"  {feature_as_string(feature):20}: {weight:.4f}")

    @classmethod
    def from_feature_list(cls, feature_names: Sequence[FEATURE_TYPE]) -> Self:
        """Take a list of features without weights. Initialise new
        model with all weights set to zero, ready for training"""
        new_model = dict()
        for feature in feature_names:
            # need to check which are pastel_functions and convert to Callables
            if feature in pastel_functions.__all__:
                new_model[getattr(pastel_functions, str(feature))] = 0.0
            else:
                new_model[feature] = 0.0
        new_model[BiasType.BIAS] = 0.0
        return cls(new_model)

    @classmethod
    def load_model(cls, model_file: str) -> Self:
        """Load model from JSON file. Convert any functions in the model
        from their names to Callable functions."""

        with open(model_file, "rt", encoding="utf-8") as json_in:
            model_json = json.load(json_in)
        # replace function names with function objects found in pastel_functions module
        new_model = {}
        for feature, weight in model_json.items():
            if feature in pastel_functions.__all__:
                new_model[getattr(pastel_functions, feature)] = weight
            elif feature == BIAS_KEY:
                new_model[BiasType.BIAS] = weight
            else:
                new_model[feature] = weight

        return cls(new_model)

    def save_model(self, model_path: str) -> None:
        """
        Save the questions, functions and associated weights to a local JSON file
        Convert callables to their names (strings) first
        """

        # Store the name of each function; all functions are in pastel_functions
        # so we know where to find them after re-loading a model.
        model_json = {
            feature_as_string(feature): float(weight)
            for feature, weight in self.model.items()
        }
        with open(model_path, "wt", encoding="utf-8") as json_out:
            json.dump(model_json, json_out, indent=2)

    def create_copy_with_different_model(
        self, model: dict[FEATURE_TYPE, float]
    ) -> "PastelModel":
        """Return a new model of the same kind - same backend, same caching -
        but with a different set of features and weights."""
        return type(self)(model)

    def get_bias(self) -> float:
        """Return just the bias weight"""
        # Every model will have a bias term. If this returns a key error, something's wrong with the model itself
        return self.model[BiasType.BIAS]

    def get_questions(self) -> list[str]:
        """Return just the questions of a model as a list of strings.
        (No weights are returned, nor are the bias term or function components)"""
        questions = []
        for feature in self.model.keys():
            if isinstance(feature, str):
                questions.append(feature)

        return questions

    def get_functions(self) -> list[Callable[[Sentence], float]]:
        """Return just the functions of a model as a list of strings.
        (No weights are returned, nor are the bias term or question components)"""
        functions = []
        for feature in self.model.keys():
            if callable(feature):
                functions.append(feature)

        return functions

    @abstractmethod
    async def get_answers_to_questions(
        self, sentences: list[Sentence]
    ) -> dict[Sentence, dict[FEATURE_TYPE, float]]:
        """
        Get answers for a given list of sentences.
        For each sentence, this Returns a dictionary mapping features to scores.

        Implementations may return fewer entries than they were given: a sentence
        the backend could not answer for should be omitted rather than given
        made-up answers. make_predictions() scores any omitted sentence as 0.0.
        """
        raise NotImplementedError

    def _get_function_answers_for_single_sentence(
        self, sentence: Sentence
    ) -> dict[FEATURE_TYPE, float]:
        """Runs all the functions in the model on the given sentence.
        These are computed locally, so this is shared by every backend."""
        return {f: f(sentence) for f in self.get_functions()}

    def quantify_answers(
        self, answers: Sequence[dict[FEATURE_TYPE, float]]
    ) -> ARRAY_TYPE:
        """Build numpy array of answers from list of dicts of answers, with one
        dict per sentence.
        Output array will have one row per sentence and one col per feature
        AND the order should match the features in the model, complete with bias column.
        """
        all_answers = []
        for sentence_answers in answers:
            numeric_answers = [0.0] * len(self.model)
            # read through dict of features, getting answer for each one = column
            for idx, feature in enumerate(self.model.keys()):
                if feature == BiasType.BIAS:
                    # We don't get an "answer" for the bias term - it's always 1.0
                    numeric_answers[idx] = 1.0
                else:
                    numeric_answers[idx] = sentence_answers[feature]

            # that's one row done... need to build a whole array!
            all_answers.append(numeric_answers)
        X = np.array(all_answers)
        return X

    def get_scores_from_answers(
        self, answers: Sequence[dict[FEATURE_TYPE, float]]
    ) -> ARRAY_TYPE:
        """Return the predicted score for each sentence.
        This is a linear regression model so the answers are theoretically unbounded,
        but will typically be in the range of the training data.
        answers_num: a numeric vector representing the answers to each question in turn,
        with 1.0 meaning 'yes' and 0.0 meaning 'no'. This will typically be the
        output from get_answers_to_questions()"""

        if sum([abs(w) for w in self.model.values()]) == 0:
            raise ValueError("Must train weights before predicting.")

        X = self.quantify_answers(answers)

        # then calculate & return the dot product, giving one score per sentence:
        weights = np.array(list(self.model.values()))
        scores = X.dot(weights)
        return scores

    async def make_predictions(
        self, sentences: list[Sentence]
    ) -> dict[Sentence, ScoreAndAnswers]:
        """Use the Pastel questions and weights model to generate
        a score for each of a list of sentences. Return this along with
        the questions and their scores."""
        answers = await self.get_answers_to_questions(sentences)
        if answers:
            scores = self.get_scores_from_answers(list(answers.values()))
        else:
            scores = np.array([])

        scores_dict = {}
        for sentence, score in zip(answers.keys(), scores):
            scores_dict[sentence.sentence_text] = float(score)

        for sentence in sentences:
            if sentence.sentence_text not in scores_dict:
                scores_dict[sentence.sentence_text] = 0.0
            if sentence not in answers.keys():
                answers[sentence] = {}

        return {
            sentence: ScoreAndAnswers(
                sentence=sentence,
                score=scores_dict[sentence.sentence_text],
                answers=answers[sentence],
            )
            for sentence in sentences
        }

    def update_predictions(
        self, sentences: list[Sentence], old_answers: list[dict[FEATURE_TYPE, float]]
    ) -> dict[Sentence, ScoreAndAnswers]:
        """Takes a list of sentences and their original LLM and function answers,
        then re-runs the functions only and updates the scores with these new answers.
        Returns ScoresAndAnswers for each sentence as before."""
        new_answers = [
            self._get_function_answers_for_single_sentence(sentence)
            for sentence in sentences
        ]
        updated_answers = [old | new for old, new in zip(old_answers, new_answers)]
        updated_scores = self.get_scores_from_answers(updated_answers)

        updated_scores_and_answers = {
            sentence: ScoreAndAnswers(
                sentence=sentence,
                score=score,
                answers=answers,
            )
            for sentence, score, answers in zip(
                sentences, updated_scores, updated_answers
            )
        }
        return updated_scores_and_answers
