"""Performs linear regression to optimise & save model weights"""

import asyncio
import json
import logging
from typing import TypedDict

import numpy as np
from scipy.optimize import least_squares

from pastel import pastel
from pastel.models import Sentence


class SCORED_EXAMPLES_TYPE(TypedDict):
    sentence_text: str
    score: float
    claim_types: list[str]


_logger = logging.getLogger(__name__)


def lin_reg(X: pastel.ARRAY_TYPE, y: pastel.ARRAY_TYPE) -> pastel.ARRAY_TYPE:
    """Calculates optimum weight vector of linear regression model. This is the
    best-fit line through the X,y data.
    Minimise squared error for y = w.x
    One column of X should be all 1's corresponding to the bias (or intercept
    term) in the model. Without it, the line would always go through the
    origin (0,0) which is an unnecessary constraint."""

    def residuals(ww: pastel.ARRAY_TYPE) -> pastel.ARRAY_TYPE:
        """Define the residual function.
        This calculates the difference between the predicted values (y) and the actual values X.ww
        The smaller the residuals, the better the fit.
        """
        return X @ ww - y

    # Initial guess for the weight vector (including the bias term)
    w0 = np.ones(X.shape[1])

    # Use least squares to minimize the residuals. This calculates the set of weights that
    # produces the smallest sum of squared errors for the training data - i.e. the best fit.
    result = least_squares(residuals, w0)

    return result.x


def load_examples(filename: str) -> list[SCORED_EXAMPLES_TYPE]:
    """Load examples from file. Each row in the JSONL file should be a like
    { "sentence_text": "Of these respondents....",  "score": "4.0",  "claim_types": [
    "quantity", "personal" ]}
    The "score" field is for checkworthiness, conventionally in the range 1-5"""
    examples = []
    with open(filename, "rt", encoding="utf-8") as fin:
        for line in fin:
            example = json.loads(line)
            examples.append(example)
    return examples


def learn_weights(
    training_data_filename: str, pasteliser: pastel.Pastel
) -> pastel.ARRAY_TYPE:
    """Minimise sum squared error of labelled data set to find optimal
    set of weights. Note that first weight is for a constant term, so the
    weight vector is one longer than the number of questions in the prompt."""

    scored_examples = load_examples(training_data_filename)
    examples = [
        Sentence(ex["sentence_text"], tuple(ex["claim_types"]))
        for ex in scored_examples
    ]

    answers = asyncio.run(pasteliser.get_answers_to_questions(examples))
    predictions = pasteliser.quantify_answers(list(answers.values()))

    score_lookup = {x["sentence_text"]: float(x["score"]) for x in scored_examples}
    targs = [score_lookup[sentence.sentence_text] for sentence in answers.keys()]

    targs_arr = np.array(targs)
    pred_arr = np.array(predictions)
    weights = lin_reg(pred_arr, targs_arr)

    for idx, k in enumerate(pasteliser.model.keys()):
        pasteliser.model[k] = weights[idx]
    return weights
