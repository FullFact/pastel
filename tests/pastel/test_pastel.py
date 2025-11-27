import json
import tempfile
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from pytest import mark, param

from pastel.models import Sentence
from pastel.pastel import BiasType, Pastel

Q1 = "Is the statement factual?"
Q2 = "Does the statement contain bias?"


@pytest.fixture
def pastel_instance() -> Pastel:
    pasteliser = Pastel(
        {
            BiasType.BIAS: 1.0,
            Q1: -3.0,
            Q2: 2.0,
        }
    )
    return pasteliser


def test_load_file(pastel_instance: Pastel) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".json"
    ) as temp_file:
        model = {
            "bias": 1.0,
            Q1: -3.0,
            Q2: 2.0,
        }
        json.dump(model, temp_file)
    loaded: Pastel = Pastel.load_model(temp_file.name)
    assert loaded.model == pastel_instance.model


def test_make_prompt(pastel_instance: Pastel) -> None:
    sentence = Sentence("The sky is blue.", tuple("quantity"))
    prompt = pastel_instance.make_prompt(sentence)
    assert "[QUESTIONS]" not in prompt
    assert "[SENT1]" not in prompt
    assert "The sky is blue." in prompt
    assert "Is the statement factual?" in prompt
    assert "Does the statement contain bias?" in prompt
    assert "Is this a load of old nonsense" not in prompt


def test_get_scores_from_answers(pastel_instance: Pastel) -> None:
    answers = [{Q1: 1.0, Q2: 1.0}, {Q1: 0.0, Q2: 1.0}]
    scores = pastel_instance.get_scores_from_answers(answers)
    expected_scores = np.array([0.0, 3.0])
    # [1.0 (=bias) + -3.0*1.0 + 2.0*1 = 0.0 ,
    #  1.0 + -3.0 * 0 + 2.0*1 = 3.0
    assert np.allclose(scores, expected_scores)


def test_get_scores_from_answers_no_weights(pastel_instance: Pastel) -> None:
    for k in pastel_instance.model.keys():
        pastel_instance.model[k] = 0.0
    answers = [{Q1: 1, Q2: 1}, {Q1: 0, Q2: 1}]
    with pytest.raises(ValueError):
        pastel_instance.get_scores_from_answers(answers)


def test_quantify_answers(pastel_instance: Pastel) -> None:
    answers = [{Q1: 1.0, Q2: 0.0}, {Q1: 1.0, Q2: 1.0}]
    numeric_answers = pastel_instance.quantify_answers(answers)
    print(numeric_answers)
    # One row of output per sentence (i.e. input dict):
    assert numeric_answers.shape[0] == len(answers)
    # First column is bias term so should be all 1's:
    # (NB: Model above defines first term is bias)
    assert all(x == 1 for x in numeric_answers[:, 0])
    # Given no sentences, return no answers
    assert pastel_instance.quantify_answers([]).shape[0] == 0


@patch(
    "pastel.pastel.run_prompt",
    side_effect=ValueError("Gemini failed"),
)
async def test_retries(mock_run_prompt: AsyncMock, pastel_instance: Pastel) -> None:
    sentence = Sentence("This is a claim.", tuple("quantity"))
    try:
        await pastel_instance._get_answers_for_single_sentence(sentence)
        assert False
    except Exception:
        assert True

    assert mock_run_prompt.call_count == 3


@mark.parametrize(
    "sentences,return_values,expected",
    [
        param(
            [Sentence("s1", ()), Sentence("s2", ())],
            [{Q1: 1.0, Q2: 1.0}, {Q1: 1.0, Q2: 0.0}],
            {
                Sentence("s1", ()): {Q1: 1.0, Q2: 1.0},
                Sentence("s2", ()): {Q1: 1.0, Q2: 0.0},
            },
            id="Normal case",
        ),
        param(
            [Sentence("s1", ()), Sentence("s2", ())],
            [{Q1: 1.0, Q2: 1.0}, ValueError()],
            {Sentence("s1", ()): {Q1: 1.0, Q2: 1.0}},
            id="One sentence fails",
        ),
        param(
            [Sentence("s1", ()), Sentence("s2", ())],
            [ValueError(), ValueError()],
            {},
            id="All sentences fail",
        ),
    ],
)
async def test_get_answers_to_questions(
    sentences: list[Sentence],
    return_values: list[dict[str, float] | BaseException],
    expected: dict[str, dict[str, float]],
    pastel_instance: Pastel,
):
    with patch.object(
        pastel_instance, "_get_answers_for_single_sentence", side_effect=return_values
    ):
        answers = await pastel_instance.get_answers_to_questions(sentences)
        assert answers == expected


@mark.parametrize(
    "sentences,answers,expected",
    [
        param(
            ["s1", "s2"],
            {"s1": {Q1: 0.0, Q2: 1.0}, "s2": {Q1: 0.0, Q2: 0.5}},
            np.array([3.0, 2.0]),
            id="Normal case",
        ),
        param(
            ["s1", "s2"],
            {"s1": {Q1: 0.0, Q2: 1.0}},
            np.array([3.0, 0.0]),
            id="One sentence fails",
        ),
        param(
            ["s1", "s2"],
            {},
            np.array([0.0, 0.0]),
            id="All sentences fail",
        ),
    ],
)
def test_make_predictions(
    sentences: list[str],
    answers: dict[str, dict[str, float]],
    expected: np.ndarray,
    pastel_instance: Pastel,
):
    with patch.object(
        pastel_instance, "get_answers_to_questions", return_value=answers
    ):
        predictions = pastel_instance.make_predictions(sentences)
        assert all(predictions == expected)
