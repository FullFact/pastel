import json
import tempfile
from unittest.mock import AsyncMock, call, patch

import numpy as np
import pytest
from pytest import mark, param

from pastel.models import FEATURE_TYPE, BiasType, ScoreAndAnswers, Sentence
from pastel.pastel import Pastel

# mypy: ignore-errors
# getting "Untyped decorator makes function ... untyped " so ignoring for now:

Q1: FEATURE_TYPE = "Is the statement factual?"
Q2: FEATURE_TYPE = "Does the statement contain bias?"


@pytest.fixture
def pastel_instance() -> Pastel:
    pasteliser = Pastel({BiasType.BIAS: 1.0, Q1: -3.0, Q2: 2.0})
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
    answers = [{Q1: 1.0, Q2: 1.0}, {Q1: 0.0, Q2: 1.0}]
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
    "pastel.pastel.run_prompt_async",
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
            [Sentence("s1", tuple("quantity")), Sentence("s2", tuple("quantity"))],
            [{Q1: 1.0, Q2: 1.0}, {Q1: 1.0, Q2: 0.0}],
            {
                Sentence("s1", tuple("quantity")): {Q1: 1.0, Q2: 1.0},
                Sentence("s2", tuple("quantity")): {Q1: 1.0, Q2: 0.0},
            },
            id="Normal case",
        ),
        param(
            [Sentence("s1", tuple("quantity")), Sentence("s2", tuple("quantity"))],
            [{Q1: 1.0, Q2: 1.0}, ValueError()],
            {Sentence("s1", tuple("quantity")): {Q1: 1.0, Q2: 1.0}},
            id="One sentence fails",
        ),
        param(
            [Sentence("s1", tuple("quantity")), Sentence("s2", tuple("quantity"))],
            [ValueError(), ValueError()],
            {},
            id="All sentences fail",
        ),
    ],
)
async def test_get_answers_to_questions(
    sentences: list[Sentence],
    return_values: list[dict[str, float] | BaseException],
    expected: dict[Sentence, dict[str, float]],
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
            [Sentence("s1", tuple("quantity")), Sentence("s2", tuple("quantity"))],
            {
                Sentence("s1", tuple("quantity")): {Q1: 0.0, Q2: 1.0},
                Sentence("s2", tuple("quantity")): {Q1: 0.0, Q2: 0.5},
            },
            {
                Sentence("s1", tuple("quantity")): ScoreAndAnswers(
                    sentence=Sentence("s1", tuple("quantity")),
                    score=3.0,
                    answers={Q1: 0.0, Q2: 1.0},
                ),
                Sentence("s2", tuple("quantity")): ScoreAndAnswers(
                    sentence=Sentence("s2", tuple("quantity")),
                    score=2.0,
                    answers={Q1: 0.0, Q2: 0.5},
                ),
            },
            id="Normal case",
        ),
        param(
            [Sentence("s1", tuple("quantity")), Sentence("s2", tuple("quantity"))],
            {Sentence("s1", tuple("quantity")): {Q1: 0.0, Q2: 1.0}},
            {
                Sentence("s1", tuple("quantity")): ScoreAndAnswers(
                    sentence=Sentence("s1", tuple("quantity")),
                    score=3.0,
                    answers={Q1: 0.0, Q2: 1.0},
                ),
                Sentence("s2", tuple("quantity")): ScoreAndAnswers(
                    sentence=Sentence("s2", tuple("quantity")), score=0.0, answers={}
                ),
            },
            id="One sentence fails",
        ),
        param(
            [Sentence("s1", tuple("quantity")), Sentence("s2", tuple("quantity"))],
            {},
            {
                Sentence("s1", tuple("quantity")): ScoreAndAnswers(
                    sentence=Sentence("s1", tuple("quantity")), score=0.0, answers={}
                ),
                Sentence("s2", tuple("quantity")): ScoreAndAnswers(
                    sentence=Sentence("s2", tuple("quantity")), score=0.0, answers={}
                ),
            },
            id="All sentences fail",
        ),
    ],
)
async def test_make_predictions(
    sentences: list[Sentence],
    answers: dict[str, dict[str, float]],
    expected: dict[Sentence, ScoreAndAnswers],
    pastel_instance: Pastel,
):
    with patch.object(
        pastel_instance, "get_answers_to_questions", return_value=answers
    ):
        predictions = await pastel_instance.make_predictions(sentences)
        assert predictions == expected


def test_update_predictions(pastel_instance):
    sentences = [
        Sentence(c, tuple("quantity")) for c in ["claim 1", "claim 2", "claim 3"]
    ]
    old_answers = [{Q1: 1.0, Q2: 0.0}, {Q1: 0.0, Q2: 1.0}, {Q1: 1.0, Q2: 1.0}]

    with (
        patch.object(
            pastel_instance,
            "_get_function_answers_for_single_sentence",
            return_value={"updated_feature": 1.0},
        ) as mock_get_func_answers,
        patch.object(
            pastel_instance,
            "get_scores_from_answers",
            return_value=np.array([1.0, 2.0, 3.0]),
        ) as mock_get_scores,
    ):
        updates = pastel_instance.update_predictions(sentences, old_answers)

        mock_get_func_answers.assert_has_calls(
            [call(sentence) for sentence in sentences]
        )

        mock_get_scores.assert_called_once()

        assert len(updates) == len(sentences)
        for sentence, score, old_answer in zip(sentences, [1.0, 2.0, 3.0], old_answers):
            assert sentence in updates
            assert isinstance(updates[sentence], ScoreAndAnswers)
            assert updates[sentence].sentence == sentence
            assert updates[sentence].score == score
            expected_answers = old_answer | {"updated_feature": 1.0}
            assert updates[sentence].answers == expected_answers
