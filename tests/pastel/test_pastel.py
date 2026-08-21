import json
import tempfile
from unittest.mock import call, patch

import numpy as np
import pytest
from pytest import mark, param

from pastel.models import FEATURE_TYPE, BiasType, ScoreAndAnswers, Sentence
from pastel.pastel import PastelModel

# mypy: ignore-errors
# getting "Untyped decorator makes function ... untyped " so ignoring for now:

Q1: FEATURE_TYPE = "Is the statement factual?"
Q2: FEATURE_TYPE = "Does the statement contain bias?"


class DummyPastel(PastelModel):
    """PastelModel is abstract, so the shared behaviour is tested through a
    backend that answers nothing. Tests that need answers patch them in."""

    async def get_answers_to_questions(
        self, sentences: list[Sentence]
    ) -> dict[Sentence, dict[FEATURE_TYPE, float]]:
        return {}


@pytest.fixture
def pastel_instance() -> PastelModel:
    pasteliser = DummyPastel({BiasType.BIAS: 1.0, Q1: -3.0, Q2: 2.0})
    return pasteliser


def test_load_file(pastel_instance: PastelModel) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".json"
    ) as temp_file:
        model = {
            "bias": 1.0,
            Q1: -3.0,
            Q2: 2.0,
        }
        json.dump(model, temp_file)
    loaded: PastelModel = DummyPastel.load_model(temp_file.name)
    assert loaded.model == pastel_instance.model


def test_save_load_round_trip_with_functions() -> None:
    """Functions and the bias term are saved by name and come back as the same
    features, so a saved model is the model that was trained."""
    model = DummyPastel.from_feature_list([Q1, "is_claim_type_quantity"])
    model.model = {feature: 1.5 for feature in model.model}

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".json"
    ) as temp_file:
        path = temp_file.name
    model.save_model(path)

    with open(path, "rt", encoding="utf-8") as json_in:
        assert set(json.load(json_in)) == {Q1, "is_claim_type_quantity", "bias"}

    assert DummyPastel.load_model(path).model == model.model


def test_with_model(pastel_instance: PastelModel) -> None:
    """A new model of the same kind, with different features and weights."""
    updated = pastel_instance.create_copy_with_different_model(
        {BiasType.BIAS: 0.5, Q1: 1.0}
    )
    assert isinstance(updated, DummyPastel)
    assert updated.model == {BiasType.BIAS: 0.5, Q1: 1.0}
    # the original is untouched
    assert pastel_instance.get_questions() == [Q1, Q2]


def test_get_scores_from_answers(pastel_instance: PastelModel) -> None:
    answers = [{Q1: 1.0, Q2: 1.0}, {Q1: 0.0, Q2: 1.0}]
    scores = pastel_instance.get_scores_from_answers(answers)
    expected_scores = np.array([0.0, 3.0])
    # [1.0 (=bias) + -3.0*1.0 + 2.0*1 = 0.0 ,
    #  1.0 + -3.0 * 0 + 2.0*1 = 3.0
    assert np.allclose(scores, expected_scores)


def test_get_scores_from_answers_no_weights(pastel_instance: PastelModel) -> None:
    for k in pastel_instance.model.keys():
        pastel_instance.model[k] = 0.0
    answers = [{Q1: 1.0, Q2: 1.0}, {Q1: 0.0, Q2: 1.0}]
    with pytest.raises(ValueError):
        pastel_instance.get_scores_from_answers(answers)


def test_quantify_answers(pastel_instance: PastelModel) -> None:
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
    pastel_instance: PastelModel,
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
