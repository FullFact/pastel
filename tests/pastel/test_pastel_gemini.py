"""Tests for the Gemini-backed Pastel model. These moved out of test_pastel.py
when the Gemini prompting was split out of the PastelModel base class."""

from unittest.mock import AsyncMock, patch

import pytest
import tenacity
from pytest import mark, param

from pastel.models import FEATURE_TYPE, BiasType, Sentence
from pastel.pastel_gemini import PastelGemini

# mypy: ignore-errors
# getting "Untyped decorator makes function ... untyped " so ignoring for now:

Q1: FEATURE_TYPE = "Is the statement factual?"
Q2: FEATURE_TYPE = "Does the statement contain bias?"


@pytest.fixture
def pastel_instance() -> PastelGemini:
    return PastelGemini({BiasType.BIAS: 1.0, Q1: -3.0, Q2: 2.0})


def test_make_prompt(pastel_instance: PastelGemini) -> None:
    sentence = Sentence("The sky is blue.", tuple("quantity"))
    prompt = pastel_instance._make_prompt(sentence)
    assert "[QUESTIONS]" not in prompt
    assert "[SENT1]" not in prompt
    assert "The sky is blue." in prompt
    assert "Is the statement factual?" in prompt
    assert "Does the statement contain bias?" in prompt
    assert "Is this a load of old nonsense" not in prompt


@patch(
    "pastel.pastel_gemini.run_prompt_async",
    side_effect=ValueError("Gemini failed"),
)
async def test_retries(
    mock_run_prompt: AsyncMock, pastel_instance: PastelGemini
) -> None:
    sentence = Sentence("This is a claim.", tuple("quantity"))
    with pytest.raises(Exception):
        await pastel_instance._get_answers_for_single_sentence(sentence)

    assert mock_run_prompt.call_count == 3


@mark.parametrize(
    "raw_output,expected",
    [
        param("0. Yes\n1. No", {Q1: 1.0, Q2: 0.0}, id="yes/no"),
        param("0. yes\n1. unsure", {Q1: 1.0, Q2: 0.5}, id="unsure maps to 0.5"),
    ],
)
async def test_parses_llm_answers(
    raw_output: str,
    expected: dict[FEATURE_TYPE, float],
    pastel_instance: PastelGemini,
) -> None:
    with patch(
        "pastel.pastel_gemini.run_prompt_async",
        new=AsyncMock(return_value=raw_output),
    ):
        answers = await pastel_instance._get_llm_answers_for_single_sentence(
            Sentence("a claim", tuple("quantity"))
        )
    assert answers == expected


async def test_wrong_number_of_answers_raises(pastel_instance: PastelGemini) -> None:
    """A reply that doesn't line up with the questions must not be guessed at.
    ValueError is retryable, so tenacity re-raises it as RetryError."""
    with patch(
        "pastel.pastel_gemini.run_prompt_async",
        new=AsyncMock(return_value="0. Yes"),
    ):
        with pytest.raises(tenacity.RetryError):
            await pastel_instance._get_llm_answers_for_single_sentence(
                Sentence("a claim", tuple("quantity"))
            )


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
    pastel_instance: PastelGemini,
):
    with patch.object(
        pastel_instance, "_get_answers_for_single_sentence", side_effect=return_values
    ):
        answers = await pastel_instance.get_answers_to_questions(sentences)
        assert answers == expected


def test_labels_default_to_empty() -> None:
    assert PastelGemini({BiasType.BIAS: 1.0, Q1: 1.0}).labels == {}


@patch("pastel.pastel_gemini.run_prompt_async", new_callable=AsyncMock)
async def test_billing_labels_are_sent_with_every_call(mock_run_prompt) -> None:
    """Vertex billing labels let this model's Gemini spend be separated out in
    Google Cloud billing, so every call has to carry them."""
    mock_run_prompt.return_value = "0. yes"
    labels = {"team": "afc", "job": "checkworthy"}
    model = PastelGemini({BiasType.BIAS: 1.0, Q1: 1.0}, labels=labels)

    await model.get_answers_to_questions([Sentence("A claim.", ("quantity",))])

    assert mock_run_prompt.await_args.kwargs["labels"] == labels


def test_labels_survive_loading_and_copying(tmp_path) -> None:
    """Training copies models constantly (beam search, cross-validation), so a
    copy that dropped the labels would silently stop billing correctly."""
    import json

    model_file = tmp_path / "model.json"
    model_file.write_text(json.dumps({"bias": 1.0, Q1: 0.5}), encoding="utf-8")

    labels = {"team": "afc"}
    loaded = PastelGemini.load_model(str(model_file), labels=labels)
    assert loaded.labels == labels

    copied = loaded.create_copy_with_different_model({BiasType.BIAS: 2.0, Q2: 1.0})
    assert isinstance(copied, PastelGemini)
    assert copied.labels == labels

    assert PastelGemini.from_dict({"bias": 1.0}, labels=labels).labels == labels
    assert PastelGemini.from_feature_list([Q1], labels=labels).labels == labels
