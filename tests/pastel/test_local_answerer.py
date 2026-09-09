"""Tests for batching in the local answerer, with the encoder stubbed out."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from pastel.local import local_answerer

# mypy: ignore-errors

# The stubs below recover a sentence's identity by counting its "x"s, so the
# question prefix must not contain any.
QUESTION = "Is it so?"


class FakeTokenizer:
    """Encodes each text as the number of "x"s in it, and records the batches
    it was asked to encode."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        self.batches.append(list(texts))
        return {"markers": torch.tensor([text.count("x") for text in texts])}


class FakeModel:
    """Answers each sentence with its own marker, so an answer that ends up
    against the wrong sentence is visible in the result."""

    def __call__(self, markers: Any) -> Any:
        classes = int(markers.max()) + 1
        return SimpleNamespace(logits=torch.nn.functional.one_hot(markers, classes))


@pytest.fixture
def fake_encoder(monkeypatch) -> FakeTokenizer:
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(
        local_answerer, "_cached_model", lambda question: (FakeModel(), tokenizer)
    )
    return tokenizer


def test_answers_come_back_in_the_callers_order(fake_encoder, monkeypatch) -> None:
    """Sentences are batched by length rather than in order, so each answer has
    to be put back where its sentence came from."""
    monkeypatch.setattr(local_answerer, "BATCH_SIZE", 4)
    lengths = [7, 1, 12, 3, 9, 2, 11, 5, 4, 10, 6, 8]
    sentences = ["x" * length for length in lengths]

    answers = local_answerer.answer_question(QUESTION, sentences)

    assert answers == [float(length) for length in lengths]


def test_batches_group_sentences_of_similar_length(fake_encoder, monkeypatch) -> None:
    monkeypatch.setattr(local_answerer, "BATCH_SIZE", 4)
    lengths = [7, 1, 12, 3, 9, 2, 11, 5, 4, 10, 6, 8]

    local_answerer.answer_question(QUESTION, ["x" * length for length in lengths])

    batched = [len(text) for batch in fake_encoder.batches for text in batch]
    assert batched == sorted(batched)
    assert [len(batch) for batch in fake_encoder.batches] == [4, 4, 4]


def test_no_sentences_needs_no_batches(fake_encoder) -> None:
    assert local_answerer.answer_question(QUESTION, []) == []
    assert fake_encoder.batches == []
