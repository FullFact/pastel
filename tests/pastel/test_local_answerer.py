"""Tests for batching and head selection in the local answerer, with the
shared encoder stubbed out."""

from typing import Any

import pytest
import torch

from pastel.local import local_answerer
from pastel.local.multi_head_encoder import N_CLASSES

# mypy: ignore-errors

# One head per question, in the order the model map would record them.
QUESTIONS = ["Is it so?", "Is it not?", "Is it though?"]


class FakeTokenizer:
    """Encodes each sentence as the number of "x"s in it, and records the
    batches it was asked to encode."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        self.batches.append(list(texts))
        markers = torch.tensor([text.count("x") for text in texts])
        return {"input_ids": markers, "attention_mask": torch.ones_like(markers)}


class FakeModel:
    """Head `h` answers yes for a sentence exactly when bit `h` of that
    sentence's marker is set, so both the sentence an answer belongs to and the
    head it came from are visible in the result."""

    n_heads = len(QUESTIONS)

    def head_logits(self, input_ids: Any, attention_mask: Any) -> torch.Tensor:
        bits = torch.stack(
            [(input_ids >> head) & 1 for head in range(self.n_heads)], dim=1
        )
        return torch.nn.functional.one_hot(bits, N_CLASSES).float()


def expected(marker: int, head: int) -> float:
    return float((marker >> head) & 1)


@pytest.fixture
def fake_encoder(monkeypatch) -> FakeTokenizer:
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(
        local_answerer, "_cached_model", lambda: (FakeModel(), tokenizer)
    )
    monkeypatch.setattr(local_answerer, "head_for_question", QUESTIONS.index)
    return tokenizer


def test_answers_come_back_in_the_callers_order(fake_encoder, monkeypatch) -> None:
    """Sentences are batched by length rather than in order, so each answer has
    to be put back where its sentence came from."""
    monkeypatch.setattr(local_answerer, "BATCH_SIZE", 2)
    markers = [5, 1, 7, 3, 6, 2, 4]
    sentences = ["x" * marker for marker in markers]

    answers = local_answerer.answer_questions(QUESTIONS, sentences)

    for head, question in enumerate(QUESTIONS):
        assert answers[question] == [expected(marker, head) for marker in markers]


def test_each_question_is_answered_by_its_own_head(fake_encoder) -> None:
    """The questions are answered in one pass, so each one has to take the
    answer from its own head - not from its position in the request."""
    asked = [QUESTIONS[2], QUESTIONS[0]]
    markers = [3, 4]

    answers = local_answerer.answer_questions(asked, ["x" * m for m in markers])

    assert set(answers) == set(asked)
    assert answers[QUESTIONS[2]] == [expected(3, 2), expected(4, 2)]
    assert answers[QUESTIONS[0]] == [expected(3, 0), expected(4, 0)]


def test_one_question_uses_the_same_pass(fake_encoder, monkeypatch) -> None:
    monkeypatch.setattr(local_answerer, "BATCH_SIZE", 8)
    markers = [1, 2, 3]

    answers = local_answerer.answer_question(QUESTIONS[1], ["x" * m for m in markers])

    assert answers == [expected(marker, 1) for marker in markers]
    assert len(fake_encoder.batches) == 1


def test_batches_group_sentences_of_similar_length(fake_encoder, monkeypatch) -> None:
    monkeypatch.setattr(local_answerer, "BATCH_SIZE", 4)
    markers = [7, 1, 12, 3, 9, 2, 11, 5, 4, 10, 6, 8]

    local_answerer.answer_questions(QUESTIONS, ["x" * m for m in markers])

    batched = [len(text) for batch in fake_encoder.batches for text in batch]
    assert batched == sorted(batched)
    assert [len(batch) for batch in fake_encoder.batches] == [4, 4, 4]


def test_the_sentence_is_the_whole_input(fake_encoder) -> None:
    """Each head answers one fixed question, so the question text is not part
    of the input - the model would only be paying to encode a constant."""
    sentences = ["A sentence.", "Another one."]

    local_answerer.answer_questions(QUESTIONS, sentences)

    assert fake_encoder.batches == [sentences]


def test_no_sentences_needs_no_batches(fake_encoder) -> None:
    assert local_answerer.answer_questions(QUESTIONS, []) == {
        question: [] for question in QUESTIONS
    }
    assert fake_encoder.batches == []


def test_a_head_the_model_does_not_have_is_rejected(fake_encoder, monkeypatch) -> None:
    """A model map recording more questions than the model was trained for
    would otherwise be answered by whatever head sits at that index."""
    monkeypatch.setattr(local_answerer, "head_for_question", lambda question: 9)

    with pytest.raises(ValueError, match="too few to answer"):
        local_answerer.answer_questions([QUESTIONS[0]], ["A sentence."])
