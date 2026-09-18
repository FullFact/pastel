"""Tests for the split handed to the multi-head trainer.

`without_sentences_in` is the one guard worth testing here: a sentence used to
train any head is one the shared body has seen, so leaving it in an evaluation
split makes the model look better than it is and shows no other symptom.
"""

from pastel.local.multi_head_encoder import IGNORE_LABEL
from pastel.local.training import Split

# mypy: ignore-errors

TRAIN = Split(sentences=["a", "b"], labels=[[1, 0], [0, 1]])


def test_answered_ignores_unlabelled_sentences() -> None:
    """A sentence labelled for only some questions still trains the heads it
    has answers for, so each head sees a different number of rows."""
    split = Split(sentences=["a", "b"], labels=[[1, IGNORE_LABEL], [0, 1]])

    assert split.answered(0) == [1, 0]
    assert split.answered(1) == [1]
    assert len(split) == 2


def test_sentences_seen_in_training_are_dropped() -> None:
    evaluation = Split(sentences=["b", "c"], labels=[[0, 1], [1, 1]])

    kept, dropped = evaluation.without_sentences_in(TRAIN)

    assert kept.sentences == ["c"]
    assert kept.labels == [[1, 1]]
    assert dropped == 1


def test_a_clean_split_is_left_alone() -> None:
    evaluation = Split(sentences=["c", "d"], labels=[[1, 1], [0, 0]])

    kept, dropped = evaluation.without_sentences_in(TRAIN)

    assert kept == evaluation
    assert dropped == 0
