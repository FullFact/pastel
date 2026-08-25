from pathlib import Path

import numpy as np
import pytest

from pastel.models import FEATURE_TYPE, BiasType, Sentence
from pastel.pastel import PastelModel
from training.cached_pastel import CachedPastel
from training.db_manager import DatabaseManager

Q1 = "Is the statement factual?"
Q2 = "Does the statement contain bias?"


class DummyPastel(PastelModel):
    """A backend that answers every question with 1.0 and counts its calls."""

    def __init__(self, model: dict[FEATURE_TYPE, float] | None = None) -> None:
        if model is None:
            model = {
                BiasType.BIAS: 1.0,
                Q1: -3.0,
                Q2: 2.0,
            }
        super().__init__(model)
        self.calls: list[tuple[frozenset[str], frozenset[str]]] = []

    async def get_answers_to_questions(
        self, sentences: list[Sentence]
    ) -> dict[Sentence, dict[FEATURE_TYPE, float]]:
        self.calls.append(
            (
                frozenset(self.get_questions()),
                frozenset(s.sentence_text for s in sentences),
            )
        )
        return {s: {q: 1.0 for q in self.get_questions()} for s in sentences}

    def create_copy_with_different_model(
        self, model: dict[FEATURE_TYPE, float]
    ) -> "PastelModel":
        """Share the call log with the sub-models the cache creates, so a test
        can see every question the backend was asked."""
        sub = type(self)(model)
        sub.calls = self.calls
        return sub


@pytest.fixture
def db(tmp_path: Path) -> DatabaseManager:
    return DatabaseManager(str(tmp_path / "test_responses.db"))


@pytest.fixture
def sentences() -> list[Sentence]:
    return [
        Sentence(text, tuple(["quantity"]))
        for text in ["sentence one", "sentence two", "sentence three"]
    ]


def test_cached_pastel_instantiation(db: DatabaseManager) -> None:
    dummy = DummyPastel()
    cached = CachedPastel.from_pastel(dummy, db)
    assert isinstance(cached, CachedPastel)
    assert cached.db is db
    assert np.allclose(
        np.array(list(cached.model.values())),
        np.array(list(dummy.model.values())),
    )
    assert list(cached.model.keys()) == list(dummy.model.keys())


def test_from_pastel_does_not_wrap_twice(db: DatabaseManager) -> None:
    dummy = DummyPastel()
    cached = CachedPastel.from_pastel(dummy, db)
    assert CachedPastel.from_pastel(cached) is cached
    # Re-wrapping with a different db swaps the cache rather than nesting one
    other = CachedPastel.from_pastel(cached, db)
    assert other.inner is dummy


def test_model_tracks_the_wrapped_model(db: DatabaseManager) -> None:
    dummy = DummyPastel()
    cached = CachedPastel.from_pastel(dummy, db)
    dummy.model = {BiasType.BIAS: 5.0, Q1: 1.0}
    assert cached.model == dummy.model
    assert cached.get_bias() == 5.0


def test_with_model_keeps_backend_and_cache(db: DatabaseManager) -> None:
    cached = CachedPastel.from_pastel(DummyPastel(), db)
    updated = cached.create_copy_with_different_model({BiasType.BIAS: 0.5, Q1: 1.0})
    assert isinstance(updated, CachedPastel)
    assert updated.db is db
    assert isinstance(updated.inner, DummyPastel)
    assert updated.get_questions() == [Q1]


async def test_answers_are_cached_across_calls(
    db: DatabaseManager, sentences: list[Sentence]
) -> None:
    dummy = DummyPastel()
    cached = CachedPastel.from_pastel(dummy, db)

    first = await cached.get_answers_to_questions(sentences)
    assert set(first.keys()) == set(sentences)
    assert all(a == {Q1: 1.0, Q2: 1.0} for a in first.values())
    assert len(dummy.calls) == 1, "the backend should be asked once on a cold cache"

    for _ in range(2):
        again = await cached.get_answers_to_questions(sentences)
        assert again == first
        assert len(dummy.calls) == 1, "later calls should be served from the cache"


async def test_only_missing_questions_are_asked(
    db: DatabaseManager, sentences: list[Sentence]
) -> None:
    """The point of caching per (question, sentence): adding a question to a
    model should only cost backend calls for the new question."""
    dummy = DummyPastel({BiasType.BIAS: 1.0, Q1: -3.0})
    await CachedPastel.from_pastel(dummy, db).get_answers_to_questions(sentences)
    assert dummy.calls == [
        (frozenset([Q1]), frozenset(s.sentence_text for s in sentences))
    ]

    bigger = DummyPastel({BiasType.BIAS: 1.0, Q1: -3.0, Q2: 2.0})
    answers = await CachedPastel.from_pastel(bigger, db).get_answers_to_questions(
        sentences
    )
    assert bigger.calls == [
        (frozenset([Q2]), frozenset(s.sentence_text for s in sentences))
    ], "Q1 was already cached, so only Q2 should be asked"
    assert all(a == {Q1: 1.0, Q2: 1.0} for a in answers.values())


async def test_sentences_are_grouped_by_their_missing_questions(
    db: DatabaseManager, sentences: list[Sentence]
) -> None:
    """Sentences missing the same questions share a single backend call."""
    db.write_responses([(Q1, sentences[0].sentence_text, 0.0)])
    dummy = DummyPastel()
    answers = await CachedPastel.from_pastel(dummy, db).get_answers_to_questions(
        sentences
    )

    assert sorted(dummy.calls, key=lambda c: len(c[1])) == [
        # sentence one only needs Q2; the other two need both questions
        (frozenset([Q2]), frozenset(["sentence one"])),
        (frozenset([Q1, Q2]), frozenset(["sentence two", "sentence three"])),
    ]
    # the cached 0.0 for sentence one is kept, not overwritten
    assert answers[sentences[0]] == {Q1: 0.0, Q2: 1.0}


async def test_unanswered_sentences_are_omitted(
    db: DatabaseManager, sentences: list[Sentence]
) -> None:
    class HalfAnsweringPastel(DummyPastel):
        async def get_answers_to_questions(
            self, sentences: list[Sentence]
        ) -> dict[Sentence, dict[FEATURE_TYPE, float]]:
            answers = await super().get_answers_to_questions(sentences)
            # Simulate the backend failing on one sentence
            return {
                s: a for s, a in answers.items() if s.sentence_text != "sentence two"
            }

    cached = CachedPastel.from_pastel(HalfAnsweringPastel(), db)
    answers = await cached.get_answers_to_questions(sentences)

    assert set(s.sentence_text for s in answers) == {"sentence one", "sentence three"}
    # ...and the failure isn't cached, so a later call retries it
    assert db.get_response(Q1, "sentence two") is None


async def test_functions_are_answered_but_not_cached(db: DatabaseManager) -> None:
    from pastel import pastel_functions

    model: dict[FEATURE_TYPE, float] = {
        BiasType.BIAS: 1.0,
        Q1: -3.0,
        pastel_functions.is_claim_type_quantity: 2.0,
    }
    sentence = Sentence("a sentence", tuple(["quantity"]))
    cached = CachedPastel.from_pastel(DummyPastel(model), db)

    answers = await cached.get_answers_to_questions([sentence])
    assert answers[sentence] == {Q1: 1.0, pastel_functions.is_claim_type_quantity: 1.0}
    assert db.get_unique_questions() == [Q1], "only questions belong in the cache"


async def test_no_questions_needs_no_backend_call(db: DatabaseManager) -> None:
    """A model of functions alone (as beam search starts with) shouldn't hit the backend."""
    dummy = DummyPastel({BiasType.BIAS: 1.0})
    cached = CachedPastel.from_pastel(dummy, db)
    sentence = Sentence("a sentence", tuple(["quantity"]))

    assert await cached.get_answers_to_questions([sentence]) == {sentence: {}}
    assert dummy.calls == []


async def test_empty_sentence_list(db: DatabaseManager) -> None:
    cached = CachedPastel.from_pastel(DummyPastel(), db)
    assert await cached.get_answers_to_questions([]) == {}


def test_get_cached_questions(db: DatabaseManager, sentences: list[Sentence]) -> None:
    cached = CachedPastel.from_pastel(DummyPastel(), db)
    assert cached.get_cached_questions() == []
    db.write_responses(
        [(Q2, "sentence one", 1.0), (Q1, "sentence one", 0.5)],
    )
    assert cached.get_cached_questions() == sorted([Q1, Q2])
