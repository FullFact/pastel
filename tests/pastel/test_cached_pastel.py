import numpy as np

from pastel.models import FEATURE_TYPE, BiasType, Sentence
from pastel.pastel import Pastel
from training.cached_pastel import CachedPastel

Q1 = "Is the statement factual?"
Q2 = "Does the statement contain bias?"


class DummyPastel(Pastel):
    def __init__(self, questions=None):
        if questions is None:
            questions = {
                BiasType.BIAS: 1.0,
                Q1: -3.0,
                Q2: 2.0,
            }
        super().__init__(questions)

    async def get_answers_to_questions(
        self, sentences: list[Sentence]
    ) -> dict[Sentence, dict[FEATURE_TYPE, float]]:
        return {s: {Q1: 1.0} for s in sentences}

    # def make_predictions(self, sentences):
    #     answers = self.get_answers_to_questions(sentences)
    #     return self.get_scores_from_answers(answers)


def test_cached_pastel_instantiation():
    dummy = DummyPastel()
    cached = CachedPastel.from_pastel(dummy)
    assert isinstance(cached, CachedPastel)
    assert hasattr(cached, "db")
    assert np.allclose(
        np.array(list(cached.model.values())),
        np.array(list(dummy.model.values())),
    )
    assert list(cached.model.keys()) == list(dummy.model.keys())


def test_from_pastel_copies_model():
    dummy = DummyPastel(
        {
            BiasType.BIAS: 1.0,
            Q1: 2.0,
            Q2: 3.0,
        }
    )
    cached = CachedPastel.from_pastel(dummy)
    assert np.allclose(
        np.array(list(cached.model.values())),
        np.array(list(dummy.model.values())),
    )
    assert list(cached.model.keys()) == list(dummy.model.keys())


async def test_make_predictions_caching(monkeypatch):
    dummy = DummyPastel(
        {
            BiasType.BIAS: 1.0,
            Q1: 2.0,
            Q2: 3.0,
        }
    )
    cached = CachedPastel.from_pastel(dummy)
    sentences = [
        Sentence("sentence one", tuple(["quantity"])),
        Sentence("sentence two", tuple(["quantity"])),
        Sentence("sentence three", tuple(["quantity"])),
        Sentence("sentence four", tuple(["quantity"])),
        Sentence("sentence five", tuple(["quantity"])),
    ]
    # Simulate a cache using a dict instead of a database
    cache = {}

    def fake_get_response(question, sentence):
        return cache.get((question, sentence), None)

    def fake_write_response(question, sentence, response):
        cache[(question, sentence)] = response

    cached.db.get_response = fake_get_response
    cached.db.write_response = fake_write_response
    # Patch Pastel.get_answers_to_questions to return a fixed array and count calls
    call_count = {"count": 0}

    async def counted_get_answers(
        self, sentences: list[Sentence]
    ) -> dict[Sentence, dict[FEATURE_TYPE, float]]:
        call_count["count"] += 1
        print("in test / counted_get_answers")
        _ = [print({sentence: {q: 1.0 for q in self.model}}) for sentence in sentences]
        return {sentence: {q: 1.0 for q in self.model} for sentence in sentences}

    monkeypatch.setattr(Pastel, "get_answers_to_questions", counted_get_answers)
    # First call should call get_answers_to_questions and populate the cache
    assert call_count["count"] == 0, "get_answers_to_questions hasn't been called yet"
    results1 = await cached.get_answers_to_questions(sentences)
    for sentence in sentences:
        for answer in results1[sentence].values():
            assert answer == 1.0
    assert (
        call_count["count"] == 1
    ), "get_answers_to_questions should be called once when a new response is added to the cache"

    # Second call should use the cache (no new calls to get_answers_to_questions)
    results2 = await cached.get_answers_to_questions(sentences)
    for sentence in sentences:
        for answer in results2[sentence].values():
            assert answer == 1.0
    # Assert that get_answers_to_questions was only called once (on the first call)
    assert (
        call_count["count"] == 1
    ), "get_answers_to_questions should only be called once, using cache on second call"

    # Third call should again use the cache (no new calls to get_answers_to_questions)
    results3 = await cached.get_answers_to_questions(sentences)
    for sentence in sentences:
        for answer in results3[sentence].values():
            assert answer == 1.0
    # Assert that get_answers_to_questions was only called once (on the first call)
    assert (
        call_count["count"] == 1
    ), "get_answers_to_questions should only be called once, using cache on third call"
