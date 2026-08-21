"""Cache the answers to Pastel questions in a local database.
This is useful for local experiments etc. but shouldn't be used (or useful) in production
"""

import logging
from collections import defaultdict

from pastel.models import FEATURE_TYPE, BiasType, Sentence
from pastel.pastel import PastelModel
from training.db_manager import DatabaseManager

_logger = logging.getLogger(__name__)


class CachedPastel(PastelModel):
    """Wraps any other Pastel model and caches its answers in a database.

    Answers are cached per (question, sentence) pair, so adding one question to a
    model only costs backend calls for that new question - which is what makes
    feature selection over many overlapping question sets affordable.

    Functions are computed locally and cheaply, so they are never cached.
    """

    def __init__(self, inner: PastelModel, db: DatabaseManager | None = None) -> None:
        """Wrap `inner`, answering from `db` wherever possible and delegating
        the rest to `inner`."""
        self.inner = inner
        super().__init__(inner.model)
        self.db = db if db is not None else DatabaseManager()

    @property
    def model(self) -> dict[FEATURE_TYPE, float]:
        """A cache has no model of its own: it always reports the wrapped model,
        so the two can never drift apart."""
        return self.inner.model

    @model.setter
    def model(self, model: dict[FEATURE_TYPE, float]) -> None:
        self.inner.model = model

    @classmethod
    def from_pastel(
        cls, pastel: PastelModel, db: DatabaseManager | None = None
    ) -> "CachedPastel":
        """Wrap an existing Pastel model in a cache. Wrapping a model that is
        already cached returns it as-is rather than adding a second layer."""
        if isinstance(pastel, CachedPastel):
            return pastel if db is None else cls(pastel.inner, db)
        return cls(pastel, db)

    def create_copy_with_different_model(
        self, model: dict[FEATURE_TYPE, float]
    ) -> "PastelModel":
        """A new model with different features and weights, but the same
        backend and the same cache."""
        return CachedPastel(self.inner.create_copy_with_different_model(model), self.db)

    def get_cached_questions(self) -> list[str]:
        """
        Get a list of all unique questions that have responses in the cache.

        Returns:
            List of questions that have been asked and cached, sorted alphabetically
        """
        return self.db.get_unique_questions()

    def _backend_for(self, questions: frozenset[str]) -> PastelModel:
        """The wrapped model restricted to `questions`, so a cache miss only
        asks the backend about the questions that actually missed.
        The weights are irrelevant here - this is only used to fetch answers."""
        sub_model: dict[FEATURE_TYPE, float] = {BiasType.BIAS: self.get_bias()}
        for question in questions:
            sub_model[question] = self.model[question]
        return self.inner.create_copy_with_different_model(sub_model)

    async def get_answers_to_questions(
        self, sentences: list[Sentence]
    ) -> dict[Sentence, dict[FEATURE_TYPE, float]]:
        """Answer every feature in the model for each sentence, using the cache
        where it can and the wrapped model where it can't.
        For each sentence, this returns a dictionary mapping features to scores.
        Sentences the backend could not answer are omitted, as the base class requires.
        """
        questions = self.get_questions()

        # One database round trip for every (question, sentence) pair we need.
        cached = self.db.get_responses(
            questions, [sentence.sentence_text for sentence in sentences]
        )

        answers: dict[Sentence, dict[FEATURE_TYPE, float]] = {}
        # Sentences grouped by which questions they are missing, so that each
        # group needs one backend call for only the questions it actually needs.
        gaps: dict[frozenset[str], list[Sentence]] = defaultdict(list)

        for sentence in sentences:
            hits: dict[FEATURE_TYPE, float] = {}
            missing: set[str] = set()
            for question in questions:
                response = cached.get((question, sentence.sentence_text))
                if response is None:
                    missing.add(question)
                else:
                    hits[question] = response
            answers[sentence] = hits | self._get_function_answers_for_single_sentence(
                sentence
            )
            if missing:
                gaps[frozenset(missing)].append(sentence)

        for missing_questions, missing_sentences in gaps.items():
            _logger.info(
                "Cache miss: asking the model %d question(s) about %d sentence(s)",
                len(missing_questions),
                len(missing_sentences),
            )
            backend = self._backend_for(missing_questions)
            new_answers = await backend.get_answers_to_questions(missing_sentences)

            written = self.db.write_responses(
                (question, sentence.sentence_text, response)
                for sentence, sentence_answers in new_answers.items()
                for question, response in sentence_answers.items()
                # Only questions are cached; functions are recomputed each time.
                if isinstance(question, str)
            )
            _logger.info("Wrote %d new response(s) to the cache", written)

            for sentence, sentence_answers in new_answers.items():
                answers[sentence].update(sentence_answers)

        # The backend may not have answered every sentence we asked about (e.g.
        # Gemini timing out), so drop any sentence that is still incomplete.
        return {
            sentence: sentence_answers
            for sentence, sentence_answers in answers.items()
            if all(question in sentence_answers for question in questions)
        }
