import asyncio
from typing import Self, Sequence

from pastel.local.local_answerer import answer_question, preload_models
from pastel.local.model_registry import (
    missing_questions,
    require_available_questions,
)
from pastel.local.questions import QUESTIONS
from pastel.models import FEATURE_TYPE, Sentence
from pastel.pastel import PastelModel


class PastelLocal(PastelModel):
    """Answers the model's questions with the locally fine-tuned encoder models.

    There is one fine-tuned model per question in `pastel.local.questions.QUESTIONS`,
    so this backend can only answer questions drawn from that list. QUESTIONS is
    a declaration, though - whether a question's model has actually been trained
    is a separate matter, reported by
    `pastel.local.model_registry.available_questions()`. Constructing a model
    only checks the declaration, which is cheap; a declared-but-untrained
    question raises FileNotFoundError when it is first answered.
    """

    def __init__(self, model: dict[FEATURE_TYPE, float]) -> None:
        super().__init__(model)

        unsupported = [q for q in self.get_questions() if q not in QUESTIONS]
        if unsupported:
            raise ValueError(
                "PastelLocal has no fine-tuned model for the following question(s): "
                + "; ".join(unsupported)
                + ". Only questions listed in pastel.local.questions.QUESTIONS "
                "can be answered locally."
            )

    @classmethod
    def from_available_questions(
        cls, extra_features: Sequence[FEATURE_TYPE] = ()
    ) -> Self:
        """A new untrained model over every question that has a trained model
        on disk, plus any extra features given. Use this in preference to
        QUESTIONS when a partly-trained set of models is expected - during
        development, or before every question has been fine-tuned."""
        return cls.from_feature_list([*require_available_questions(), *extra_features])

    @staticmethod
    def untrained_questions() -> list[str]:
        """Declared questions whose models have not been trained (or cannot be
        found). Answering one of these raises FileNotFoundError."""
        return missing_questions()

    def preload(self) -> None:
        """Load this model's fine-tuned models into memory now, rather than on
        the first call to get_answers_to_questions(). Worth doing before timing
        anything, or before a long batch run - and it surfaces a missing model
        up front rather than part-way through a batch."""
        preload_models(self.get_questions())

    async def get_answers_to_questions(
        self, sentences: list[Sentence]
    ) -> dict[Sentence, dict[FEATURE_TYPE, float]]:
        """
        Get answers for a given list of sentences.
        For each sentence, this Returns a dictionary mapping features to scores.
        """
        if not sentences:
            return {}

        answers: dict[Sentence, dict[FEATURE_TYPE, float]] = {
            sentence: {} for sentence in sentences
        }

        # We run each question model against all sentences. Inference is
        # synchronous and CPU/GPU-bound, so keep it off the event loop.
        sentence_texts = [sentence.sentence_text for sentence in sentences]
        for question in self.get_questions():
            question_answers = await asyncio.to_thread(
                answer_question, question, sentence_texts
            )
            for sentence, answer in zip(sentences, question_answers):
                answers[sentence][question] = answer

        # Then get values from the functions
        for sentence in sentences:
            answers[sentence] |= self._get_function_answers_for_single_sentence(
                sentence
            )

        return answers
