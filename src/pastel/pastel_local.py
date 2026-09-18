import asyncio
from typing import Self, Sequence

from pastel.local.local_answerer import answer_questions, preload_models
from pastel.local.model_registry import has_model, require_available_questions
from pastel.models import FEATURE_TYPE, Sentence
from pastel.pastel import PastelModel


class PastelLocal(PastelModel):
    """Answers the model's questions with the locally fine-tuned encoder.

    One model answers every question, with a head per question, so this
    backend can only answer questions that have been trained and recorded in
    the model map - which is what `pastel.local.available_questions()` reports.
    The questions themselves belong to the downstream task, not the library.
    """

    def __init__(self, model: dict[FEATURE_TYPE, float]) -> None:
        super().__init__(model)

        unsupported = [q for q in self.get_questions() if not has_model(q)]
        if unsupported:
            raise ValueError(
                "PastelLocal has no fine-tuned model for the following question(s): "
                + "; ".join(unsupported)
                + ". Train one with pastel.local.training, or run "
                "`python -m pastel.local` to see what is available and where."
            )

    @classmethod
    def from_available_questions(
        cls, extra_features: Sequence[FEATURE_TYPE] = ()
    ) -> Self:
        """A new untrained model over every question that has a trained head on
        disk, plus any extra features given."""
        return cls.from_feature_list([*require_available_questions(), *extra_features])

    def preload(self) -> None:
        """Load the fine-tuned model into memory now, rather than on the first
        call to get_answers_to_questions(). Worth doing before a long batch run
        or before timing anything, and it surfaces a missing model up front."""
        preload_models(self.get_questions())

    async def get_answers_to_questions(
        self, sentences: list[Sentence]
    ) -> dict[Sentence, dict[FEATURE_TYPE, float]]:
        """Answers for a given list of sentences, as a dict of features to
        scores per sentence."""
        if not sentences:
            return {}

        answers: dict[Sentence, dict[FEATURE_TYPE, float]] = {
            sentence: {} for sentence in sentences
        }

        # One pass of the shared encoder answers every question, so they all go
        # together. Inference is synchronous and CPU-bound, so keep it off the
        # event loop.
        questions = self.get_questions()
        if questions:
            question_answers = await asyncio.to_thread(
                answer_questions,
                questions,
                [sentence.sentence_text for sentence in sentences],
            )
            for question, scores in question_answers.items():
                for sentence, answer in zip(sentences, scores):
                    answers[sentence][question] = answer

        for sentence in sentences:
            answers[sentence] |= self._get_function_answers_for_single_sentence(
                sentence
            )

        return answers
