import asyncio
from typing import Self, Sequence

from pastel.local.local_answerer import answer_question, preload_models
from pastel.local.model_registry import has_model, require_available_questions
from pastel.models import FEATURE_TYPE, Sentence
from pastel.pastel import PastelModel


class PastelLocal(PastelModel):
    """Answers the model's questions with the locally fine-tuned encoder models.

    There is one fine-tuned model per question, so this backend can only answer
    questions that have been trained and recorded in the model map - which is
    what `pastel.local.model_registry.available_questions()` reports. The
    questions themselves belong to the downstream task, not to this library.
    """

    def __init__(self, model: dict[FEATURE_TYPE, float]) -> None:
        super().__init__(model)

        unsupported = [q for q in self.get_questions() if not has_model(q)]
        if unsupported:
            raise ValueError(
                "PastelLocal has no fine-tuned model for the following question(s): "
                + "; ".join(unsupported)
                + ". Train one with local_models.finetune_encoder, or run "
                "`python -m pastel.local` to see what is available and where."
            )

    @classmethod
    def from_available_questions(
        cls, extra_features: Sequence[FEATURE_TYPE] = ()
    ) -> Self:
        """A new untrained model over every question that has a trained model
        on disk, plus any extra features given."""
        return cls.from_feature_list([*require_available_questions(), *extra_features])

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
