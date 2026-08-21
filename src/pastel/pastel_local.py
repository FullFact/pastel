import asyncio

from local_models.local_answerer import answer_question, preload_models
from local_models.questions import QUESTIONS
from pastel.models import FEATURE_TYPE, Sentence
from pastel.pastel import PastelModel


class PastelLocal(PastelModel):
    """Answers the model's questions with the locally fine-tuned encoder models.

    There is one fine-tuned model per question in `local_models.questions.QUESTIONS`,
    so this backend can only answer questions drawn from that list.
    """

    def __init__(self, model: dict[FEATURE_TYPE, float]) -> None:
        super().__init__(model)

        unsupported = [q for q in self.get_questions() if q not in QUESTIONS]
        if unsupported:
            raise ValueError(
                "PastelLocal has no fine-tuned model for the following question(s): "
                + "; ".join(unsupported)
                + ". Only questions listed in local_models.questions.QUESTIONS "
                "can be answered locally."
            )

    def preload(self) -> None:
        """Load this model's fine-tuned models into memory now, rather than on
        the first call to get_answers_to_questions(). Worth doing before timing
        anything, or before a long batch run."""
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
