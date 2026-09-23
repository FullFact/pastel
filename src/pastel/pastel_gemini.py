import asyncio
import logging
from textwrap import dedent

import tenacity
from genai_utils.gemini import run_prompt_async
from google.api_core import exceptions as core_exceptions

from pastel.models import FEATURE_TYPE, Sentence
from pastel.pastel import PastelModel

_logger = logging.getLogger(__name__)

RETRYABLE_EXCEPTIONS = (
    core_exceptions.ResourceExhausted,
    core_exceptions.InternalServerError,
    core_exceptions.ServiceUnavailable,
    core_exceptions.DeadlineExceeded,
    ValueError,
)


def log_retry_attempt(retry_state: tenacity.RetryCallState) -> None:
    """Log the retry attempt number and the exception that occurred."""
    if (not retry_state.outcome) or (not retry_state.next_action):
        return

    _logger.info(
        f"Retrying request due to {retry_state.outcome.exception()}..."
        f"Attempt #{retry_state.attempt_number}, "
        f"waiting {retry_state.next_action.sleep:.2f} seconds."
    )


class PastelGemini(PastelModel):
    """Answers the model's questions by sending them all to Gemini in a single
    prompt per sentence."""

    def __init__(
        self,
        model: dict[FEATURE_TYPE, float],
        labels: dict[str, str] | None = None,
    ) -> None:
        """As PastelModel, plus optional Vertex billing labels.

        The labels are attached to every Gemini call this model makes, so its
        spend can be separated out in Google Cloud billing. They are merged
        with (and take precedence over) any GENAI_LABEL_* environment variables
        picked up by genai_utils.
        """
        super().__init__(model)
        self.labels = labels or {}

    def create_copy_with_different_model(
        self, model: dict[FEATURE_TYPE, float]
    ) -> "PastelGemini":
        """A new model with different features and weights, still billed
        against the same labels."""
        return type(self)(model, labels=self.labels)

    async def get_answers_to_questions(
        self, sentences: list[Sentence]
    ) -> dict[Sentence, dict[FEATURE_TYPE, float]]:
        """
        Get answers for a given list of sentences.
        For each sentence, this Returns a dictionary mapping features to scores.
        """
        jobs = [
            self._get_answers_for_single_sentence(sentence) for sentence in sentences
        ]
        answers = await asyncio.gather(*jobs, return_exceptions=True)

        # return the answers which didn't cause an exception
        return {
            s: a for s, a in zip(sentences, answers) if not isinstance(a, BaseException)
        }

    def _make_prompt(self, sentence: Sentence) -> str:
        """Makes a prompt for a single given sentence."""

        questions = self.get_questions()

        prompt = dedent("""
            Your task is to answer a series of questions about a sentence. Ensure your answers are truthful and reliable.
            You are expected to answer with ‘Yes’ or ‘No’ but you are also allowed to answer with ‘Unsure’ if you do not
            have enough information or context to provide a reliable answer.
            Your response should be limited to the question number and yes/no/unsure.
            Example output:
            0. Yes
            1. Yes
            2. No

            Here are the questions:
            [QUESTIONS]

            Here is the sentence: ```[SENT1]```
            """)
        # extract the PastelFeatures whose type is string
        prompt = prompt.replace(
            "[QUESTIONS]",
            "\n".join([f"Question {idx} {q}" for idx, q in enumerate(questions)]),
        )
        prompt = prompt.replace("[SENT1]", sentence.sentence_text)

        return prompt

    @staticmethod
    def _label_mapping(label: str) -> float:
        """Map yes/no/other response to 1/0/0.5 respectively.
        If model responds 'unsure', 'don't know', 'uncertain' etc. then return 0.5.
        """
        label_map = {"y": 1.0, "n": 0.0}
        return label_map.get(label[0].lower(), 0.5)

    @tenacity.retry(
        wait=tenacity.wait_random_exponential(multiplier=1, max=60),
        stop=tenacity.stop_after_attempt(3),
        retry=tenacity.retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before=log_retry_attempt,
    )
    async def _get_llm_answers_for_single_sentence(
        self, sentence: Sentence
    ) -> dict[FEATURE_TYPE, float]:
        """Runs all genAI questions on the given sentence."""
        sent_answers: dict[FEATURE_TYPE, float] = {}
        prompt = self._make_prompt(sentence)

        raw_output = await run_prompt_async(prompt, labels=self.labels)
        raw_output = raw_output.strip().lower()

        if "question" in raw_output:
            output = raw_output[raw_output.index("0") :]
        else:
            output = raw_output
        answers = output.split("\n")  # e.g. ["1. yes", "2. no"]

        if len(answers) == len(self.get_questions()):
            for q, a in zip(self.get_questions(), answers):
                sent_answers[q] = self._label_mapping(a.split()[1])

        else:
            raise ValueError(
                f"Failed to parse output for the sentence: {sentence.sentence_text}. Output received: {output}"
            )
        return sent_answers

    async def _get_answers_for_single_sentence(
        self, sentence: Sentence
    ) -> dict[FEATURE_TYPE, float]:
        """Answer every feature in the model for one sentence: the questions go
        to Gemini, the functions are computed locally."""
        llm_sent_answers = await self._get_llm_answers_for_single_sentence(sentence)
        function_sent_answers = self._get_function_answers_for_single_sentence(sentence)

        return llm_sent_answers | function_sent_answers
