from local_models.local_answerer import answer_question
from pastel.models import Sentence
from pastel.pastel import FEATURE_TYPE, PastelModel


class PastelLocal(PastelModel):
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

        # We run each question model against all sentences
        sentence_texts = [sentence.sentence_text for sentence in sentences]
        for question in self.get_questions():
            question_answers = answer_question(question, sentence_texts)
            for sentence, answer in zip(sentences, question_answers):
                answers[sentence][question] = answer

        # Then get values from the functions
        for function in self.get_functions():
            for sentence in sentences:
                answers[sentence][function] = function(sentence)

        return answers
