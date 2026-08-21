# Uses local, pre-trained encoder models to answer questions about sentences.

import logging
from typing import Any

from local_models.model_registry import (
    MODEL_CATEGORY,
    MODELS,
    MODELS_DIR,
    model_id_for_question,
)
from local_models.questions import QUESTIONS

MAX_LENGTH = 128
BATCH_SIZE = 32

_logger = logging.getLogger(__name__)

# One (model, tokenizer) pair per model id, e.g. "q03". Loading a model takes
# seconds, so they are kept for the process lifetime once loaded.
_model_cache: dict[str, tuple[Any, Any]] = {}


def _load_model(model_id: str) -> tuple[Any, Any]:
    """Load the latest checkpoint of the fine-tuned model called `model_id`."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    checkpoint_dir = MODELS_DIR / MODEL_CATEGORY / model_id

    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    latest = checkpoints[-1]
    _logger.info("Loading model from %s", latest)

    base_model_id = MODELS[MODEL_CATEGORY]
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForSequenceClassification.from_pretrained(str(latest))
    model.eval()
    return model, tokenizer


def _cached_model(question: str) -> tuple[Any, Any]:
    """The (model, tokenizer) pair that answers `question`, loading it once."""
    model_id = model_id_for_question(question)
    if model_id not in _model_cache:
        _model_cache[model_id] = _load_model(model_id)
    return _model_cache[model_id]


def preload_models(questions: list[str] | None = None) -> None:
    """Load models into the cache up front, so the first call to
    answer_question() doesn't pay for it. Defaults to every question."""
    for question in QUESTIONS if questions is None else questions:
        _cached_model(question)


def answer_question(question: str, sentences: list[str]) -> list[float]:
    """
    Answers the question for the given list of sentences.
    Returns one score per sentence, in the same order as the input.
    """
    import torch

    model, tokenizer = _cached_model(question)

    input_text = [question + " " + sentence for sentence in sentences]

    answers: list[float] = []
    for start in range(0, len(input_text), BATCH_SIZE):
        batch = input_text[start : start + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = model(**inputs).logits

        answers.extend(torch.argmax(logits, dim=-1).float().tolist())

    return answers


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    preload_models()
    sentences = [
        "Scientists have shown that tamoxifen patients are more likely to develop deadly blood clots and cancer.",
        "Rubbing olive oil onto a lump under your skin will make it disappear in a few days.",
    ]
    answers = {question: answer_question(question, sentences) for question in QUESTIONS}

    for idx, sentence in enumerate(sentences):
        print(f"\n{'*' * 80}\n{sentence}\n")
        for question in QUESTIONS:
            print(f"{question[:60]:60s}  {answers[question][idx]}")
