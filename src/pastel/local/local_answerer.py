# Uses local, pre-trained encoder models to answer questions about sentences.

import logging
from typing import Any

from pastel.local.model_registry import (
    MODEL_CATEGORY,
    MODELS,
    available_questions,
    latest_checkpoint,
    model_id_for_question,
)

MAX_LENGTH = 128
BATCH_SIZE = 32

# Answering questions locally needs transformers and torch, which are an
# optional extra so that Gemini-only users don't have to install them.
MISSING_DEPENDENCIES_HINT = (
    "Answering Pastel questions locally needs the optional inference "
    "dependencies (transformers and torch). Install them with "
    "`uv sync --extra local`, or `pip install 'pastel[local]'`."
)

_logger = logging.getLogger(__name__)

# One (model, tokenizer) pair per model id, e.g. "q03". Loading a model takes
# seconds, so they are kept for the process lifetime once loaded.
_model_cache: dict[str, tuple[Any, Any]] = {}


def _import_transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise ImportError(MISSING_DEPENDENCIES_HINT) from exc
    return transformers


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(MISSING_DEPENDENCIES_HINT) from exc
    return torch


def _load_model(question: str) -> tuple[Any, Any]:
    """Load the latest checkpoint of the model fine-tuned for `question`."""
    transformers = _import_transformers()

    checkpoint = latest_checkpoint(question)
    _logger.info("Loading model from %s", checkpoint)

    base_model_id = MODELS[MODEL_CATEGORY]
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_id)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint)
    )
    model.eval()
    return model, tokenizer


def _cached_model(question: str) -> tuple[Any, Any]:
    """The (model, tokenizer) pair that answers `question`, loading it once."""
    model_id = model_id_for_question(question)
    if model_id not in _model_cache:
        _model_cache[model_id] = _load_model(question)
    return _model_cache[model_id]


def preload_models(questions: list[str] | None = None) -> None:
    """Load models into the cache up front, so the first call to
    answer_question() doesn't pay for it. Defaults to every available question."""
    for question in available_questions() if questions is None else questions:
        _cached_model(question)


def answer_question(question: str, sentences: list[str]) -> list[float]:
    """
    Answers the question for the given list of sentences.
    Returns one score per sentence, in the same order as the input.
    """
    torch = _import_torch()

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

    questions = available_questions()
    preload_models(questions)
    sentences = [
        "Scientists have shown that tamoxifen patients are more likely to develop deadly blood clots and cancer.",
        "Rubbing olive oil onto a lump under your skin will make it disappear in a few days.",
    ]
    answers = {question: answer_question(question, sentences) for question in questions}

    for idx, sentence in enumerate(sentences):
        print(f"\n{'*' * 80}\n{sentence}\n")
        for question in questions:
            print(f"{question[:60]:60s}  {answers[question][idx]}")
