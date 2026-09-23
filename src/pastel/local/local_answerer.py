"""Answer questions about sentences with the locally fine-tuned encoder."""

import logging
import os
from typing import Any

from pastel.local.model_registry import (
    MODEL_CATEGORY,
    MODELS,
    available_questions,
    head_for_question,
    latest_checkpoint,
)

MAX_LENGTH = 128
BATCH_SIZE = 32

# Quantise the model to int8 as it loads: worth roughly 20% of inference time
# on a CPU, at a fraction of the memory. It changes the numerics, so it is off
# unless asked for, and the holdout evaluation should be re-run before a model
# is trusted with it on.
QUANTISE_ENV_VAR = "PASTEL_LOCAL_QUANTISE"
QUANTISE_ON = ("1", "true", "yes", "on")

MISSING_DEPENDENCIES_HINT = (
    "Answering Pastel questions locally needs the optional inference "
    "dependencies (transformers and torch). Install them with "
    "`uv sync --extra local`, or `pip install 'pastel[local]'`."
)

_logger = logging.getLogger(__name__)

# One model answers every question, and loading it takes seconds, so the
# (model, tokenizer) pair is kept for the process lifetime.
_loaded: tuple[Any, Any] | None = None


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(MISSING_DEPENDENCIES_HINT) from exc
    return torch


def _quantise(model: Any) -> Any:
    """The model with its linear layers quantised to int8, if
    PASTEL_LOCAL_QUANTISE asks for it."""
    if os.environ.get(QUANTISE_ENV_VAR, "").lower() not in QUANTISE_ON:
        return model

    try:
        from torchao.quantization import (
            Int8DynamicActivationInt8WeightConfig,
            quantize_,
        )
    except ImportError as exc:
        raise ImportError(
            f"{QUANTISE_ENV_VAR} needs torchao, one of the optional inference "
            "dependencies. Install them with `uv sync --extra local`, or unset "
            f"{QUANTISE_ENV_VAR}."
        ) from exc

    _logger.info("Quantising the model's linear layers to int8")
    quantize_(model, Int8DynamicActivationInt8WeightConfig())  # in place
    return model


def _load_model() -> tuple[Any, Any]:
    """Load the latest checkpoint of the fine-tuned model."""
    try:
        import transformers

        from pastel.local.multi_head_encoder import MultiHeadEncoder
    except ImportError as exc:
        raise ImportError(MISSING_DEPENDENCIES_HINT) from exc

    checkpoint = latest_checkpoint()
    _logger.info("Loading model from %s", checkpoint)

    base_model_id = MODELS[MODEL_CATEGORY]
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_id)
    model = MultiHeadEncoder.from_checkpoint(checkpoint, base_model_id)
    return _quantise(model), tokenizer


def _cached_model() -> tuple[Any, Any]:
    """The (model, tokenizer) pair that answers every question, loaded once."""
    global _loaded
    if _loaded is None:
        _loaded = _load_model()
    return _loaded


def _heads_for(questions: list[str], model: Any) -> list[int]:
    """The head that answers each question, checking the trained model really
    has it. A map listing questions the model was not trained for would
    otherwise be answered by whatever head sits at that index."""
    heads = [head_for_question(question) for question in questions]
    untrained = [
        question for question, head in zip(questions, heads) if head >= model.n_heads
    ]
    if untrained:
        raise ValueError(
            f"The trained model has {model.n_heads} head(s), too few to answer: "
            + "; ".join(untrained)
            + ". One model answers every question, so they have to be "
            "retrained together."
        )
    return heads


def preload_models(questions: list[str] | None = None) -> None:
    """Load the model now, so the first call to answer_questions() doesn't pay
    for it, and check it has a head for each question rather than failing
    part-way through a batch. Defaults to every available question."""
    model, _ = _cached_model()
    _heads_for(available_questions() if questions is None else questions, model)


def answer_questions(
    questions: list[str], sentences: list[str]
) -> dict[str, list[float]]:
    """Answer every question for every sentence, one score per sentence per
    question in the order the sentences were given.

    One pass of the shared encoder answers every question, so asking all of
    them costs little more than asking one.
    """
    torch = _import_torch()

    model, tokenizer = _cached_model()
    heads = _heads_for(questions, model)

    answers = {question: [0.0] * len(sentences) for question in questions}

    # The sentence is the whole input: each head answers one fixed question, so
    # prefixing the question text would spend a large part of every forward
    # pass on a constant.
    #
    # Every batch is padded to its longest member, so batching in file order
    # makes short sentences pay for long ones. Grouping by length cuts that
    # waste; the answers are written back into the caller's order.
    by_length = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))

    for start in range(0, len(by_length), BATCH_SIZE):
        indices = by_length[start : start + BATCH_SIZE]
        inputs = tokenizer(
            [sentences[i] for i in indices],
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        with torch.no_grad():
            # (sentences, heads, classes)
            logits = model.head_logits(inputs["input_ids"], inputs["attention_mask"])

        predictions = torch.argmax(logits, dim=-1).float().tolist()
        for row, index in enumerate(indices):
            for question, head in zip(questions, heads):
                answers[question][index] = predictions[row][head]

    return answers


def answer_question(question: str, sentences: list[str]) -> list[float]:
    """Answer one question for a list of sentences. Answering several costs
    barely more, so prefer answer_questions() for more than one."""
    return answer_questions([question], sentences)[question]
