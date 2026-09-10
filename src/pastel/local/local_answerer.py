# Uses a local, fine-tuned encoder model to answer questions about sentences.

import logging
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

# Answering questions locally needs transformers and torch, which are an
# optional extra so that Gemini-only users don't have to install them.
MISSING_DEPENDENCIES_HINT = (
    "Answering Pastel questions locally needs the optional inference "
    "dependencies (transformers and torch). Install them with "
    "`uv sync --extra local`, or `pip install 'pastel[local]'`."
)

_logger = logging.getLogger(__name__)

# One model answers every question, so there is a single (model, tokenizer)
# pair. Loading it takes seconds, so it is kept for the process lifetime.
_loaded: tuple[Any, Any] | None = None


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(MISSING_DEPENDENCIES_HINT) from exc
    return torch


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
    return model, tokenizer


def _cached_model() -> tuple[Any, Any]:
    """The (model, tokenizer) pair that answers every question, loaded once."""
    global _loaded
    if _loaded is None:
        _loaded = _load_model()
    return _loaded


def _heads_for(questions: list[str], model: Any) -> list[int]:
    """The head that answers each question, checking the trained model really
    has it. A map listing questions the model was not trained for would
    otherwise be answered by whatever head happens to sit at that index."""
    heads = [head_for_question(question) for question in questions]
    untrained = [
        question for question, head in zip(questions, heads) if head >= model.n_heads
    ]
    if untrained:
        raise ValueError(
            f"The trained model has {model.n_heads} head(s), too few to answer: "
            + "; ".join(untrained)
            + ". One model answers every question, so they have to be "
            "retrained together with local_models.finetune_encoder."
        )
    return heads


def preload_models(questions: list[str] | None = None) -> None:
    """Load the model into the cache up front, so the first call to
    answer_questions() doesn't pay for it. Also checks the model has a head for
    each question, rather than failing part-way through a batch. Defaults to
    every available question."""
    model, _ = _cached_model()
    _heads_for(available_questions() if questions is None else questions, model)


def answer_questions(
    questions: list[str], sentences: list[str]
) -> dict[str, list[float]]:
    """
    Answers every question for every sentence.
    Returns one score per sentence for each question, in the same order as the
    input sentences.

    One pass of the shared encoder answers every question at once, so asking
    all of them together costs little more than asking one.
    """
    torch = _import_torch()

    model, tokenizer = _cached_model()
    heads = _heads_for(questions, model)

    answers = {question: [0.0] * len(sentences) for question in questions}

    # The sentence is the whole input. Each head answers one fixed question, so
    # prefixing the question text would spend a large part of every forward
    # pass on a constant carrying no information.
    #
    # Every batch is padded to its longest member, so batching sentences in
    # file order makes short sentences pay for long ones. Grouping sentences of
    # similar length together cuts that waste; the answers are written back
    # into the caller's order.
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
    """
    Answers the question for the given list of sentences.
    Returns one score per sentence, in the same order as the input.

    Answering several questions costs barely more than answering one, so prefer
    answer_questions() when you want more than this.
    """
    return answer_questions([question], sentences)[question]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    questions = available_questions()
    preload_models(questions)
    sentences = [
        "Scientists have shown that tamoxifen patients are more likely to develop deadly blood clots and cancer.",
        "Rubbing olive oil onto a lump under your skin will make it disappear in a few days.",
    ]
    answers = answer_questions(questions, sentences)

    for idx, sentence in enumerate(sentences):
        print(f"\n{'*' * 80}\n{sentence}\n")
        for question in questions:
            print(f"{question[:60]:60s}  {answers[question][idx]}")
