"""Which fine-tuned model answers which question, and where those models live.

Each question gets its own fine-tuned model, saved in a directory named after
a short id (`q00`, `q01`, ...). Training allocates those ids and records them
in `model_map.json`; inference has to look up the same mapping, or it would
answer a question with another question's model.

Before this map existed, ids were implicitly the question's index in
`QUESTIONS`, so that is still the fallback when a question is not in the map.
Note that the fallback is only correct while `QUESTIONS` keeps the order the
models were trained in - which is exactly why the map is written.

`QUESTIONS` is a hand-maintained declaration of what the local backend is
meant to answer. What it can *actually* answer is whatever has a trained model
on disk, which is what `available_questions()` reports.
"""

import json
import os
from pathlib import Path

from pastel.local.questions import QUESTIONS

MODELS: dict[str, str] = {
    "ModernBERT-multilingual": "jhu-clsp/mmBERT-base",
    "mDeBERTa-v3-base": "microsoft/mdeberta-v3-base",
    "XLM-RoBERTa-base": "FacebookAI/xlm-roberta-base",
}

MODEL_CATEGORY = "ModernBERT-multilingual"  # only using this one for now
MODEL_MAP_FILENAME = "model_map.json"

# Where the fine-tuned models are kept. The default is relative, so it only
# resolves when the working directory is the repo root - set the environment
# variable to an absolute path anywhere else, production included.
MODELS_DIR_ENV_VAR = "PASTEL_LOCAL_MODELS_DIR"
DEFAULT_MODELS_DIR = Path("data/local_models/models")


def models_dir() -> Path:
    """The directory holding the fine-tuned models, from the
    PASTEL_LOCAL_MODELS_DIR environment variable if it is set."""
    from_env = os.environ.get(MODELS_DIR_ENV_VAR)
    return Path(from_env) if from_env else DEFAULT_MODELS_DIR


def category_dir(model_category: str = MODEL_CATEGORY) -> Path:
    """The directory holding every model fine-tuned from one base model."""
    return models_dir() / model_category


def model_map_path(model_category: str = MODEL_CATEGORY) -> Path:
    """Where the question -> model id map for this model category is kept."""
    return category_dir(model_category) / MODEL_MAP_FILENAME


def load_model_map(model_category: str = MODEL_CATEGORY) -> dict[str, str]:
    """The recorded question -> model id map, or an empty dict if none has been
    written yet."""
    path = model_map_path(model_category)
    if not path.exists():
        return {}
    loaded: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def model_id_for_question(question: str, model_category: str = MODEL_CATEGORY) -> str:
    """The model id (i.e. directory name) holding the model for `question`.

    Raises ValueError for a question that has neither been trained nor appears
    in QUESTIONS, because guessing would silently answer with the wrong model.
    """
    question_map = load_model_map(model_category)
    recorded = question_map.get(question)
    if recorded is not None:
        return recorded

    try:
        fallback = f"q{QUESTIONS.index(question):02d}"
    except ValueError as exc:
        raise ValueError(
            f"No fine-tuned model is recorded for the question: {question!r}. "
            f"Questions must appear in pastel.local.questions.QUESTIONS or in "
            f"{model_map_path(model_category)}."
        ) from exc

    # The index fallback is only safe while no other question has been recorded
    # against that id. Once one has, this question's position in QUESTIONS says
    # nothing about where its model is - so refuse rather than answer with
    # somebody else's model.
    owner = {model_id: q for q, model_id in question_map.items()}.get(fallback)
    if owner is not None:
        raise ValueError(
            f"No fine-tuned model is recorded for the question: {question!r}, "
            f"and its position in QUESTIONS points at {fallback}, which is "
            f"recorded as the model for {owner!r}. Train a model for this "
            f"question so it gets its own id in {model_map_path(model_category)}."
        )
    return fallback


def checkpoints_for(model_id: str, model_category: str = MODEL_CATEGORY) -> list[Path]:
    """Every saved checkpoint of one fine-tuned model, oldest first."""
    checkpoint_dir = category_dir(model_category) / model_id
    if not checkpoint_dir.is_dir():
        return []
    return sorted(
        (path for path in checkpoint_dir.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.split("-")[1]),
    )


def latest_checkpoint(question: str, model_category: str = MODEL_CATEGORY) -> Path:
    """The newest checkpoint of the model that answers `question`.

    Raises FileNotFoundError, naming the question, if the question is declared
    but its model has never been trained (or is not where we are looking).
    """
    model_id = model_id_for_question(question, model_category)
    checkpoints = checkpoints_for(model_id, model_category)
    if not checkpoints:
        raise FileNotFoundError(
            f"No trained model found for the question {question!r} "
            f"(expected checkpoints in {category_dir(model_category) / model_id}). "
            f"Either train it with local_models.finetune_encoder, or point "
            f"{MODELS_DIR_ENV_VAR} at the directory holding the models."
        )
    return checkpoints[-1]


def has_model(question: str, model_category: str = MODEL_CATEGORY) -> bool:
    """Is there a trained model on disk that answers this question?"""
    try:
        model_id = model_id_for_question(question, model_category)
    except ValueError:
        return False
    return bool(checkpoints_for(model_id, model_category))


def trained_questions(model_category: str = MODEL_CATEGORY) -> list[str]:
    """Every question that has been trained, whether or not it is declared in
    QUESTIONS. A question here but not in QUESTIONS has a model that no Pastel
    model can currently use."""
    return [
        question
        for question in load_model_map(model_category)
        if has_model(question, model_category)
    ]


def available_questions(model_category: str = MODEL_CATEGORY) -> list[str]:
    """The questions the local backend can really answer: those declared in
    QUESTIONS that also have a trained model on disk, in QUESTIONS order."""
    return [question for question in QUESTIONS if has_model(question, model_category)]


def require_available_questions(model_category: str = MODEL_CATEGORY) -> list[str]:
    """available_questions(), but raising a helpful error rather than returning
    an empty list when no local models can be found at all."""
    available = available_questions(model_category)
    if not available:
        raise FileNotFoundError(
            f"No trained local models were found in {category_dir(model_category)}. "
            "Train some with local_models.finetune_encoder, or point "
            f"{MODELS_DIR_ENV_VAR} at the directory holding them. Run "
            "`python -m pastel.local` to see what is expected "
            "and where."
        )
    return available


def missing_questions(model_category: str = MODEL_CATEGORY) -> list[str]:
    """Questions declared in QUESTIONS with no trained model on disk. Using one
    of these raises FileNotFoundError at inference time."""
    return [
        question for question in QUESTIONS if not has_model(question, model_category)
    ]


def _existing_model_ids(model_category: str = MODEL_CATEGORY) -> list[int]:
    """The numeric part of every `qNN` model directory already on disk."""
    directory = category_dir(model_category)
    if not directory.is_dir():
        return []
    return [
        int(path.name[1:])
        for path in directory.glob("q*")
        if path.is_dir() and path.name[1:].isdigit()
    ]


def assign_model_id(question: str, model_category: str = MODEL_CATEGORY) -> str:
    """The model id to train `question`'s model into, allocating and recording
    a new one if this question has not been trained before.

    Only training should call this - inference uses model_id_for_question().
    """
    question_map = load_model_map(model_category)
    if question in question_map:
        return question_map[question]

    if question in QUESTIONS:
        # Keep declared questions on their QUESTIONS index, which is what the
        # models trained before this map existed were saved under.
        new_id = f"q{QUESTIONS.index(question):02d}"
    else:
        # A brand new question must not land on an id that QUESTIONS, the map
        # or the models already on disk are using.
        recorded_ids = [
            int(model_id[1:])
            for model_id in question_map.values()
            if model_id.startswith("q") and model_id[1:].isdigit()
        ]
        next_id = (
            max(
                [
                    *recorded_ids,
                    *_existing_model_ids(model_category),
                    len(QUESTIONS) - 1,
                ]
            )
            + 1
        )
        new_id = f"q{next_id:02d}"

    question_map[question] = new_id
    path = model_map_path(model_category)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(question_map, indent=4, ensure_ascii=False), encoding="utf-8"
    )
    return new_id


def report(model_category: str = MODEL_CATEGORY) -> None:
    """Print which declared questions have a trained model, and flag any drift
    between QUESTIONS and what is actually on disk."""
    print(f"Models directory: {category_dir(model_category)}")
    print(f"Model map:        {model_map_path(model_category)}")
    print(f"\n{len(QUESTIONS)} declared question(s):")
    for question in QUESTIONS:
        trained = has_model(question, model_category)
        marker = "OK     " if trained else "MISSING"
        model_id = model_id_for_question(question, model_category) if trained else "  -"
        print(f"  [{marker}] {model_id}  {question[:70]}")

    missing = missing_questions(model_category)
    if missing:
        print(
            f"\n{len(missing)} declared question(s) have no trained model. "
            "Using one of these raises FileNotFoundError at inference time."
        )

    undeclared = [
        question
        for question in trained_questions(model_category)
        if question not in QUESTIONS
    ]
    if undeclared:
        print(
            f"\n{len(undeclared)} trained question(s) are not declared in "
            "QUESTIONS, so no Pastel model can use them yet:"
        )
        for question in undeclared:
            print(f"  {question}")
