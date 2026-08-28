"""Which fine-tuned model answers which question, and where those models live.

Each question gets its own fine-tuned model, saved in a directory named after
a short id (`q00`, `q01`, ...). Training allocates those ids and records them
in `model_map.json`; inference looks up the same mapping, or it would answer a
question with another question's model.

That map is the only source of truth for which questions the local backend can
answer. The library declares no questions of its own: which questions to ask,
and the weights to combine them with, belong to the downstream task.
"""

import json
import os
from pathlib import Path

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

    Raises ValueError for a question the map has no entry for, because guessing
    would silently answer with the wrong model.
    """
    recorded = load_model_map(model_category).get(question)
    if recorded is None:
        raise ValueError(
            f"No fine-tuned model is recorded for the question: {question!r}. "
            f"Train one with local_models.finetune_encoder, which records it "
            f"in {model_map_path(model_category)}."
        )
    return recorded


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

    Raises FileNotFoundError, naming the question, if the question is recorded
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


def available_questions(model_category: str = MODEL_CATEGORY) -> list[str]:
    """The questions the local backend can answer: every question in the model
    map with a trained model on disk, in the order the map records them."""
    return [
        question
        for question in load_model_map(model_category)
        if has_model(question, model_category)
    ]


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

    # A new question must not land on an id the map or the models already on
    # disk are using, or training it would overwrite an existing model.
    recorded_ids = [
        int(model_id[1:])
        for model_id in question_map.values()
        if model_id.startswith("q") and model_id[1:].isdigit()
    ]
    next_id = max([*recorded_ids, *_existing_model_ids(model_category), -1]) + 1
    new_id = f"q{next_id:02d}"

    question_map[question] = new_id
    path = model_map_path(model_category)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(question_map, indent=4, ensure_ascii=False), encoding="utf-8"
    )
    return new_id


def report(model_category: str = MODEL_CATEGORY) -> None:
    """Print which recorded questions have a trained model on disk."""
    print(f"Models directory: {category_dir(model_category)}")
    print(f"Model map:        {model_map_path(model_category)}")

    question_map = load_model_map(model_category)
    if not question_map:
        print("\nNo questions are recorded in the model map.")
        return

    print(f"\n{len(question_map)} recorded question(s):")
    missing = []
    for question, model_id in question_map.items():
        trained = has_model(question, model_category)
        if not trained:
            missing.append(question)
        marker = "OK     " if trained else "MISSING"
        print(f"  [{marker}] {model_id}  {question[:70]}")

    if missing:
        print(
            f"\n{len(missing)} recorded question(s) have no trained model. "
            "Using one of these raises FileNotFoundError at inference time."
        )
