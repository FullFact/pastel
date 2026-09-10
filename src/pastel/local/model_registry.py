"""Which head of the fine-tuned model answers which question, and where that
model lives.

One encoder answers every question, with a small classification head per
question. `model_map.json` records which head index answers which question:
training allocates the indices, inference looks up the same mapping, or it
would answer a question with another question's head.

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
MODEL_MAP_FILENAME = "model_map.json"  # question -> head index

# The one fine-tuned model, under its base model's directory.
MODEL_DIR_NAME = "multi_head"

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
    """The directory holding everything fine-tuned from one base model."""
    return models_dir() / model_category


def model_dir(model_category: str = MODEL_CATEGORY) -> Path:
    """The directory the fine-tuned model's checkpoints are saved in."""
    return category_dir(model_category) / MODEL_DIR_NAME


def model_map_path(model_category: str = MODEL_CATEGORY) -> Path:
    """Where the question -> head index map for this model category is kept."""
    return category_dir(model_category) / MODEL_MAP_FILENAME


def load_model_map(model_category: str = MODEL_CATEGORY) -> dict[str, int]:
    """The recorded question -> head index map, or an empty dict if none has
    been written yet."""
    path = model_map_path(model_category)
    if not path.exists():
        return {}
    loaded: dict[str, int] = json.loads(path.read_text(encoding="utf-8"))
    wrong_type = [
        question for question, head in loaded.items() if not isinstance(head, int)
    ]
    if wrong_type:
        raise ValueError(
            f"{path} maps questions to something other than a head index "
            f"(e.g. {loaded[wrong_type[0]]!r}). Maps written before one model "
            "answered every question recorded a per-question model id instead; "
            "those models have to be retrained together with "
            "local_models.finetune_encoder."
        )
    return loaded


def head_for_question(question: str, model_category: str = MODEL_CATEGORY) -> int:
    """The index of the head that answers `question`.

    Raises ValueError for a question the map has no entry for, because guessing
    would silently answer with the wrong head.
    """
    recorded = load_model_map(model_category).get(question)
    if recorded is None:
        raise ValueError(
            f"No fine-tuned model is recorded for the question: {question!r}. "
            f"Train one with local_models.finetune_encoder, which records it "
            f"in {model_map_path(model_category)}."
        )
    return recorded


def head_count(model_category: str = MODEL_CATEGORY) -> int:
    """How many heads a model covering every recorded question needs. Only
    training needs this - inference reads the count from the checkpoint."""
    heads = load_model_map(model_category).values()
    return max(heads) + 1 if heads else 0


def checkpoints(model_category: str = MODEL_CATEGORY) -> list[Path]:
    """Every saved checkpoint of the fine-tuned model, oldest first."""
    directory = model_dir(model_category)
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.split("-")[1]),
    )


def latest_checkpoint(model_category: str = MODEL_CATEGORY) -> Path:
    """The newest checkpoint of the fine-tuned model.

    Raises FileNotFoundError if it has never been trained, or is not where we
    are looking.
    """
    saved = checkpoints(model_category)
    if not saved:
        raise FileNotFoundError(
            f"No trained model found (expected checkpoints in "
            f"{model_dir(model_category)}). Either train one with "
            f"local_models.finetune_encoder, or point {MODELS_DIR_ENV_VAR} at "
            "the directory holding the models."
        )
    return saved[-1]


def has_model(question: str, model_category: str = MODEL_CATEGORY) -> bool:
    """Is there a trained model on disk with a head for this question?"""
    if question not in load_model_map(model_category):
        return False
    return bool(checkpoints(model_category))


def available_questions(model_category: str = MODEL_CATEGORY) -> list[str]:
    """The questions the local backend can answer: every question in the model
    map, in the order the map records them, if the model has been trained."""
    if not checkpoints(model_category):
        return []
    return list(load_model_map(model_category))


def require_available_questions(model_category: str = MODEL_CATEGORY) -> list[str]:
    """available_questions(), but raising a helpful error rather than returning
    an empty list when no local model can be found at all."""
    available = available_questions(model_category)
    if not available:
        raise FileNotFoundError(
            f"No trained local model was found in {model_dir(model_category)}. "
            "Train one with local_models.finetune_encoder, or point "
            f"{MODELS_DIR_ENV_VAR} at the directory holding it. Run "
            "`python -m pastel.local` to see what is expected "
            "and where."
        )
    return available


def assign_head(question: str, model_category: str = MODEL_CATEGORY) -> int:
    """The head index that answers `question`, allocating and recording a new
    one if this question has not been trained before.

    Only training should call this - inference uses head_for_question().
    """
    question_map = load_model_map(model_category)
    if question in question_map:
        return question_map[question]

    new_head = head_count(model_category)
    question_map[question] = new_head
    path = model_map_path(model_category)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(question_map, indent=4, ensure_ascii=False), encoding="utf-8"
    )
    return new_head


def report(model_category: str = MODEL_CATEGORY) -> None:
    """Print which recorded questions have a trained head on disk."""
    print(f"Model directory: {model_dir(model_category)}")
    print(f"Model map:       {model_map_path(model_category)}")

    question_map = load_model_map(model_category)
    if not question_map:
        print("\nNo questions are recorded in the model map.")
        return

    trained = bool(checkpoints(model_category))
    print(f"\n{len(question_map)} recorded question(s):")
    for question, head in question_map.items():
        marker = "OK     " if trained else "MISSING"
        print(f"  [{marker}] head {head:2d}  {question[:70]}")

    if not trained:
        print(
            "\nThe model has never been trained (or is not where we are "
            "looking), so none of these questions can be answered. One model "
            "answers all of them, so they are trained together."
        )
