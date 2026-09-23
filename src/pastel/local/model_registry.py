"""Which head of the fine-tuned model answers which question, and where that
model lives.

One encoder answers every question, with a small classification head per
question. `model_map.json` records which head index answers which question:
training writes it, inference reads it, or a question would be answered by
another question's head. It is the only source of truth for what the local
backend can answer - the library declares no questions of its own.
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
MODEL_DIR_NAME = "multi_head"

# Where the fine-tuned models are kept. The default is relative, so it only
# resolves from the repo root - set the environment variable to an absolute
# path anywhere else, production included.
MODELS_DIR_ENV_VAR = "PASTEL_LOCAL_MODELS_DIR"
DEFAULT_MODELS_DIR = Path("data/local_models/models")


def models_dir() -> Path:
    """The directory holding the fine-tuned models, from
    PASTEL_LOCAL_MODELS_DIR if it is set."""
    from_env = os.environ.get(MODELS_DIR_ENV_VAR)
    return Path(from_env) if from_env else DEFAULT_MODELS_DIR


def category_dir(model_category: str = MODEL_CATEGORY) -> Path:
    """The directory holding everything fine-tuned from one base model."""
    return models_dir() / model_category


def model_dir(model_category: str = MODEL_CATEGORY) -> Path:
    """The directory the fine-tuned model's checkpoints are saved in."""
    return category_dir(model_category) / MODEL_DIR_NAME


def model_map_path(model_category: str = MODEL_CATEGORY) -> Path:
    """Where the question -> head index map is kept."""
    return category_dir(model_category) / MODEL_MAP_FILENAME


def load_model_map(model_category: str = MODEL_CATEGORY) -> dict[str, int]:
    """The recorded question -> head index map, or `{}` if none was written."""
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
            "those models have to be retrained."
        )
    return loaded


def record_heads(
    questions_in_head_order: list[str],
    models_dir: Path | None = None,
    model_category: str = MODEL_CATEGORY,
) -> dict[str, int]:
    """Write the question -> head index map, and return it.

    Takes the questions in the order the heads were trained in - the only
    thing that says which head answers which. The map is replaced rather than
    merged: the body is shared, so a retrain produces a whole new model and a
    leftover entry would point at a head trained for something else.
    """
    path = (
        model_map_path(model_category)
        if models_dir is None
        else Path(models_dir) / model_category / MODEL_MAP_FILENAME
    )
    heads = {question: head for head, question in enumerate(questions_in_head_order)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(heads, indent=4, ensure_ascii=False), encoding="utf-8")
    return heads


def head_for_question(question: str, model_category: str = MODEL_CATEGORY) -> int:
    """The index of the head that answers `question`. Raises rather than
    guessing, which would answer with the wrong head."""
    recorded = load_model_map(model_category).get(question)
    if recorded is None:
        raise ValueError(
            f"No fine-tuned model is recorded for the question: {question!r}. "
            f"Training records it in {model_map_path(model_category)}."
        )
    return recorded


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
    """The newest checkpoint of the fine-tuned model."""
    saved = checkpoints(model_category)
    if not saved:
        raise FileNotFoundError(
            f"No trained model found (expected checkpoints in "
            f"{model_dir(model_category)}). Either train one, or point "
            f"{MODELS_DIR_ENV_VAR} at the directory holding the models."
        )
    return saved[-1]


def has_model(question: str, model_category: str = MODEL_CATEGORY) -> bool:
    """Is there a trained model on disk with a head for this question?"""
    if question not in load_model_map(model_category):
        return False
    return bool(checkpoints(model_category))


def available_questions(model_category: str = MODEL_CATEGORY) -> list[str]:
    """The questions the local backend can answer: every question in the model
    map, in the order it records them, if the model has been trained."""
    if not checkpoints(model_category):
        return []
    return list(load_model_map(model_category))


def require_available_questions(model_category: str = MODEL_CATEGORY) -> list[str]:
    """available_questions(), raising rather than returning an empty list when
    no local model can be found at all."""
    available = available_questions(model_category)
    if not available:
        raise FileNotFoundError(
            f"No trained local model was found in {model_dir(model_category)}. "
            f"Train one, or point {MODELS_DIR_ENV_VAR} at the directory "
            "holding it. `python -m pastel.local` shows what is expected where."
        )
    return available


def report(model_category: str = MODEL_CATEGORY) -> None:
    """Print which recorded questions have a trained head on disk."""
    print(f"Model directory: {model_dir(model_category)}")
    print(f"Model map:       {model_map_path(model_category)}")

    question_map = load_model_map(model_category)
    if not question_map:
        print("\nNo questions are recorded in the model map.")
        return

    marker = "OK     " if checkpoints(model_category) else "MISSING"
    print(f"\n{len(question_map)} recorded question(s):")
    for question, head in question_map.items():
        print(f"  [{marker}] head {head:2d}  {question[:70]}")

    if marker == "MISSING":
        print(
            "\nThe model has never been trained (or is not where we are "
            "looking), so none of these questions can be answered."
        )
