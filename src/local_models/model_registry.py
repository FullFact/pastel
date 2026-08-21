"""Which fine-tuned model answers which question.

Each question gets its own fine-tuned model, saved in a directory named after
a short id (`q00`, `q01`, ...). Training allocates those ids and records them
in `model_map.json`; inference has to look up the same mapping, or it would
answer a question with another question's model.

Before this map existed, ids were implicitly the question's index in
`QUESTIONS`, so that is still the fallback when a question is not in the map.
Note that the fallback is only correct while `QUESTIONS` keeps the order the
models were trained in - which is exactly why the map is written.
"""

import json
from pathlib import Path

from local_models.questions import QUESTIONS

MODELS: dict[str, str] = {
    "ModernBERT-multilingual": "jhu-clsp/mmBERT-base",
    "mDeBERTa-v3-base": "microsoft/mdeberta-v3-base",
    "XLM-RoBERTa-base": "FacebookAI/xlm-roberta-base",
}

MODEL_CATEGORY = "ModernBERT-multilingual"  # only using this one for now
MODELS_DIR = Path("data/local_models/models")
MODEL_MAP_FILENAME = "model_map.json"


def model_map_path(model_category: str = MODEL_CATEGORY) -> Path:
    """Where the question -> model id map for this model category is kept."""
    return MODELS_DIR / model_category / MODEL_MAP_FILENAME


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
    recorded = load_model_map(model_category).get(question)
    if recorded is not None:
        return recorded

    try:
        return f"q{QUESTIONS.index(question):02d}"
    except ValueError as exc:
        raise ValueError(
            f"No fine-tuned model is recorded for the question: {question!r}. "
            f"Questions must appear in local_models.questions.QUESTIONS or in "
            f"{model_map_path(model_category)}."
        ) from exc


def _existing_model_ids(model_category: str = MODEL_CATEGORY) -> list[int]:
    """The numeric part of every `qNN` model directory already on disk."""
    category_dir = MODELS_DIR / model_category
    if not category_dir.is_dir():
        return []
    return [
        int(path.name[1:])
        for path in category_dir.glob("q*")
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
