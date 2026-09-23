"""Answering Pastel questions with a locally fine-tuned encoder.

One encoder answers every question, with a classification head per question.
`model_registry` records which head answers which; `local_answerer` loads the
model and runs it; `training` fine-tunes a new one.

Needs the optional dependencies: `uv sync --extra local` to use a model,
`--extra train` to train one.
"""

from pastel.local.local_answerer import (
    answer_question,
    answer_questions,
    preload_models,
)
from pastel.local.model_registry import (
    available_questions,
    has_model,
    head_for_question,
    models_dir,
    record_heads,
    require_available_questions,
)

__all__ = [
    "answer_question",
    "answer_questions",
    "available_questions",
    "has_model",
    "head_for_question",
    "models_dir",
    "preload_models",
    "record_heads",
    "require_available_questions",
]
