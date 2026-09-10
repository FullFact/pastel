"""Answering Pastel questions with locally fine-tuned encoder models.

One encoder is fine-tuned to answer every question, with a small
classification head per question, so this side of the library is only the
*inference* half: the registry saying which head answers which question, and
loading the model to run it. Training it lives outside the library, in
`local_models`.

Needs the optional inference dependencies:

    uv sync --extra local
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
    "require_available_questions",
]
