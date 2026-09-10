"""Answering Pastel questions with locally fine-tuned encoder models.

One small encoder model is fine-tuned per question, so this side of the
library is only the *inference* half: the registry saying which model answers
which question, and loading those models to run them. Training new ones lives
outside the library, in `local_models`.

Needs the optional inference dependencies:

    uv sync --extra local
"""

from pastel.local.local_answerer import answer_question, preload_models
from pastel.local.model_registry import (
    available_questions,
    has_model,
    model_id_for_question,
    models_dir,
    require_available_questions,
)

__all__ = [
    "answer_question",
    "available_questions",
    "has_model",
    "model_id_for_question",
    "models_dir",
    "preload_models",
    "require_available_questions",
]
