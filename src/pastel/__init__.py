"""Pastel: score a piece of text by asking a fixed list of yes/no questions
about it and combining the answers with a set of learned weights.

Two backends answer those questions and are otherwise interchangeable:

* `PastelGemini` sends the questions to Gemini in a single prompt per sentence.
* `PastelLocal` uses one locally fine-tuned encoder model per question, and so
  can only answer questions that have been fine-tuned and recorded in its
  model map.

Choose one at runtime with `get_backend()` rather than by editing an import,
so the same code can be run either way.
"""

import os
from typing import Type

from pastel.pastel import PastelModel
from pastel.pastel_gemini import PastelGemini
from pastel.pastel_local import PastelLocal

BACKENDS: dict[str, Type[PastelModel]] = {
    "gemini": PastelGemini,
    "local": PastelLocal,
}
BACKEND_ENV_VAR = "PASTEL_BACKEND"
DEFAULT_BACKEND = "gemini"

__all__ = [
    "BACKENDS",
    "BACKEND_ENV_VAR",
    "DEFAULT_BACKEND",
    "PastelGemini",
    "PastelLocal",
    "PastelModel",
    "get_backend",
]


def get_backend(name: str | None = None) -> Type[PastelModel]:
    """Return the Pastel class to use for answering questions.

    `name` is one of the keys of BACKENDS. If it is None, the PASTEL_BACKEND
    environment variable is used, falling back to DEFAULT_BACKEND.
    """
    if name is None:
        name = os.environ.get(BACKEND_ENV_VAR, DEFAULT_BACKEND)
    try:
        return BACKENDS[name.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown Pastel backend {name!r}. "
            f"Choose one of: {', '.join(sorted(BACKENDS))}."
        ) from None
