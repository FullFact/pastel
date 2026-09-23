"""Pastel: score a piece of text by asking a fixed list of yes/no questions
about it and combining the answers with a set of learned weights.

Two interchangeable backends answer those questions: `PastelGemini` sends them
to Gemini in one prompt per sentence; `PastelLocal` uses a locally fine-tuned
encoder, and so can only answer questions it has been trained for. Choose one
at runtime with `get_backend()` rather than by editing an import.
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
    """The Pastel class to use for answering questions: one of the keys of
    BACKENDS, or `$PASTEL_BACKEND` falling back to DEFAULT_BACKEND."""
    if name is None:
        name = os.environ.get(BACKEND_ENV_VAR, DEFAULT_BACKEND)
    try:
        return BACKENDS[name.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown Pastel backend {name!r}. "
            f"Choose one of: {', '.join(sorted(BACKENDS))}."
        ) from None
