import enum
from collections.abc import Callable
from dataclasses import dataclass
from typing import Tuple, TypeAlias

from pydantic import BaseModel


class BiasType(enum.Enum):
    """Used as the key for the bias term in Pastel models"""

    BIAS = "BIAS"


@dataclass(frozen=True)
class Sentence:
    """Sentence text plus metadata (list of claim types)"""

    sentence_text: str
    claim_type: Tuple[str, ...] = ()


FEATURE_TYPE: TypeAlias = Callable[[Sentence], float] | str | BiasType


class ScoreAndAnswers(BaseModel):
    """Used to parse scores for sentences and store the answers to
    PASTEL questions."""

    sentence: Sentence
    score: float
    answers: dict[FEATURE_TYPE, float]
