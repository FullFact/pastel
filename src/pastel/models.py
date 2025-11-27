from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Sentence:
    """Sentence text plus metadata (list of claim types)"""

    sentence_text: str
    claim_type: Tuple[str, ...] = ()
