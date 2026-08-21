"""Simple demo of cached pastel: using a Pastel model with a local database for
long-term caching of responses.

    python scripts/demo_cached_pastel.py                  # Gemini (the default)
    python scripts/demo_cached_pastel.py --backend local  # fine-tuned models

The cache wraps any backend, so the second pass over the same sentences makes
no model calls at all.
"""

import argparse
import asyncio
from typing import Type

from local_models.questions import QUESTIONS
from pastel import BACKENDS, DEFAULT_BACKEND, PastelLocal, PastelModel, get_backend
from pastel.models import Sentence
from training.cached_pastel import CachedPastel
from training.db_manager import DatabaseManager

# Only the Gemini backend can answer the questions in the checked-in example
# model; the local backend answers the questions it has models for.
GEMINI_EXAMPLE_MODEL = "scripts/example_pastel_model.json"


def DANGER_clear_database() -> None:
    """Delete all data from cache database.
    Use with caution!"""
    db = DatabaseManager()
    db.clear_responses()


def build_model(backend: Type[PastelModel]) -> PastelModel:
    """A model the given backend can answer. Weights are arbitrary for the
    local backend - this demo is about the caching, not the scores."""
    if issubclass(backend, PastelLocal):
        model = backend.from_feature_list(list(QUESTIONS))
        model.model = {feature: 0.5 for feature in model.model}
        return model
    return backend.load_model(GEMINI_EXAMPLE_MODEL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=sorted(BACKENDS),
        default=None,
        help="Which backend answers the questions. Defaults to the "
        f"PASTEL_BACKEND environment variable, or {DEFAULT_BACKEND}.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    args = parse_args()

    texts = [
        "In 2019, the FDA approved a version of the drug for treatment-resistant depression, which is only available at a doctor's office or a clinic.",
        "According to the Institute for Fiscal Studies, spending on working-age health-related benefits overall - which includes out of work incapacity benefits - rose from £36bn in 2019-20 to £48bn in 2023-24 - and is projected to rise to even further, to more than £60bn, by 2029.",
        "Since 2020, Jim has demanded more mental health services for his family and himself.",
        "Rubbing olive oil into your scalp can prevent baldness and cure diabetes!!!",
    ]
    test_sentences = [Sentence(t, tuple(["quantity"])) for t in texts]

    # Load a regular Pastel model and wrap it in a CachedPastel
    pastel = build_model(get_backend(args.backend))
    cached_pastel = CachedPastel.from_pastel(pastel)
    cached_pastel.display_model()
    print(f"Cache holds {cached_pastel.db.count_responses():,} responses")

    # Use cached_pastel exactly the same as the Pastel model it wraps:
    scores = asyncio.run(cached_pastel.make_predictions(test_sentences))
    _ = [print(f"{scores[e].score:4.1f} \t{e.sentence_text}") for e in test_sentences]

    print("-" * 100)

    # second pass of the same sentences will make zero calls to the backend
    scores = asyncio.run(cached_pastel.make_predictions(test_sentences))
    _ = [print(f"{scores[e].score:4.1f} \t{e.sentence_text}") for e in test_sentences]
