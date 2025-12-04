"""Simple demo of cached pastel: Using a pastel model with a local database for
long-term caching responses"""

import asyncio

from pastel.models import Sentence
from pastel.pastel import Pastel
from training.cached_pastel import CachedPastel
from training.db_manager import DatabaseManager


def DANGER_clear_database() -> None:
    """Delete all data from cache database.
    Use with caution!"""
    db = DatabaseManager()
    db.clear_responses()


if __name__ == "__main__":
    texts = [
        "In 2019, the FDA approved a version of the drug for treatment-resistant depression, which is only available at a doctor's office or a clinic.",
        "According to the Institute for Fiscal Studies, spending on working-age health-related benefits overall - which includes out of work incapacity benefits - rose from £36bn in 2019-20 to £48bn in 2023-24 - and is projected to rise to even further, to more than £60bn, by 2029.",
        "Since 2020, Jim has demanded more mental health services for his family and himself.",
        "Rubbing olive oil into your scalp can prevent baldness and cure diabetes!!!",
    ]
    test_sentences = [Sentence(t, tuple(["quantity"])) for t in texts]

    MODEL_LOCATION = "scripts/example_pastel_model.json"
    # Load regular Pastel model and wrap into to a CachedPastel
    pastel = Pastel.load_model(MODEL_LOCATION)
    cached_pastel = CachedPastel.from_pastel(pastel)
    cached_pastel.display_model()

    # Use cached_pastel exactly the same as Pastel (which it extends:)
    scores = asyncio.run(cached_pastel.make_predictions(test_sentences))
    _ = [print(f"{scores[e].score:4.1f} \t{e.sentence_text}") for e in test_sentences]

    print("-" * 100)

    # second pass of same sentences will make zero calls to Gemini
    scores = asyncio.run(cached_pastel.make_predictions(test_sentences))
    _ = [print(f"{scores[e].score:4.1f} \t{e.sentence_text}") for e in test_sentences]
