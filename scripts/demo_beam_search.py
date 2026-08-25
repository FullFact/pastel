"""Feature selection for a Pastel model.

The backend that answers the questions is chosen at runtime:

    python scripts/demo_beam_search.py                  # Gemini (the default)
    python scripts/demo_beam_search.py --backend local  # fine-tuned models

Every answer is cached in the local database whichever backend is used, so a
second run over the same (question, sentence) pairs makes no model calls.
"""

import argparse
from typing import Type

from pastel import BACKENDS, DEFAULT_BACKEND, PastelLocal, PastelModel, get_backend
from pastel.local import require_available_questions
from pastel.models import FEATURE_TYPE
from training.beam_search import run_beam_search
from training.db_manager import DatabaseManager

# Just a small set to test things out. The local backend can't use these -
# it only answers questions it has a fine-tuned model for.
SAMPLE_QUESTIONS = [
    "Answer 'yes' if this sentence is making a specific claim or answer 'no' if it is vague or unclear",
    "Does this sentence relate to many people?",
    "Is this sentence about someone's personal experience?",
    "Does the sentence contain specific numbers or quantities?",
    "Does the sentence contain compare quantities, such as 'more' or 'less'?",
    "Could believing this claim harm someone's health?",
    "Could believing this claim lead to violence",
]


def get_functions() -> list[FEATURE_TYPE]:
    """Return list of all functions available to Pastel"""
    from pastel import pastel_functions

    all_functs: list[FEATURE_TYPE] = [
        getattr(pastel_functions, str(feature)) for feature in pastel_functions.__all__
    ]
    return all_functs


def get_questions(
    backend: Type[PastelModel], use_all_from_db: bool = False
) -> list[str]:
    """The questions to select from.

    The local backend is limited to the questions whose fine-tuned models have
    actually been trained. Otherwise, either load every question in the local
    cached-pastel database or return a sample.
    """
    if issubclass(backend, PastelLocal):
        return require_available_questions()
    if use_all_from_db:
        db = DatabaseManager()
        return db.get_unique_questions()
    return list(SAMPLE_QUESTIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=sorted(BACKENDS),
        default=None,
        help="Which backend answers the questions. Defaults to the "
        f"PASTEL_BACKEND environment variable, or {DEFAULT_BACKEND}.",
    )
    parser.add_argument(
        "--all-questions-from-db",
        action="store_true",
        help="Select from every question already in the cache database, "
        "rather than a small sample. Ignored by the local backend.",
    )
    parser.add_argument("--beta", type=int, default=4, help="Beam width.")
    parser.add_argument(
        "--max-iter", type=int, default=10, help="Maximum number of features."
    )
    return parser.parse_args()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    args = parse_args()
    backend = get_backend(args.backend)

    features: list[FEATURE_TYPE] = list(
        get_questions(backend, args.all_questions_from_db)
    )
    features.extend(get_functions())

    best_model, F1 = run_beam_search(
        features, beta=args.beta, max_iter=args.max_iter, backend=backend
    )
    if best_model:
        print(f"\nBest model (with {F1=:.5}):")
        best_model.display_model()
    else:
        print("No model found!")
