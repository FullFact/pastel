from pastel import pastel_functions
from training.beam_search import run_beam_search
from training.db_manager import DatabaseManager


def get_functions() -> list[str]:
    """Return list of all functions available to Pastel"""
    all_functs = [
        getattr(pastel_functions, str(feature)) for feature in pastel_functions.__all__
    ]
    return all_functs


def get_questions(use_all_from_db: bool = False) -> list[str]:
    """Either load all available questions from the local cached-pastel database
    or return a sample."""
    if use_all_from_db:
        db = DatabaseManager()
        all_questions = db.get_unique_questions()
    else:
        # just a small set to test things out
        all_questions = [
            "Answer 'yes' if this sentence is making a specific claim or answer 'no' if it is vague or unclear",
            "Does this sentence relate to many people?",
            "Is this sentence about someone's personal experience?",
            "Does the sentence contain specific numbers or quantities?",
            "Does the sentence contain compare quantities, such as 'more' or 'less'?",
            "Could believing this claim harm someone's health?",
            "Could believing this claim lead to violence",
        ]
    return all_questions


if __name__ == "__main__":
    features = get_questions(True)
    features.extend(get_functions())
    best_model, F1 = run_beam_search(features, beta=4, max_iter=10)
    if best_model:
        print(f"\nBest model (with {F1=:.5}):")
        best_model.display_model()
    else:
        print("No model found!")
