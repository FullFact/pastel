"""Demo of the two Pastel backends: training a model, loading one and scoring
sentences with it.

The backend is chosen at runtime, so the same demo exercises both:

    python scripts/demo_pastel.py                  # Gemini (the default)
    python scripts/demo_pastel.py --backend local  # locally fine-tuned models
    PASTEL_BACKEND=local python scripts/demo_pastel.py

The local backend can only answer the questions it has fine-tuned models for
(local_models.questions.QUESTIONS), so the demo picks its question set to suit
whichever backend is in use.
"""

import argparse
import asyncio
import json
import tempfile
from typing import Type

from local_models.questions import QUESTIONS
from pastel import (
    BACKENDS,
    DEFAULT_BACKEND,
    PastelLocal,
    PastelModel,
    get_backend,
)
from pastel.models import FEATURE_TYPE, BiasType, Sentence
from pastel.optimise_weights import learn_weights

# A model file whose questions the Gemini backend can answer. The local backend
# has no equivalent, because its questions are fixed by which models exist.
GEMINI_EXAMPLE_MODEL = "scripts/example_pastel_model.json"
TRAINED_MODEL_OUT = "scripts/new_demo_pastel_model.json"


def demo_questions(backend: Type[PastelModel]) -> list[str]:
    """A question set the given backend can actually answer."""
    if issubclass(backend, PastelLocal):
        return list(QUESTIONS)
    return [
        "Is this sentence about olive oil?",
        "Is this about a disease or illness?",
    ]


def demo_predict(pasteliser: PastelModel) -> None:
    # pass a few examples & see what scores we get:
    texts = [
        "Over a similar time period, reported mental health problems have also jumped from 8% to 10% of working-age people to between 13% and 15%, according to the Institute for Fiscal Studies.",
        "For every 4in increase in height above average, cancer risk increases by 18 per cent in women and 11 per cent in men, reported researchers at the Karolinska Institute in Sweden in 2015.",
        "Researchers at Oxford University in 2017 found that every extra 4in of height above average increases a man's risk of developing aggressive prostate cancer by 21 per cent and their chance of dying by 17 per cent.",
        "SEVEN in 10 women will experience period pain - often physically and mentally debilitating - for almost four solid years of their life, according to research.",
        "It's been 70 years since the Toon celebrated getting their hands on some silverware, when they beat Manchester City to win the 1955 FA Cup.",
        "Alexander Isak did a very, very cool thing against Virgil van Dijk when they played in at St James' Park earlier in the season, which finished 3-3..",
        "We've got a fairly similar formation set up for both teams - they're going to set up as 4-3-3 or 4-2-3-1, fairly similar.",
        "The supplier serves about a quarter of the UK's population, mostly across London and parts of southern England, and employs 8,000 people.",
        "Environment Secretary Steve Reed has previously said government intervention in Thames Water would 'cost billions and take years'.",
    ]
    examples = [Sentence(t, tuple(["quantity"])) for t in texts]

    scores = asyncio.run(pasteliser.make_predictions(examples))
    _ = [print(f"{scores[e].score:4.1f} \t{e.sentence_text}") for e in examples]


TRAINING_EXAMPLES = [
    {
        "sentence_text": '"Ending tax breaks for private schools will raise £1.8bn a year by 2029/30 to help deliver 6,500 new teachers and raise school standards, supporting the 94 per cent of children in state schools to achieve and thrive.',
        "score": "4.0",
        "claim_types": ["quantity"],
    },
    {
        "sentence_text": "'Ending tax breaks for private schools will increase investment in state education - raising £1.8 billion a year by 2030.",
        "score": "4.0",
        "claim_types": ["quantity", "predictions"],
    },
    {
        "sentence_text": '"Since the Labour government imposed VAT at the ­standard rate of 20 per cent on private school fees, thousands of parents whose children attend independent schools because of special educational needs and disabilities (Send) have been forced into drastic financial decisions. A study commissioned by the Education Not Taxation campaign has found that one in five of these families has already remortgaged their home to help finance the cost of rising school fees, and one in eight has sold their home and moved.The campaign is bringing a discrimination case against the government, set to begin on Tuesday, in which parents will argue that Send children have been disproportionately affected by the ­introduction of VAT on school fees." - The Times',
        "score": "3.0",
        "claim_types": ["quantity", "correlation", "other"],
    },
    {
        "sentence_text": 'Money from taxing private school fees will support the 94 per cent of children in state schools to "achieve and thrive", the Treasury said.',
        "score": "4.0",
        "claim_types": ["quantity", "predictions"],
    },
    {
        "sentence_text": "Of these respondents, 12 per cent said they had moved home or downsized to pay for Labour's 20 per cent VAT on private school fees, which came into force in January.",
        "score": "4.0",
        "claim_types": ["quantity"],
    },
]

# Claim-type features to add alongside the questions when training. These are
# names from pastel_functions; from_feature_list() resolves them to functions.
DEMO_FUNCTIONS = [
    "is_claim_type_quantity",
    "is_claim_type_personal",
    "is_claim_type_predictions",
    "is_claim_type_rules",
]


def demo_learn(pasteliser: PastelModel) -> PastelModel:
    """Train a copy of `pasteliser` - same questions, plus a few claim-type
    functions - on a handful of scored examples, and return the trained model."""
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".jsonl"
    ) as temp_file:
        for eg in TRAINING_EXAMPLES:
            temp_file.write(json.dumps(eg) + "\n")
        temp_training_data_file_path = temp_file.name

    # from_feature_list() resolves the function names to the functions
    # themselves and adds the bias term, so the new model is a valid one.
    features = pasteliser.get_questions() + DEMO_FUNCTIONS
    new_pasteliser = type(pasteliser).from_feature_list(features)

    # learn_weights() writes the optimised weights back into the model it is
    # given, so there is nothing to assign here.
    learn_weights(temp_training_data_file_path, new_pasteliser)
    return new_pasteliser


def demo(backend: Type[PastelModel]) -> None:
    questions = demo_questions(backend)

    # train a new model
    print("TRAIN A NEW MODEL")
    print("-" * 100)
    pasteliser = backend.from_feature_list(questions)
    trained = demo_learn(pasteliser)
    trained.display_model()
    trained.save_model(TRAINED_MODEL_OUT)
    demo_predict(trained)
    print("-" * 100)

    # load a model from a dictionary
    print("LOAD FROM DICTIONARY")
    print("-" * 100)
    model: dict[FEATURE_TYPE, float] = {BiasType.BIAS: 1.0}
    for question in questions:
        model[question] = 0.5
    demo_predict(backend(model))
    print("-" * 100)

    # load a model from a file
    print("LOAD FROM FILE")
    print("-" * 100)
    # Only the Gemini backend can answer the questions in the checked-in
    # example model, so the local backend reloads the model just trained.
    model_file = (
        TRAINED_MODEL_OUT if issubclass(backend, PastelLocal) else GEMINI_EXAMPLE_MODEL
    )
    print(f"Loading {model_file}")
    from_file = backend.load_model(model_file)
    from_file.display_model()
    demo_predict(from_file)
    print("-" * 100)

    print("RE-TRAIN THE LOADED MODEL")
    print("-" * 100)
    print("Old model:")
    from_file.display_model()
    print("New model:")
    demo_learn(from_file).display_model()


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
    demo(get_backend(args.backend))
