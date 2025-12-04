import copy
import json
import tempfile
import asyncio

from pastel.models import BiasType, Sentence
from pastel.optimise_weights import learn_weights
from pastel.pastel import Pastel


def demo_predict(pasteliser: Pastel) -> None:
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


def demo_learn(pasteliser: Pastel) -> Pastel:
    training_egs = [
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
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".jsonl"
    ) as temp_file:
        for eg in training_egs:
            temp_file.write(json.dumps(eg) + "\n")
        temp_training_data_file_path = temp_file.name

    new_pasteliser = copy.copy(pasteliser)
    new_pasteliser.model["is_claim_type_quantity"] = 0
    new_pasteliser.model["is_claim_type_personal"] = 0
    new_pasteliser.model["is_claim_type_predictions"] = 0
    new_pasteliser.model["is_claim_type_rules"] = 0
    new_weights = learn_weights(temp_training_data_file_path, new_pasteliser)

    new_pasteliser.model = {
        feat: weight for feat, weight in zip(new_pasteliser.model.keys(), new_weights)
    }
    return new_pasteliser


def demo() -> None:
    # train a new model
    print("TRAIN A NEW MODEL")
    print("-" * 100)
    questions = [
        "Is this sentence about olive oil?",
        "Is this about a disease or illness?",
    ]
    pasteliser = Pastel.from_feature_list(questions)
    demo_learn(pasteliser)
    pasteliser.save_model("scripts/new_demo_pastel_model.json")
    demo_predict(pasteliser)
    print("-" * 100)

    # load a model from a dictionary
    print("LOAD FROM DICTIONARY")
    print("-" * 100)
    model = {
        BiasType.BIAS: 1.0,
        "Is this sentence about olive oil?": 0.9,
        "Is this about a disease or illness?": 0.3,
    }

    pasteliser = Pastel(model)
    demo_predict(pasteliser)
    print("-" * 100)

    # load a model from a file
    print("LOAD FROM FILE")
    print("-" * 100)
    pasteliser = Pastel.load_model("scripts/example_pastel_model.json")
    demo_predict(pasteliser)


if __name__ == "__main__":
    demo()
    pastel = Pastel.load_model("scripts/example_pastel_model.json")
    demo_predict(pastel)
    print("Old model:")
    pastel.display_model()
    new_pastel = demo_learn(pastel)
    print("New model:")
    new_pastel.display_model()
