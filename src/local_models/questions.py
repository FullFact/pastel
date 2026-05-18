# Single source of truth for the question list used in label_sentences.py and finetune_encoder.py.
# Questions taken from scripts/example_pastel_model.json (excluding "bias").

QUESTIONS: list[str] = [
    "Is this making a claim that is too good to be true?",
    "Could believing this claim harm someone's health?",
    "Does this sentence relate to many people?",
    "Is this sentence likely to be believed by many people?",
    "Could believing this claim lead to violence?",
    "Does the sentence contain compare quantities, such as 'more' or 'less'?",
    "Answer 'yes' if this is a general or universal claim or answer 'no' if it is about a single event or individual",
    "Does the sentence discuss superlatives, such as 'biggest ever' or  'fastest growth'?",
    "Is this sentence interesting to the average reader?",
    "Does the sentence suggest a course of action?",
]
