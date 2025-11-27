# Functions called by Pastel models should take a string and return a float.
# The output will be multiplied by the corresponding weight defined in the Pastel model.
# The plan is to enhance these with functions for claim-types and news-categories,
# among others.

__all__ = [
    "is_short",
    "has_number",
    "is_claim_type_personal",
    "is_claim_type_quantity",
    "is_claim_type_correlation",
    "is_claim_type_rules",
    "is_claim_type_predictions",
    "is_claim_type_voting",
    "is_claim_type_opinion",
    "is_claim_type_support",
    "is_claim_type_other",
    "is_claim_type_not_claim",
]


from pastel.models import Sentence


def is_short(sentence: Sentence) -> float:
    """Demo function: is the text short?"""
    return float(len(sentence.sentence_text) < 30)


def has_number(sentence: Sentence) -> float:
    """Is there a number (0-9) in this sentence?"""
    return any(char.isdigit() for char in sentence.sentence_text)


def is_claim_type_personal(sentence: Sentence) -> float:
    """Does this sentence have this claim type?"""
    return "personal" in sentence.claim_type


def is_claim_type_quantity(sentence: Sentence) -> float:
    """Does this sentence have this claim type?"""
    return "quantity" in sentence.claim_type


def is_claim_type_correlation(sentence: Sentence) -> float:
    """Does this sentence have this claim type?"""
    return "correlation" in sentence.claim_type


def is_claim_type_rules(sentence: Sentence) -> float:
    """Does this sentence have this claim type?"""
    return "rules" in sentence.claim_type


def is_claim_type_predictions(sentence: Sentence) -> float:
    """Does this sentence have this claim type?"""
    return "predictions" in sentence.claim_type


def is_claim_type_voting(sentence: Sentence) -> float:
    """Does this sentence have this claim type?"""
    return "voting" in sentence.claim_type


def is_claim_type_opinion(sentence: Sentence) -> float:
    """Does this sentence have this claim type?"""
    return "opinion" in sentence.claim_type


def is_claim_type_support(sentence: Sentence) -> float:
    """Does this sentence have this claim type?"""
    return "support" in sentence.claim_type


def is_claim_type_other(sentence: Sentence) -> float:
    """Does this sentence have this claim type?"""
    return "other" in sentence.claim_type


def is_claim_type_not_claim(sentence: Sentence) -> float:
    """Does this sentence have this claim type?"""
    return "not_claim" in sentence.claim_type
