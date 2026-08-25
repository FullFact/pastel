"""Tests for choosing a backend and for the local backend's constraints."""

import pytest

from pastel import (
    BACKEND_ENV_VAR,
    DEFAULT_BACKEND,
    PastelGemini,
    PastelLocal,
    get_backend,
)
from pastel.local.questions import QUESTIONS
from pastel.models import BiasType

# mypy: ignore-errors


def test_default_backend_is_gemini(monkeypatch) -> None:
    monkeypatch.delenv(BACKEND_ENV_VAR, raising=False)
    assert DEFAULT_BACKEND == "gemini"
    assert get_backend() is PastelGemini


def test_backend_by_name() -> None:
    assert get_backend("gemini") is PastelGemini
    assert get_backend("local") is PastelLocal
    assert get_backend("LOCAL") is PastelLocal


def test_backend_from_environment(monkeypatch) -> None:
    monkeypatch.setenv(BACKEND_ENV_VAR, "local")
    assert get_backend() is PastelLocal
    # an explicit name still wins over the environment
    assert get_backend("gemini") is PastelGemini


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown Pastel backend"):
        get_backend("hal9000")


def test_local_accepts_its_own_questions() -> None:
    model = PastelLocal.from_feature_list(list(QUESTIONS) + ["is_claim_type_quantity"])
    assert model.get_questions() == list(QUESTIONS)
    assert len(model.get_functions()) == 1


def test_local_rejects_questions_it_has_no_model_for() -> None:
    with pytest.raises(ValueError, match="no fine-tuned model"):
        PastelLocal({BiasType.BIAS: 1.0, "Is this sentence about olive oil?": 1.0})


def test_local_rejects_unsupported_questions_when_copied() -> None:
    """The check has to survive the copy used throughout training, or an
    unanswerable question could sneak in that way."""
    model = PastelLocal.from_feature_list([QUESTIONS[0]])
    with pytest.raises(ValueError, match="no fine-tuned model"):
        model.create_copy_with_different_model(
            {BiasType.BIAS: 1.0, "Is this sentence about olive oil?": 1.0}
        )


def test_gemini_accepts_any_question() -> None:
    model = PastelGemini({BiasType.BIAS: 1.0, "Is this sentence about olive oil?": 1.0})
    assert model.get_questions() == ["Is this sentence about olive oil?"]


async def test_local_answers_questions_and_functions(monkeypatch) -> None:
    """The local backend answers each question with its own model and computes
    the functions itself."""
    from pastel import pastel_local
    from pastel.models import Sentence

    asked = []

    def fake_answer_question(question: str, sentences: list[str]) -> list[float]:
        asked.append(question)
        return [1.0] * len(sentences)

    monkeypatch.setattr(pastel_local, "answer_question", fake_answer_question)

    model = PastelLocal.from_feature_list(
        [QUESTIONS[0], QUESTIONS[1], "is_claim_type_quantity"]
    )
    sentences = [
        Sentence("A claim about something.", ("quantity",)),
        Sentence("Another claim.", ()),
    ]
    answers = await model.get_answers_to_questions(sentences)

    assert asked == [QUESTIONS[0], QUESTIONS[1]]
    assert set(answers) == set(sentences)
    for sentence in sentences:
        assert answers[sentence][QUESTIONS[0]] == 1.0
        assert answers[sentence][QUESTIONS[1]] == 1.0
    # the claim-type function is computed locally, per sentence
    quantity = model.get_functions()[0]
    assert answers[sentences[0]][quantity] == 1.0
    assert answers[sentences[1]][quantity] == 0.0


async def test_local_with_no_sentences() -> None:
    model = PastelLocal.from_feature_list([QUESTIONS[0]])
    assert await model.get_answers_to_questions([]) == {}


def test_cached_model_preserves_billing_labels() -> None:
    """The cache wraps a backend, so it must not lose the wrapped model's
    labels when it copies it during training."""
    from training.cached_pastel import CachedPastel
    from training.db_manager import DatabaseManager

    labels = {"team": "afc"}
    inner = PastelGemini({BiasType.BIAS: 1.0, "Any question?": 1.0}, labels=labels)
    cached = CachedPastel.from_pastel(inner, DatabaseManager(":memory:"))

    copied = cached.create_copy_with_different_model(
        {BiasType.BIAS: 1.0, "Another question?": 1.0}
    )
    assert isinstance(copied, CachedPastel)
    assert copied.inner.labels == labels


def test_local_dependency_error_is_actionable(monkeypatch) -> None:
    """Without the optional extra installed, the failure should say how to fix
    it rather than surfacing a bare ModuleNotFoundError from deep inside."""
    import builtins

    from pastel.local import local_answerer

    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    with pytest.raises(ImportError, match="uv sync --extra local"):
        local_answerer.answer_question(QUESTIONS[0], ["A sentence."])
