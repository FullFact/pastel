"""Tests for choosing a backend and for the local backend's constraints."""

import json
from pathlib import Path

import pytest

from pastel import (
    BACKEND_ENV_VAR,
    DEFAULT_BACKEND,
    PastelGemini,
    PastelLocal,
    get_backend,
)
from pastel.local import model_registry
from pastel.models import BiasType

# mypy: ignore-errors

CATEGORY = "ModernBERT-multilingual"
QUESTIONS = [
    "Does this sentence relate to many people?",
    "Could believing this claim harm someone's health?",
]


@pytest.fixture
def trained_models(tmp_path: Path, monkeypatch) -> Path:
    """A models directory with a trained model whose heads answer QUESTIONS,
    which is what makes them answerable by the local backend."""
    monkeypatch.setenv(model_registry.MODELS_DIR_ENV_VAR, str(tmp_path))
    category = tmp_path / CATEGORY
    category.mkdir()
    mapping = {question: head for head, question in enumerate(QUESTIONS)}
    (category / model_registry.MODEL_MAP_FILENAME).write_text(
        json.dumps(mapping), encoding="utf-8"
    )
    (category / model_registry.MODEL_DIR_NAME / "checkpoint-100").mkdir(parents=True)
    return tmp_path


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


def test_local_accepts_trained_questions(trained_models: Path) -> None:
    model = PastelLocal.from_feature_list(list(QUESTIONS) + ["is_claim_type_quantity"])
    assert model.get_questions() == list(QUESTIONS)
    assert len(model.get_functions()) == 1


def test_local_rejects_questions_it_has_no_model_for(trained_models: Path) -> None:
    with pytest.raises(ValueError, match="no fine-tuned model"):
        PastelLocal({BiasType.BIAS: 1.0, "Is this sentence about olive oil?": 1.0})


def test_local_rejects_unsupported_questions_when_copied(trained_models: Path) -> None:
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


async def test_local_answers_questions_and_functions(
    monkeypatch, trained_models: Path
) -> None:
    """The local backend answers every question in one pass of the shared
    encoder and computes the functions itself."""
    from pastel import pastel_local
    from pastel.models import Sentence

    asked = []

    def fake_answer_questions(
        questions: list[str], sentences: list[str]
    ) -> dict[str, list[float]]:
        asked.append(questions)
        return {question: [1.0] * len(sentences) for question in questions}

    monkeypatch.setattr(pastel_local, "answer_questions", fake_answer_questions)

    model = PastelLocal.from_feature_list(
        [QUESTIONS[0], QUESTIONS[1], "is_claim_type_quantity"]
    )
    sentences = [
        Sentence("A claim about something.", ("quantity",)),
        Sentence("Another claim.", ()),
    ]
    answers = await model.get_answers_to_questions(sentences)

    # one call, not one per question
    assert asked == [[QUESTIONS[0], QUESTIONS[1]]]
    assert set(answers) == set(sentences)
    for sentence in sentences:
        assert answers[sentence][QUESTIONS[0]] == 1.0
        assert answers[sentence][QUESTIONS[1]] == 1.0
    # the claim-type function is computed locally, per sentence
    quantity = model.get_functions()[0]
    assert answers[sentences[0]][quantity] == 1.0
    assert answers[sentences[1]][quantity] == 0.0


async def test_local_with_no_sentences(trained_models: Path) -> None:
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
        local_answerer.answer_questions(QUESTIONS, ["A sentence."])
