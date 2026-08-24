"""Tests for the question -> model id mapping shared by training and inference.

Getting this wrong means a question is answered by another question's model,
which produces plausible-looking but wrong answers rather than an error, so
these check the mapping directly.
"""

import json
from pathlib import Path

import pytest

from pastel.local import model_registry
from pastel.local.questions import QUESTIONS

# mypy: ignore-errors

CATEGORY = "ModernBERT-multilingual"


@pytest.fixture
def models_dir(tmp_path: Path, monkeypatch) -> Path:
    """Point the registry at an empty temporary models directory, the way a
    deployment would."""
    monkeypatch.setenv(model_registry.MODELS_DIR_ENV_VAR, str(tmp_path))
    (tmp_path / CATEGORY).mkdir()
    return tmp_path


def add_trained_model(models_dir: Path, model_id: str, step: int = 100) -> Path:
    """Fake a trained model by creating one of its checkpoint directories."""
    checkpoint = models_dir / CATEGORY / model_id / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    return checkpoint


def write_map(models_dir: Path, mapping: dict) -> None:
    (models_dir / CATEGORY / model_registry.MODEL_MAP_FILENAME).write_text(
        json.dumps(mapping), encoding="utf-8"
    )


def test_recorded_mapping_wins_over_the_questions_index(models_dir: Path) -> None:
    """The whole point of the map: a question's model id does not have to match
    its position in QUESTIONS."""
    write_map(models_dir, {QUESTIONS[0]: "q07"})
    assert model_registry.model_id_for_question(QUESTIONS[0]) == "q07"


def test_falls_back_to_the_questions_index(models_dir: Path) -> None:
    """Models trained before the map existed are named by QUESTIONS index."""
    assert model_registry.model_id_for_question(QUESTIONS[3]) == "q03"


def test_unknown_question_raises(models_dir: Path) -> None:
    with pytest.raises(ValueError, match="No fine-tuned model is recorded"):
        model_registry.model_id_for_question("Is this sentence about olive oil?")


def test_assign_keeps_declared_questions_on_their_index(models_dir: Path) -> None:
    assert model_registry.assign_model_id(QUESTIONS[2]) == "q02"
    # and records it, so the next lookup doesn't rely on the fallback
    assert model_registry.load_model_map() == {QUESTIONS[2]: "q02"}


def test_assign_is_stable_for_the_same_question(models_dir: Path) -> None:
    first = model_registry.assign_model_id("A brand new question?")
    assert model_registry.assign_model_id("A brand new question?") == first


def test_assign_does_not_collide_with_the_questions_list(models_dir: Path) -> None:
    """A new question must land past every index QUESTIONS already claims,
    otherwise training it would overwrite an existing model."""
    new_id = model_registry.assign_model_id("A brand new question?")
    assert new_id == f"q{len(QUESTIONS):02d}"


def test_assign_does_not_collide_with_models_on_disk(models_dir: Path) -> None:
    """Even with no map, an existing model directory must not be overwritten."""
    high = len(QUESTIONS) + 4
    (models_dir / CATEGORY / f"q{high:02d}").mkdir()
    assert model_registry.assign_model_id("A brand new question?") == f"q{high + 1:02d}"


def test_assign_does_not_collide_with_recorded_ids(models_dir: Path) -> None:
    high = len(QUESTIONS) + 2
    write_map(models_dir, {"An older new question?": f"q{high:02d}"})
    assert model_registry.assign_model_id("A brand new question?") == f"q{high + 1:02d}"


def test_models_dir_comes_from_the_environment(tmp_path: Path, monkeypatch) -> None:
    """Nothing should depend on the working directory being the repo root."""
    monkeypatch.delenv(model_registry.MODELS_DIR_ENV_VAR, raising=False)
    assert model_registry.models_dir() == model_registry.DEFAULT_MODELS_DIR

    monkeypatch.setenv(model_registry.MODELS_DIR_ENV_VAR, str(tmp_path))
    assert model_registry.models_dir() == tmp_path
    assert model_registry.model_map_path().is_relative_to(tmp_path)


def test_nothing_is_available_without_trained_models(models_dir: Path) -> None:
    """QUESTIONS is a declaration; an empty models directory means none of it
    can actually be answered."""
    assert model_registry.available_questions() == []
    assert model_registry.missing_questions() == list(QUESTIONS)
    assert model_registry.has_model(QUESTIONS[0]) is False


def test_available_questions_reflects_what_is_on_disk(models_dir: Path) -> None:
    add_trained_model(models_dir, "q00")
    add_trained_model(models_dir, "q02")

    assert model_registry.available_questions() == [QUESTIONS[0], QUESTIONS[2]]
    assert QUESTIONS[1] in model_registry.missing_questions()


def test_available_questions_follows_the_recorded_map(models_dir: Path) -> None:
    """A question whose model was trained out of order is still found - and the
    question whose index that id collides with is not falsely claimed."""
    write_map(models_dir, {QUESTIONS[1]: "q09"})
    add_trained_model(models_dir, "q09")

    assert model_registry.available_questions() == [QUESTIONS[1]]


def test_index_fallback_refuses_an_id_another_question_owns(
    models_dir: Path,
) -> None:
    """QUESTIONS[9] would fall back to q09, but the map gives q09 to another
    question - answering with it would be silently wrong."""
    write_map(models_dir, {QUESTIONS[1]: "q09"})
    add_trained_model(models_dir, "q09")

    with pytest.raises(ValueError, match="recorded as the model for"):
        model_registry.model_id_for_question(QUESTIONS[9])
    assert model_registry.has_model(QUESTIONS[9]) is False


def test_latest_checkpoint_picks_the_newest(models_dir: Path) -> None:
    add_trained_model(models_dir, "q00", step=50)
    newest = add_trained_model(models_dir, "q00", step=1000)
    # sorted numerically, not as strings - "1000" must beat "50"
    assert model_registry.latest_checkpoint(QUESTIONS[0]) == newest


def test_latest_checkpoint_names_the_untrained_question(models_dir: Path) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        model_registry.latest_checkpoint(QUESTIONS[0])
    message = str(excinfo.value)
    assert QUESTIONS[0] in message
    assert model_registry.MODELS_DIR_ENV_VAR in message


def test_trained_questions_includes_undeclared_ones(models_dir: Path) -> None:
    """A model trained for a question that was never added to QUESTIONS is
    findable, so the drift can be reported."""
    write_map(models_dir, {"A question nobody declared?": "q42"})
    add_trained_model(models_dir, "q42")

    assert model_registry.trained_questions() == ["A question nobody declared?"]
    assert model_registry.available_questions() == []


def test_require_available_questions_explains_an_empty_directory(
    models_dir: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="No trained local models"):
        model_registry.require_available_questions()

    add_trained_model(models_dir, "q00")
    assert model_registry.require_available_questions() == [QUESTIONS[0]]
