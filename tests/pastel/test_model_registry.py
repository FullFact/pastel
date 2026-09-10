"""Tests for the question -> model id mapping shared by training and inference.

Getting this wrong means a question is answered by another question's model,
which produces plausible-looking but wrong answers rather than an error, so
these check the mapping directly.
"""

import json
from pathlib import Path

import pytest

from pastel.local import model_registry

# mypy: ignore-errors

CATEGORY = "ModernBERT-multilingual"
Q1 = "Does this sentence relate to many people?"
Q2 = "Is this sentence a joke or satirical?"


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


def test_the_map_is_the_source_of_truth(models_dir: Path) -> None:
    write_map(models_dir, {Q1: "q07"})
    assert model_registry.model_id_for_question(Q1) == "q07"


def test_unmapped_question_raises(models_dir: Path) -> None:
    """Nothing is guessed from a question's position in any list, so a question
    the map doesn't know is an error rather than another question's model."""
    write_map(models_dir, {Q1: "q00"})
    with pytest.raises(ValueError, match="No fine-tuned model is recorded"):
        model_registry.model_id_for_question("Is this sentence about olive oil?")


def test_assign_is_stable_for_the_same_question(models_dir: Path) -> None:
    first = model_registry.assign_model_id(Q1)
    assert model_registry.assign_model_id(Q1) == first


def test_assign_starts_from_zero_and_records_the_id(models_dir: Path) -> None:
    assert model_registry.assign_model_id(Q1) == "q00"
    assert model_registry.load_model_map() == {Q1: "q00"}


def test_assign_does_not_collide_with_models_on_disk(models_dir: Path) -> None:
    """Even with no map, an existing model directory must not be overwritten."""
    (models_dir / CATEGORY / "q04").mkdir()
    assert model_registry.assign_model_id(Q1) == "q05"


def test_assign_does_not_collide_with_recorded_ids(models_dir: Path) -> None:
    write_map(models_dir, {Q2: "q12"})
    assert model_registry.assign_model_id(Q1) == "q13"


def test_models_dir_comes_from_the_environment(tmp_path: Path, monkeypatch) -> None:
    """Nothing should depend on the working directory being the repo root."""
    monkeypatch.delenv(model_registry.MODELS_DIR_ENV_VAR, raising=False)
    assert model_registry.models_dir() == model_registry.DEFAULT_MODELS_DIR

    monkeypatch.setenv(model_registry.MODELS_DIR_ENV_VAR, str(tmp_path))
    assert model_registry.models_dir() == tmp_path
    assert model_registry.model_map_path().is_relative_to(tmp_path)


def test_nothing_is_available_without_trained_models(models_dir: Path) -> None:
    """A recorded question whose model has not been trained is not available."""
    write_map(models_dir, {Q1: "q00"})
    assert model_registry.available_questions() == []
    assert model_registry.has_model(Q1) is False


def test_available_questions_reflects_what_is_on_disk(models_dir: Path) -> None:
    write_map(models_dir, {Q1: "q00", Q2: "q01"})
    add_trained_model(models_dir, "q00")

    assert model_registry.available_questions() == [Q1]


def test_available_questions_follows_the_recorded_map(models_dir: Path) -> None:
    """A question whose model was trained out of order is still found."""
    write_map(models_dir, {Q1: "q09"})
    add_trained_model(models_dir, "q09")

    assert model_registry.available_questions() == [Q1]


def test_latest_checkpoint_picks_the_newest(models_dir: Path) -> None:
    write_map(models_dir, {Q1: "q00"})
    add_trained_model(models_dir, "q00", step=50)
    newest = add_trained_model(models_dir, "q00", step=1000)
    # sorted numerically, not as strings - "1000" must beat "50"
    assert model_registry.latest_checkpoint(Q1) == newest


def test_latest_checkpoint_names_the_untrained_question(models_dir: Path) -> None:
    write_map(models_dir, {Q1: "q00"})
    with pytest.raises(FileNotFoundError) as excinfo:
        model_registry.latest_checkpoint(Q1)
    message = str(excinfo.value)
    assert Q1 in message
    assert model_registry.MODELS_DIR_ENV_VAR in message


def test_require_available_questions_explains_an_empty_directory(
    models_dir: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="No trained local models"):
        model_registry.require_available_questions()

    write_map(models_dir, {Q1: "q00"})
    add_trained_model(models_dir, "q00")
    assert model_registry.require_available_questions() == [Q1]
