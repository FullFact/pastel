"""Tests for the question -> head mapping shared by training and inference.

Getting this wrong means a question is answered by another question's head,
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


def add_checkpoint(models_dir: Path, step: int = 100) -> Path:
    """Fake a trained model by creating one of its checkpoint directories."""
    checkpoint = (
        models_dir / CATEGORY / model_registry.MODEL_DIR_NAME / f"checkpoint-{step}"
    )
    checkpoint.mkdir(parents=True)
    return checkpoint


def write_map(models_dir: Path, mapping: dict) -> None:
    (models_dir / CATEGORY / model_registry.MODEL_MAP_FILENAME).write_text(
        json.dumps(mapping), encoding="utf-8"
    )


def test_the_map_is_the_source_of_truth(models_dir: Path) -> None:
    write_map(models_dir, {Q1: 7})
    assert model_registry.head_for_question(Q1) == 7


def test_unmapped_question_raises(models_dir: Path) -> None:
    """Nothing is guessed from a question's position in any list, so a question
    the map doesn't know is an error rather than another question's head."""
    write_map(models_dir, {Q1: 0})
    with pytest.raises(ValueError, match="No fine-tuned model is recorded"):
        model_registry.head_for_question("Is this sentence about olive oil?")


def test_a_map_of_per_question_model_ids_is_rejected(models_dir: Path) -> None:
    """Maps written when each question had its own model recorded an id like
    "q00". Reading one as a head index would answer with head 0 for every
    question, so it has to fail instead."""
    write_map(models_dir, {Q1: "q00", Q2: "q01"})
    with pytest.raises(ValueError, match="retrained"):
        model_registry.head_for_question(Q1)


def test_assign_is_stable_for_the_same_question(models_dir: Path) -> None:
    first = model_registry.assign_head(Q1)
    assert model_registry.assign_head(Q1) == first


def test_assign_starts_from_zero_and_records_the_head(models_dir: Path) -> None:
    assert model_registry.assign_head(Q1) == 0
    assert model_registry.assign_head(Q2) == 1
    assert model_registry.load_model_map() == {Q1: 0, Q2: 1}


def test_assign_does_not_collide_with_recorded_heads(models_dir: Path) -> None:
    write_map(models_dir, {Q2: 4})
    assert model_registry.assign_head(Q1) == 5


def test_models_dir_comes_from_the_environment(tmp_path: Path, monkeypatch) -> None:
    """Nothing should depend on the working directory being the repo root."""
    monkeypatch.delenv(model_registry.MODELS_DIR_ENV_VAR, raising=False)
    assert model_registry.models_dir() == model_registry.DEFAULT_MODELS_DIR

    monkeypatch.setenv(model_registry.MODELS_DIR_ENV_VAR, str(tmp_path))
    assert model_registry.models_dir() == tmp_path
    assert model_registry.model_map_path().is_relative_to(tmp_path)
    assert model_registry.model_dir().is_relative_to(tmp_path)


def test_nothing_is_available_without_a_trained_model(models_dir: Path) -> None:
    """Recorded questions whose model has not been trained are not available."""
    write_map(models_dir, {Q1: 0})
    assert model_registry.available_questions() == []
    assert model_registry.has_model(Q1) is False


def test_every_recorded_question_is_available_once_trained(models_dir: Path) -> None:
    """One model answers all of them, so they arrive together."""
    write_map(models_dir, {Q1: 0, Q2: 1})
    add_checkpoint(models_dir)

    assert model_registry.available_questions() == [Q1, Q2]
    assert model_registry.has_model(Q1) is True


def test_a_question_the_map_does_not_record_is_not_available(models_dir: Path) -> None:
    write_map(models_dir, {Q1: 0})
    add_checkpoint(models_dir)

    assert model_registry.has_model(Q2) is False


def test_head_count_covers_every_recorded_question(models_dir: Path) -> None:
    write_map(models_dir, {Q1: 0, Q2: 3})
    assert model_registry.head_count() == 4


def test_latest_checkpoint_picks_the_newest(models_dir: Path) -> None:
    add_checkpoint(models_dir, step=50)
    newest = add_checkpoint(models_dir, step=1000)
    # sorted numerically, not as strings - "1000" must beat "50"
    assert model_registry.latest_checkpoint() == newest


def test_latest_checkpoint_says_where_it_looked(models_dir: Path) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        model_registry.latest_checkpoint()
    message = str(excinfo.value)
    assert str(model_registry.model_dir()) in message
    assert model_registry.MODELS_DIR_ENV_VAR in message


def test_require_available_questions_explains_an_empty_directory(
    models_dir: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="No trained local model"):
        model_registry.require_available_questions()

    write_map(models_dir, {Q1: 0})
    add_checkpoint(models_dir)
    assert model_registry.require_available_questions() == [Q1]
