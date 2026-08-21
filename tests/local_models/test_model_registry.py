"""Tests for the question -> model id mapping shared by training and inference.

Getting this wrong means a question is answered by another question's model,
which produces plausible-looking but wrong answers rather than an error, so
these check the mapping directly.
"""

import json
from pathlib import Path

import pytest

from local_models import model_registry
from local_models.questions import QUESTIONS

# mypy: ignore-errors

CATEGORY = "ModernBERT-multilingual"


@pytest.fixture
def models_dir(tmp_path: Path, monkeypatch) -> Path:
    """Point the registry at an empty temporary models directory."""
    monkeypatch.setattr(model_registry, "MODELS_DIR", tmp_path)
    (tmp_path / CATEGORY).mkdir()
    return tmp_path


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
