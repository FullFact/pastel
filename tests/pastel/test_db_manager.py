from pathlib import Path

import pytest

from training.db_manager import DatabaseManager


@pytest.fixture
def db(tmp_path: Path) -> DatabaseManager:
    return DatabaseManager(str(tmp_path / "test_responses.db"))


def test_write_and_read_one(db: DatabaseManager) -> None:
    assert db.get_response("q1", "s1") is None
    db.write_response("q1", "s1", 0.5)
    assert db.get_response("q1", "s1") == 0.5


def test_write_overwrites_existing_pair(db: DatabaseManager) -> None:
    db.write_response("q1", "s1", 0.0)
    db.write_response("q1", "s1", 1.0)
    assert db.get_response("q1", "s1") == 1.0
    assert db.count_responses() == 1


def test_bulk_read_returns_only_what_is_cached(db: DatabaseManager) -> None:
    assert (
        db.write_responses([("q1", "s1", 1.0), ("q1", "s2", 0.0), ("q2", "s2", 0.5)])
        == 3
    )

    found = db.get_responses(["q1", "q2"], ["s1", "s2", "s3"])
    assert found == {
        ("q1", "s1"): 1.0,
        ("q1", "s2"): 0.0,
        ("q2", "s2"): 0.5,
    }
    # Missing pairs are simply absent rather than present-and-None
    assert ("q2", "s1") not in found
    assert ("q1", "s3") not in found


def test_bulk_read_with_empty_inputs(db: DatabaseManager) -> None:
    db.write_response("q1", "s1", 1.0)
    assert db.get_responses([], ["s1"]) == {}
    assert db.get_responses(["q1"], []) == {}


def test_bulk_write_with_no_rows(db: DatabaseManager) -> None:
    assert db.write_responses([]) == 0
    assert db.count_responses() == 0


def test_bulk_read_chunks_large_sentence_lists(db: DatabaseManager) -> None:
    """More sentences than the chunk size must still all come back."""
    sentences = [f"sentence {i}" for i in range(1000)]
    db.write_responses([("q1", s, 1.0) for s in sentences])
    found = db.get_responses(["q1"], sentences)
    assert len(found) == 1000


def test_get_unique_questions_ignores_legacy_bias_rows(db: DatabaseManager) -> None:
    db.write_responses(
        [
            ("q2", "s1", 1.0),
            ("q1", "s1", 1.0),
            ("bias", "s1", 1.0),
            ("BiasType.BIAS", "s1", 1.0),
        ]
    )
    assert db.get_unique_questions() == ["q1", "q2"]


def test_delete_responses_for_question(db: DatabaseManager) -> None:
    db.write_responses([("q1", "s1", 1.0), ("q1", "s2", 1.0), ("q2", "s1", 1.0)])
    assert db.delete_responses_for_question("q1") == 2
    assert db.get_unique_questions() == ["q2"]


def test_clear_responses(db: DatabaseManager) -> None:
    db.write_responses([("q1", "s1", 1.0), ("q2", "s1", 1.0)])
    db.clear_responses()
    assert db.count_responses() == 0


def test_separate_paths_are_separate_databases(tmp_path: Path) -> None:
    """The old singleton silently ignored db_path after the first call."""
    first = DatabaseManager(str(tmp_path / "one.db"))
    second = DatabaseManager(str(tmp_path / "two.db"))
    first.write_response("q1", "s1", 1.0)

    assert second.db_path.endswith("two.db")
    assert second.get_response("q1", "s1") is None
    assert first.get_response("q1", "s1") == 1.0
