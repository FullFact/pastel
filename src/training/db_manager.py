"""
Database manager for storing and retrieving responses.
"""

import logging
import sqlite3
from contextlib import closing, contextmanager
from typing import Iterable, Iterator, List, Optional, Sequence

_logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "responses.db"

# SQLite's default limit on bound parameters is 999 on older builds, so read
# sentences in chunks rather than binding every sentence in one statement.
_CHUNK_SIZE = 400


class DatabaseManager:
    """A (question, sentence) -> response store backed by a local SQLite file.

    Reads and writes are batched: one statement per batch rather than one per
    (question, sentence) pair, which matters because a beam search over a few
    thousand sentences asks for tens of thousands of pairs at a time.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._create_table()
        # Deliberately no row count here: callers build one of these per
        # evaluation, and COUNT(*) over a large cache is not free.
        _logger.debug("Using response cache %s", db_path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a connection, commit on success, and always close it.
        sqlite3's own context manager commits but never closes."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                yield conn

    def _create_table(self) -> None:
        """Create the responses table if it doesn't exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS responses (
                    question TEXT NOT NULL,
                    sentence TEXT NOT NULL,
                    response REAL NOT NULL,
                    PRIMARY KEY (question, sentence)
                )
            """)

    def count_responses(self) -> int:
        """Total number of cached responses."""
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0])

    def clear_responses(self) -> None:
        """
        Delete all records from the responses table.
        This operation cannot be undone.
        """
        with self._connect() as conn:
            conn.execute("DELETE FROM responses")

    def delete_responses_for_question(self, question: str) -> int:
        """
        Delete all responses for a specific question from the database.

        Args:
            question: The question whose responses should be deleted

        Returns:
            Number of responses deleted

        Note:
            This operation cannot be undone.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM responses WHERE question = ?",
                (question,),
            )
            return cursor.rowcount

    def write_responses(self, responses: Iterable[tuple[str, str, float]]) -> int:
        """
        Write responses to the database in a single transaction.
        Existing (question, sentence) pairs are overwritten.

        Args:
            responses: Iterable of (question, sentence, response) triples,
                where each response is 0 (no), 1 (yes) or 0.5 (unsure)

        Returns:
            Number of rows written
        """
        rows = list(responses)
        if not rows:
            return 0

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO responses (question, sentence, response)
                VALUES (?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def write_response(self, question: str, sentence: str, response: float) -> None:
        """
        Write a single response to the database.
        If the (question, sentence) pair already exists, the response will be updated.

        Args:
            question: The question being asked
            sentence: The sentence being analyzed
            response: Float response value
        """
        _ = self.write_responses([(question, sentence, response)])

    def get_responses(
        self, questions: Sequence[str], sentences: Sequence[str]
    ) -> dict[tuple[str, str], float]:
        """
        Look up every cached response for the given questions and sentences.

        Args:
            questions: Questions to look up
            sentences: Sentence texts to look up

        Returns:
            Dict mapping (question, sentence) to its response. Pairs with no
            cached response are simply absent from the dict.
        """
        if not questions or not sentences:
            return {}

        found: dict[tuple[str, str], float] = {}
        question_slots = ",".join("?" * len(questions))

        with self._connect() as conn:
            for start in range(0, len(sentences), _CHUNK_SIZE):
                chunk = sentences[start : start + _CHUNK_SIZE]
                sentence_slots = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"""
                    SELECT question, sentence, response
                    FROM responses
                    WHERE question IN ({question_slots})
                      AND sentence IN ({sentence_slots})
                    """,
                    (*questions, *chunk),
                ).fetchall()
                for question, sentence, response in rows:
                    found[(question, sentence)] = float(response)

        return found

    def get_response(self, question: str, sentence: str) -> Optional[float]:
        """
        Retrieve a single response from the database.
        Returns None if no matching response is found.

        Args:
            question: The question to look up
            sentence: The sentence to look up

        Returns:
            Float response value if found, None otherwise
        """
        return self.get_responses([question], [sentence]).get((question, sentence))

    def get_unique_questions(self) -> List[str]:
        """
        Get a list of all unique questions in the responses table.

        Returns:
            List of questions, sorted alphabetically
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT question FROM responses ORDER BY question"
            ).fetchall()
        # Older versions of the cache wrote a row for the bias term; ignore those.
        return [row[0] for row in rows if row[0] not in ("bias", "BiasType.BIAS")]
