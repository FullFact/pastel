"""
Database manager for storing and retrieving responses.
"""

import sqlite3
from typing import List, Optional


class DatabaseManager:
    _instance = None
    db_path: str

    def __new__(cls, db_path: str = "responses.db") -> "DatabaseManager":
        """Create a singleton instance of DatabaseManager."""
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance.db_path = db_path
            cls._instance._create_table()

            # Print the number of rows in the responses table
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM responses")
                row_count = cursor.fetchone()[0]
                print(f"Database initialised; contains {row_count:,} cached responses")

        return cls._instance

    def __init__(self, db_path: str = "responses.db"):
        """Initialize is called after __new__, but we've already set up in __new__."""
        pass

    def _create_table(self) -> None:
        """Create the responses table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS responses (
                    question TEXT NOT NULL,
                    sentence TEXT NOT NULL,
                    response REAL NOT NULL,
                    PRIMARY KEY (question, sentence)
                )
            """
            )
            conn.commit()

    def clear_responses(self) -> None:
        """
        Delete all records from the responses table.
        This operation cannot be undone.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM responses")
            conn.commit()

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
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM responses
                WHERE question = ?
                """,
                (question,),
            )
            deleted_count = cursor.rowcount
            conn.commit()
            return deleted_count

    def write_response(self, question: str, sentence: str, response: float) -> None:
        """
        Write a response to the database.
        If the (question, sentence) pair already exists, the response will be updated.

        Args:
            question: The question being asked
            sentence: The sentence being analyzed
            response: Float response value
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO responses (question, sentence, response)
                VALUES (?, ?, ?)
            """,
                (question, sentence, response),
            )
            conn.commit()

    def get_response(self, question: str, sentence: str) -> Optional[float]:
        """
        Retrieve a response from the database based on the question and sentence.
        Returns None if no matching response is found.

        Args:
            question: The question to look up
            sentence: The sentence to look up

        Returns:
            Float response value if found, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT response
                FROM responses
                WHERE question = ? AND sentence = ?
            """,
                (question, sentence),
            )
            result = cursor.fetchone()
            return float(result[0]) if result else None

    def get_unique_questions(self) -> List[str]:
        """
        Get a list of all unique questions in the responses table.

        Returns:
            List of questions, sorted alphabetically
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT question FROM responses ORDER BY question")
            return [
                row[0]
                for row in cursor.fetchall()
                if row[0] not in ["bias", "BiasType.BIAS"]
            ]
