# Claude-created
"""Script 1: Use Gemini (via Pastel) to label sentences with yes/no answers to a fixed question list.

Output is a JSONL file with one record per sentence, each containing a `question_answers` dict.
The script is restart-safe: sentences already written to the output file are skipped.

Supports two input formats (auto-detected by file extension):
  - .jsonl  pastel training format: {"sentence_text": ..., "score": ..., "claim_types": [...]}
  - .json   FullFact claims export:  [{"sentence": {"text": ..., "claim_type": [...],
                                        "checkworthiness": {"fullfact": {...}}}, ...}]

Usage:
    python scripts/encoder_experiment/label_sentences.py \\
        --input /path/to/fullfact-2026-03-16-claims.json \\
        --output scripts/encoder_experiment/labelled_sentences.jsonl \\
        --batch-size 20
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from pastel.models import BiasType, Sentence
from pastel.optimise_weights import load_examples
from pastel.pastel import Pastel

from questions import QUESTIONS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("google.ai.generativelanguage").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def load_fullfact_claims(filename: str) -> list[dict]:
    """Load the FullFact claims JSON export (a JSON array of article/sentence objects).

    Maps to the same internal format as load_examples():
      {"sentence_text": ..., "score": ..., "claim_types": [...]}

    The score is the max of the fullfact checkworthiness values (or None if absent).
    This applies if a claim matches more than one topic, though the scores should be
    the same anyway.
    """
    with open(filename, "rt", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for item in data:
        sentence = item.get("sentence", {})
        text = sentence.get("text", "").strip()
        if not text:
            continue
        claim_types = sentence.get("claim_type", [])
        ff_scores = sentence.get("checkworthiness", {}).get("fullfact", {})
        score = max(ff_scores.values()) if ff_scores else None
        rows.append(
            {
                "sentence_text": text,
                "score": score,
                "claim_types": claim_types,
            }
        )
    return rows


def load_input(input_path: Path) -> list[dict]:
    """Auto-detect format by extension and return a list of normalised row dicts."""
    # if input_path.suffix.lower() == ".json":
    return load_fullfact_claims(str(input_path))
    # return load_examples(str(input_path))


def build_pastel(questions: list[str]) -> Pastel:
    """Create a Pastel with only the experiment questions (no functions, no bias beyond the auto-added one)."""
    return Pastel.from_feature_list(questions)


def load_already_labelled(output_path: Path) -> set[str]:
    """Return the set of sentence_text values already present in the output file."""
    already_done: set[str] = set()
    if not output_path.exists():
        return already_done
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    already_done.add(record["sentence_text"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return already_done


def format_output_record(
    original_row: dict,
    answers: dict,
    questions: list[str],
) -> dict:
    """Merge original JSONL fields with question answers, keeping only string-keyed answers."""
    question_answers = {
        k: v for k, v in answers.items() if isinstance(k, str) and k in questions
    }
    return {
        "sentence_text": original_row["sentence_text"],
        "score": original_row.get("score"),
        "claim_types": original_row.get("claim_types", []),
        "question_answers": question_answers,
    }


async def label_batch(
    pastel: Pastel,
    batch_rows: list[dict],
) -> list[dict]:
    """Call Pastel for a batch of rows, return formatted output records."""
    sentences = [
        Sentence(
            sentence_text=row["sentence_text"],
            claim_type=tuple(row["claim_types"]) if row.get("claim_types") else None,
        )
        for row in batch_rows
    ]

    answers_by_sentence = await pastel.get_answers_to_questions(sentences)

    records = []
    for row, sentence in zip(batch_rows, sentences):
        if sentence not in answers_by_sentence:
            logger.warning("No answer returned for: %s", row["sentence_text"][:60])
            continue
        record = format_output_record(row, answers_by_sentence[sentence], QUESTIONS)
        records.append(record)
    return records


def append_records(records: list[dict], output_path: Path) -> None:
    with output_path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="/Users/davidcorney/fullfact/data-science-scripts/data/elastic_searcher/fullfact-2026-03-31-claims.jsonl",
        help="Path to input file (.json FullFact claims export or .jsonl pastel training format)",
    )
    parser.add_argument(
        "--output",
        default="scripts/encoder_experiment/labelled_sentences.jsonl",
        help="Path for labelled output JSONL",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of sentences per Pastel call (default: 20)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    rows = load_input(input_path)
    logger.info("Loaded %d sentences from %s", len(rows), input_path)

    already_done = load_already_labelled(output_path)
    if already_done:
        logger.info("Skipping %d already-labelled sentences", len(already_done))

    pending = [row for row in rows if row["sentence_text"] not in already_done]
    if not pending:
        logger.info("All sentences already labelled. Nothing to do.")
        return

    logger.info(
        "%d sentences to label with %d questions each", len(pending), len(QUESTIONS)
    )

    pastel = build_pastel(QUESTIONS)
    batch_size = args.batch_size
    total_written = 0

    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        logger.info(
            "Batch %d/%d (%d sentences)...",
            i // batch_size + 1,
            (len(pending) + batch_size - 1) // batch_size,
            len(batch),
        )
        records = await label_batch(pastel, batch)
        append_records(records, output_path)
        total_written += len(records)
        logger.info(
            "  Written %d records (total so far: %d)", len(records), total_written
        )
        if i >= 5:
            break

    logger.info("Done. %d sentences labelled -> %s", total_written, output_path)
    # Allow gRPC background threads (used by the Gemini client) to drain
    # before the event loop closes, avoiding spurious _DeleteDummyThreadOnDel warnings.
    await asyncio.sleep(1.5)


if __name__ == "__main__":
    asyncio.run(main())
