# Claude-created script
"""Use Gemini (via Pastel) to label sentences with yes/no answers to a fixed question list.

Output is a JSONL file with one record per sentence, each containing a `question_answers` dict.
The script is restart-safe: sentences already written to the output file are skipped.

Supports two input formats (auto-detected by file extension):
  - .jsonl  Pastel training format: {"sentence_text": ..., "score": ..., "claim_types": [...]}
  - .json   FullFact claims export:  [{"sentence": {"text": ..., "claim_type": [...],
                                        "checkworthiness": {"fullfact": {...}}}, ...}]

Outputs JSONL file, one sentence per line, e.g:
{"sentence_text": ..., "score": 5.0, "claim_types": ["quantity"], "question_answers": {"Is this making a claim that is too good to be true?": 1.0, ...}}

Usage:
    python -m local_models.label_sentences \\
        --question "Is this sentence a joke or satirical?" \\
        --input data/local_models/fullfact-2026-03-31-claims.json \\
        --output data/local_models/labelled_sentences.jsonl \\
        --batch-size 20
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from pastel.models import FEATURE_TYPE, Sentence
from pastel.optimise_weights import load_examples
from pastel.pastel import PastelModel
from pastel.pastel_gemini import PastelGemini

# One normalised input/output row: sentence text plus its metadata and,
# on output, the answers gathered so far.
ROW_TYPE = dict[str, Any]

DEFAULT_INPUT = Path("data/local_models/fullfact-2026-03-31-claims.json")
DEFAULT_OUTPUT = Path("data/local_models/labelled_sentences.jsonl")
DEFAULT_BATCH_SIZE = 20

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("google.ai.generativelanguage").setLevel(logging.WARNING)
# supress noisy messages "AFC is enabled with max remote calls: 10.":
logging.getLogger("google_genai.models").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def load_fullfact_claims(filename: str) -> list[ROW_TYPE]:
    """Load the Full Fact claims JSON export (a JSON array of article/sentence objects).

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


def load_input(input_path: Path) -> list[ROW_TYPE]:
    """Auto-detect format by extension and return a list of normalised row dicts."""
    if input_path.suffix.lower() == ".json":
        return load_fullfact_claims(str(input_path))
    # .jsonl is the Pastel training format, which is already normalised apart
    # from the score being a string.
    return [
        {
            "sentence_text": row["sentence_text"],
            "score": float(row["score"]) if row.get("score") is not None else None,
            "claim_types": row.get("claim_types", []),
        }
        for row in load_examples(str(input_path))
    ]


def build_pastel(questions: list[str]) -> PastelModel:
    """Create a Pastel with only the experiment questions (no functions, no bias beyond the auto-added one)."""
    return PastelGemini.from_feature_list(questions)


def load_already_labelled(output_path: Path, question: str) -> set[str]:
    """Return sentence_text values that already have an answer for this question."""
    already_done: set[str] = set()
    if not output_path.exists():
        return already_done
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    if question in record.get("question_answers", {}):
                        already_done.add(record["sentence_text"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return already_done


def format_output_record(
    original_row: ROW_TYPE,
    answers: dict[FEATURE_TYPE, float],
    questions: list[str],
) -> ROW_TYPE:
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
    pastel: PastelModel,
    batch_rows: list[ROW_TYPE],
    questions: list[str],
) -> list[ROW_TYPE]:
    """Call Gemini (via Pastel) to label a batch of rows, return formatted output records."""
    sentences = [
        Sentence(
            sentence_text=row["sentence_text"],
            claim_type=tuple(row.get("claim_types") or ()),
        )
        for row in batch_rows
    ]

    answers_by_sentence = await pastel.get_answers_to_questions(sentences)

    records = []
    for row, sentence in zip(batch_rows, sentences):
        if sentence not in answers_by_sentence:
            logger.warning("No answer returned for: %s", row["sentence_text"][:60])
            continue
        record = format_output_record(row, answers_by_sentence[sentence], questions)
        records.append(record)
    return records


def update_records(new_records: list[ROW_TYPE], output_path: Path) -> None:
    existing: dict[str, ROW_TYPE] = {}
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        existing[record["sentence_text"]] = record
                    except (json.JSONDecodeError, KeyError):
                        pass
    for record in new_records:
        text = record["sentence_text"]
        if text in existing:
            existing[text]["question_answers"].update(record["question_answers"])
        else:
            existing[text] = record
    with output_path.open("w", encoding="utf-8") as f:
        for record in existing.values():
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def main(
    question: str,
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
) -> None:
    """Label every sentence in `input_path` with an answer to `question`.

    `limit` caps how many sentences are sent, which is useful for a cheap
    smoke test before committing to labelling the whole file.
    """
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    rows = load_input(input_path)
    logger.info("Loaded %d sentences from %s", len(rows), input_path)

    already_done = load_already_labelled(output_path, question)
    if already_done:
        logger.info("Skipping %d already-labelled sentences", len(already_done))

    pending = [row for row in rows if row["sentence_text"] not in already_done]
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        logger.info("All sentences already labelled. Nothing to do.")
        return

    logger.info("%d sentences to label", len(pending))

    pastel = build_pastel([question])
    total_written = 0

    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        logger.info(
            "Batch %d/%d (%d sentences)...",
            i // batch_size + 1,
            (len(pending) + batch_size - 1) // batch_size,
            len(batch),
        )
        records = await label_batch(pastel, batch, [question])
        update_records(records, output_path)
        total_written += len(records)
        logger.info(
            "  Written %d records (total so far: %d)", len(records), total_written
        )
        # not sure if needed; might reduce rate limits/threading errors:
        await asyncio.sleep(1.0)

    logger.info("Done. %d sentences labelled -> %s", total_written, output_path)
    # Allow gRPC background threads (used by the Gemini client) to drain
    # before the event loop closes, avoiding spurious _DeleteDummyThreadOnDel warnings.
    await asyncio.sleep(1.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question", required=True, help="The single question to label with."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only label this many sentences, for a cheap smoke test.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    args = parse_args()
    asyncio.run(
        main(
            args.question,
            input_path=args.input,
            output_path=args.output,
            batch_size=args.batch_size,
            limit=args.limit,
        )
    )
