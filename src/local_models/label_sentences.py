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
    python scripts/encoder_experiment/label_sentences.py \\
        --input /path/to/fullfact-2026-03-16-claims.json \\
        --output scripts/encoder_experiment/labelled_sentences.jsonl \\
        --batch-size 20
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

from local_models.questions import QUESTIONS
from pastel.models import BiasType, Sentence
from pastel.optimise_weights import load_examples
from pastel.pastel import PastelModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("google.ai.generativelanguage").setLevel(logging.WARNING)
# supress noisy messages "AFC is enabled with max remote calls: 10.":
logging.getLogger("google_genai.models").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def load_fullfact_claims(filename: str) -> list[dict]:
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


def load_input(input_path: Path) -> list[dict]:
    """Auto-detect format by extension and return a list of normalised row dicts."""
    # if input_path.suffix.lower() == ".json":
    return load_fullfact_claims(str(input_path))
    # return load_examples(str(input_path))


def build_pastel(questions: list[str]) -> PastelModel:
    """Create a Pastel with only the experiment questions (no functions, no bias beyond the auto-added one)."""
    return PastelModel.from_feature_list(questions)


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
    pastel: PastelModel,
    batch_rows: list[dict],
    questions: list[str],
) -> list[dict]:
    """Call Gemini (via Pastel) to label a batch of rows, return formatted output records."""
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
        record = format_output_record(row, answers_by_sentence[sentence], questions)
        records.append(record)
    return records


def update_records(new_records: list[dict], output_path: Path) -> None:
    existing: dict[str, dict] = {}
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


async def main(question: str) -> None:
    input_path = Path("data/local_models/fullfact-2026-03-31-claims.jsonl")
    output_path = Path("data/local_models/labelled_sentences.jsonl")
    batch_size = 20

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    rows = load_input(input_path)
    logger.info("Loaded %d sentences from %s", len(rows), input_path)

    already_done = load_already_labelled(output_path, question)
    if already_done:
        logger.info("Skipping %d already-labelled sentences", len(already_done))

    pending = [row for row in rows if row["sentence_text"] not in already_done]
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
        if i >= 200:
            break

    logger.info("Done. %d sentences labelled -> %s", total_written, output_path)
    # Allow gRPC background threads (used by the Gemini client) to drain
    # before the event loop closes, avoiding spurious _DeleteDummyThreadOnDel warnings.
    await asyncio.sleep(1.5)


if __name__ == "__main__":
    new_question = "Is this sentence a joke or satirical?"

    asyncio.run(main(new_question))
