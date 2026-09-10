# Claude-created
"""Fine-tune one encoder to answer every Pastel question.

The encoder body is shared and each question gets its own binary
classification head, so the questions are trained together rather than one
model at a time: a sentence labelled for only some of the questions trains
those heads and leaves the others alone. That is what makes inference a single
forward pass per sentence instead of one per question.

Because the body is shared, a retrain replaces the whole model - adding a
question means training all of them again.

The input is the bare sentence: each head answers one fixed question, so
prefixing the question text would only spend the forward pass encoding a
constant. Inference does the same.

Dependencies for local fine-tuning:
    uv sync --group ml-labeller
or
    pip install "transformers>=4.40" datasets torch accelerate scikit-learn

"""

import csv
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets as hf_datasets  # type: ignore
import numpy as np
from transformers import AutoTokenizer

from pastel.local.model_registry import (
    MODEL_CATEGORY,
    MODELS,
    assign_head,
    load_model_map,
    model_dir,
    model_map_path,
    models_dir,
)
from pastel.local.multi_head_encoder import IGNORE_LABEL, MultiHeadEncoder

logger = logging.getLogger(__name__)

LABELLED_DATA_PATH = Path("data/local_models/labelled_sentences.jsonl")

# The answer Gemini gives when it is unsure. Those sentences teach the head
# nothing, so they are left unlabelled rather than rounded one way or another.
UNSURE_ANSWER = 0.5

RANDOM_SEED = 42
MAX_LENGTH = 128  # 128 is enough for c.95% of sentences; 256 would cover them all
TEST_FRACTION = 0.2


def setup_logging(output_dir: Path) -> None:
    log_path = output_dir / "finetune.log"
    fmt = "%(asctime)s %(levelname)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger.info("Logging to %s", log_path)


@dataclass
class MultiQuestionDataset:
    """Every question's labelled data at once: one row per sentence, holding
    one label per head."""

    questions: list[str]  # in head order: questions[i] is answered by head i
    sentences: list[str]
    labels: list[list[int]]  # IGNORE_LABEL where a sentence has no answer


@dataclass
class ModelResult:
    model_name: str
    question: str
    head: int
    n_train: int
    n_test: int
    accuracy: float
    f1_binary: float
    f1_macro: float
    precision: float
    recall: float
    train_seconds: float  # for the whole model: the questions train together


def load_labelled_data(input_path: Path) -> list[dict[str, Any]]:
    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d labelled records from %s", len(records), input_path)
    return records


def questions_in(records: list[dict[str, Any]]) -> list[str]:
    """Every question the labelled data has answers for, first seen first."""
    questions: dict[str, None] = {}
    for record in records:
        questions.update(dict.fromkeys(record.get("question_answers", {})))
    return list(questions)


def questions_in_head_order(questions: list[str]) -> list[str]:
    """Allocate a head to each question, and return them in head order - which
    is the order the model's heads will be in.

    Every question the map already records has to be trained too. The body is
    shared, so a retrain replaces the previous model outright: a question left
    out would keep a head index that the new model has either not trained, or
    trained for something else.
    """
    heads = {question: assign_head(question) for question in questions}

    untrained = [question for question in load_model_map() if question not in heads]
    if untrained:
        raise ValueError(
            "One model answers every question, so a retrain has to cover all "
            "of them. These are recorded in the model map but were not passed "
            "in: " + "; ".join(untrained) + f". Either include them, or remove "
            f"them from {model_map_path()} to give up answering them."
        )

    in_head_order = sorted(questions, key=lambda question: heads[question])
    if [heads[question] for question in in_head_order] != list(range(len(heads))):
        raise ValueError(
            f"The head indices in {model_map_path()} have gaps or duplicates "
            f"({heads}), so they cannot be a model's heads. Correct the file, "
            "or delete it to allocate them again from scratch."
        )
    return in_head_order


def build_dataset(
    records: list[dict[str, Any]], questions: list[str]
) -> MultiQuestionDataset:
    """Reformat the labelled records into one row per sentence, with a label
    for every question that sentence has an answer for."""

    def label(answers: dict[str, float], question: str) -> int:
        answer = answers.get(question)
        if answer is None or answer == UNSURE_ANSWER:
            return IGNORE_LABEL
        return int(answer)

    sentences, labels = [], []
    for record in records:
        answers = record.get("question_answers", {})
        row = [label(answers, question) for question in questions]
        if all(value == IGNORE_LABEL for value in row):
            continue
        sentences.append(record["sentence_text"])
        labels.append(row)

    label_array = np.array(labels)
    for head, question in enumerate(questions):
        answered = label_array[:, head] != IGNORE_LABEL
        logger.info(
            "head %02d: %d labelled, %d yes  %s",
            head,
            int(answered.sum()),
            int((label_array[:, head] == 1).sum()),
            question[:60],
        )

    return MultiQuestionDataset(questions=questions, sentences=sentences, labels=labels)


def tokenise_dataset(
    dataset: MultiQuestionDataset, tokenizer: Any
) -> hf_datasets.Dataset:
    tokenised = tokenizer(
        dataset.sentences,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    data = {k: v for k, v in tokenised.items()}
    data["labels"] = dataset.labels
    return hf_datasets.Dataset.from_dict(data)


def binary_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (  # type: ignore
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    if len(labels) == 0:
        return dict.fromkeys(
            ("accuracy", "f1_binary", "f1_macro", "precision", "recall"), float("nan")
        )
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_binary": float(f1_score(labels, preds, average="binary", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "precision": float(
            precision_score(labels, preds, average="binary", zero_division=0)
        ),
        "recall": float(recall_score(labels, preds, average="binary", zero_division=0)),
    }


def make_compute_metrics(questions: list[str]) -> Any:
    """Metrics per question, since one number over all the heads at once would
    hide a head that has stopped working. Each head is only scored on the
    sentences that have an answer for its question."""

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, label_ids = eval_pred
        preds = np.argmax(logits, axis=-1)  # (sentences, heads)

        metrics: dict[str, float] = {}
        for head in range(len(questions)):
            answered = label_ids[:, head] != IGNORE_LABEL
            for name, value in binary_metrics(
                label_ids[answered, head], preds[answered, head]
            ).items():
                metrics[f"q{head:02d}_{name}"] = value

        # one number to compare retrains by
        metrics["mean_f1_binary"] = float(
            np.nanmean([metrics[f"q{h:02d}_f1_binary"] for h in range(len(questions))])
        )
        return metrics

    return compute_metrics


def finetune_multi_head(
    base_model_id: str,
    questions: list[str],
    train_ds: hf_datasets.Dataset,
    test_ds: hf_datasets.Dataset,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    save_checkpoints: bool,
) -> tuple[dict[str, float], float]:
    """Train the shared encoder and every head together."""
    from transformers import Trainer, TrainingArguments

    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch" if save_checkpoints else "no",
        seed=RANDOM_SEED,
        report_to="none",
        load_best_model_at_end=False,
        logging_steps=50,
    )

    model = MultiHeadEncoder.from_base_model(base_model_id, len(questions))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=make_compute_metrics(questions),
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    metrics = trainer.evaluate()
    # strip the "eval_" prefix added by Trainer
    clean = {k.replace("eval_", ""): v for k, v in metrics.items()}
    return clean, elapsed


def auto_detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def labelled_counts(dataset: hf_datasets.Dataset, head: int) -> int:
    """How many rows of a split have an answer for one question."""
    labels = np.array(dataset["labels"])
    return int((labels[:, head] != IGNORE_LABEL).sum())


CSV_FIELDNAMES = [
    "model_name",
    "head",
    "question_text",
    "n_train",
    "n_test",
    "accuracy",
    "f1_binary",
    "f1_macro",
    "precision",
    "recall",
    "train_seconds",
]


def init_results_csv(output_path: Path) -> None:
    """Create (or truncate) the results CSV and write the header row."""
    with output_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDNAMES).writeheader()
    logger.info("Results CSV initialised: %s", output_path)


def append_result_csv(result: ModelResult, output_path: Path) -> None:
    """Append a single result row to the CSV."""
    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writerow(
            {
                "model_name": result.model_name,
                "head": result.head,
                "question_text": result.question,
                "n_train": result.n_train,
                "n_test": result.n_test,
                "accuracy": f"{result.accuracy:.4f}",
                "f1_binary": f"{result.f1_binary:.4f}",
                "f1_macro": f"{result.f1_macro:.4f}",
                "precision": f"{result.precision:.4f}",
                "recall": f"{result.recall:.4f}",
                "train_seconds": f"{result.train_seconds:.1f}",
            }
        )


def train_answerer(
    questions: list[str] | None = None,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    save_checkpoints: bool = True,
) -> None:
    """Train one model to answer every one of `questions`, defaulting to every
    question the labelled data has answers for.

    The trained model replaces whatever is already on disk, and its heads are
    recorded in the model map, so the questions become available to
    PastelLocal; nothing else needs updating. Its holdout metrics go to
    results.csv and the log.
    """
    input_path = LABELLED_DATA_PATH

    models_dir().mkdir(parents=True, exist_ok=True)
    setup_logging(models_dir())

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        raise SystemExit(1)

    device = auto_detect_device()
    logger.info("Using device: %s", device)
    if device != "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    records = load_labelled_data(input_path)
    in_head_order = questions_in_head_order(
        questions_in(records) if questions is None else questions
    )
    dataset = build_dataset(records, in_head_order)

    base_model_id = MODELS[MODEL_CATEGORY]
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    # One split over sentences: every head is trained and evaluated on the same
    # sentences, so a sentence cannot be in one head's training set and another
    # head's holdout.
    split = tokenise_dataset(dataset, tokenizer).train_test_split(
        test_size=TEST_FRACTION, seed=RANDOM_SEED
    )
    train_ds, test_ds = split["train"], split["test"]
    logger.info(
        "%d question(s) over %d sentences (train=%d, test=%d)",
        len(in_head_order),
        len(dataset.sentences),
        len(train_ds),
        len(test_ds),
    )

    csv_path = models_dir() / "results.csv"
    init_results_csv(csv_path)

    metrics, elapsed = finetune_multi_head(
        base_model_id=base_model_id,
        questions=in_head_order,
        train_ds=train_ds,
        test_ds=test_ds,
        output_dir=model_dir(),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        save_checkpoints=save_checkpoints,
    )

    for head, question in enumerate(in_head_order):
        result = ModelResult(
            model_name=MODEL_CATEGORY,
            question=question,
            head=head,
            n_train=labelled_counts(train_ds, head),
            n_test=labelled_counts(test_ds, head),
            accuracy=metrics.get(f"q{head:02d}_accuracy", float("nan")),
            f1_binary=metrics.get(f"q{head:02d}_f1_binary", float("nan")),
            f1_macro=metrics.get(f"q{head:02d}_f1_macro", float("nan")),
            precision=metrics.get(f"q{head:02d}_precision", float("nan")),
            recall=metrics.get(f"q{head:02d}_recall", float("nan")),
            train_seconds=elapsed,
        )
        append_result_csv(result, csv_path)
        logger.info(
            "  head %02d  accuracy=%.3f  f1_binary=%.3f  f1_macro=%.3f  %s",
            head,
            result.accuracy,
            result.f1_binary,
            result.f1_macro,
            question[:50],
        )

    logger.info(
        "Trained %d head(s) in %.1fs, mean f1_binary=%.3f",
        len(in_head_order),
        elapsed,
        metrics.get("mean_f1_binary", float("nan")),
    )


if __name__ == "__main__":
    train_answerer()
