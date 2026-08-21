# Claude-created
"""Fine-tune encoder-only LLMs on labelled data.


Each model is fine-tuned separately for each question (NLI-style: input = question + sentence).

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
from sklearn.model_selection import StratifiedShuffleSplit  # type: ignore
from transformers import AutoTokenizer

from local_models.model_registry import assign_model_id

logger = logging.getLogger(__name__)


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


# Just using ModernBERT-multilingual but other options are available
MODELS: dict[str, str] = {
    "ModernBERT-multilingual": "jhu-clsp/mmBERT-base",
    # "mDeBERTa-v3-base": "microsoft/mdeberta-v3-base",
    # "XLM-RoBERTa-base": "FacebookAI/xlm-roberta-base",
}

RANDOM_SEED = 42
MAX_LENGTH = 128  # 128 is enough for c.95% of sentences; 256 would cover them all
TEST_FRACTION = 0.2


@dataclass
class QuestionDataset:
    # set of training data for one Pastel question
    question: str
    inputs: list[str]
    labels: list[int]


@dataclass
class ModelResult:
    model_name: str
    question: str
    question_label: str
    n_train: int
    n_test: int
    accuracy: float
    f1_binary: float
    f1_macro: float
    precision: float
    recall: float
    train_seconds: float


def load_labelled_data(input_path: Path) -> list[dict[str, Any]]:
    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d labelled records from %s", len(records), input_path)
    return records


def build_question_dataset(
    records: list[dict[str, Any]],
    question: str,
) -> QuestionDataset:
    """Reformat trainig data into a QuestionDataSet object"""
    inputs, labels = [], []
    n_filtered = 0
    for record in records:
        answers = record.get("question_answers", {})
        if question not in answers:
            print(f"No answers for question {question}")
            continue
        answer = answers[question]
        if answer == 0.5:
            n_filtered += 1
            continue
        inputs.append(question + " " + record["sentence_text"])
        labels.append(int(answer))
    if n_filtered:
        logger.info(
            "Question %r: filtered %d unsure (0.5) records, %d remaining",
            question[:50],
            n_filtered,
            len(inputs),
        )
    return QuestionDataset(question=question, inputs=inputs, labels=labels)


def split_dataset(
    qd: QuestionDataset,
    test_fraction: float = TEST_FRACTION,
    random_state: int = RANDOM_SEED,
) -> tuple[QuestionDataset, QuestionDataset] | None:
    """
    Stratified train/test split.
    Returns None (and warns) if a class has fewer than 2 examples.
    """

    labels_arr = np.array(qd.labels)
    unique, counts = np.unique(labels_arr, return_counts=True)

    if len(unique) < 2:
        logger.warning(
            "Question %r: only one class present (%s), skipping.",
            qd.question[:50],
            unique,
        )
        return None

    if counts.min() < 2:
        logger.warning(
            "Question %r: class %s has only %d example(s), skipping.",
            qd.question[:50],
            unique[counts.argmin()],
            counts.min(),
        )
        return None

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_fraction, random_state=random_state
    )
    indices = list(range(len(qd.inputs)))
    train_idx, test_idx = next(sss.split(indices, qd.labels))

    train_qd = QuestionDataset(
        question=qd.question,
        inputs=[qd.inputs[i] for i in train_idx],
        labels=[qd.labels[i] for i in train_idx],
    )
    test_qd = QuestionDataset(
        question=qd.question,
        inputs=[qd.inputs[i] for i in test_idx],
        labels=[qd.labels[i] for i in test_idx],
    )
    return train_qd, test_qd


def tokenise_dataset(qd: QuestionDataset, tokenizer: Any) -> hf_datasets.Dataset:

    tokenised = tokenizer(
        qd.inputs,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    data = {k: v for k, v in tokenised.items()}
    data["labels"] = qd.labels
    return hf_datasets.Dataset.from_dict(data)


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    from sklearn.metrics import (  # type: ignore
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    logits, label_ids = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(label_ids, preds)),
        "f1_binary": float(
            f1_score(label_ids, preds, average="binary", zero_division=0)
        ),
        "f1_macro": float(f1_score(label_ids, preds, average="macro", zero_division=0)),
        "precision": float(
            precision_score(label_ids, preds, average="binary", zero_division=0)
        ),
        "recall": float(
            recall_score(label_ids, preds, average="binary", zero_division=0)
        ),
    }


def finetune_one_model(
    model_key: str,
    model_id: str,
    train_ds: hf_datasets.Dataset,
    test_ds: hf_datasets.Dataset,
    output_dir: Path,
    question_label: str,
    epochs: int,
    batch_size: int,
    lr: float,
    save_checkpoints: bool,
) -> tuple[dict[str, float], float]:
    from transformers import (
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
    )

    """Train an encoder model to answer true/false questions"""

    checkpoint_dir = output_dir / model_key / question_label
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
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

    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
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


def get_question_id(question: str) -> str:
    """The model id (directory name) to train this question's model into.
    Shared with inference via local_models.model_registry, so that
    local_answerer looks the model up under the same name."""
    return assign_model_id(question)


def train_one_model(
    question_dataset: QuestionDataset,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    save_checkpoints: bool,
    csv_path: Path,
) -> list[ModelResult]:
    """For this question, load the annotated dataset then train a local transformer model.
    Update the file mapping questions to model names."""

    results: list[ModelResult] = []
    question = question_dataset.question
    question_label = get_question_id(question)

    # for q_idx, qd in enumerate(question_datasets):
    split = split_dataset(question_dataset)
    if split is None:
        return []
    #     fail # TODO: handle this correctly: raise exception as it's a pretty terminal failing
    train_qd, test_qd = split
    # question_label = f"q{q_idx:02d}"
    logger.info(
        "Question %s: (train=%d, test=%d)",
        question[:60],
        len(train_qd.inputs),
        len(test_qd.inputs),
    )

    for model_key, model_id in MODELS.items():
        logger.info("  Training %s (%s)...", model_key, model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        train_ds = tokenise_dataset(train_qd, tokenizer)
        test_ds = tokenise_dataset(test_qd, tokenizer)
        print(f"Training '{question[:40]}...' ")
        try:
            metrics, elapsed = finetune_one_model(
                model_key=model_key,
                model_id=model_id,
                train_ds=train_ds,
                test_ds=test_ds,
                output_dir=output_dir,
                question_label=question_label,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                save_checkpoints=save_checkpoints,
            )
        except Exception as e:
            logger.error("  Failed for %s / %s", model_key, e)
            continue

        result = ModelResult(
            model_name=model_key,
            question=question,
            question_label=question_label,
            n_train=len(train_qd.inputs),
            n_test=len(test_qd.inputs),
            accuracy=metrics.get("accuracy", float("nan")),
            f1_binary=metrics.get("f1_binary", float("nan")),
            f1_macro=metrics.get("f1_macro", float("nan")),
            precision=metrics.get("precision", float("nan")),
            recall=metrics.get("recall", float("nan")),
            train_seconds=elapsed,
        )
        results.append(result)
        append_result_csv(result, csv_path)
        logger.info(
            "    accuracy=%.3f  f1_binary=%.3f  f1_macro=%.3f  (%.1fs)",
            result.accuracy,
            result.f1_binary,
            result.f1_macro,
            elapsed,
        )

    return results


CSV_FIELDNAMES = [
    "model_name",
    "question_index",
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
                "question_index": result.question_label,
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


def build_one_question_answerer(question: str) -> None:

    input_path = Path("data/local_models/labelled_sentences.jsonl")
    output_dir = Path("data/local_models/models")
    epochs = 3
    batch_size = 16
    lr = 2e-5
    save_checkpoints = True

    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        raise SystemExit(1)

    device = auto_detect_device()
    logger.info("Using device: %s", device)
    if device != "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    records = load_labelled_data(input_path)
    question_dataset = build_question_dataset(records, question)

    csv_path = output_dir / "results.csv"
    init_results_csv(csv_path)

    _ = train_one_model(
        question_dataset=question_dataset,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        save_checkpoints=save_checkpoints,
        csv_path=csv_path,
    )


if __name__ == "__main__":
    # main()
    new_question = "Is this sentence a joke or satirical?"
    build_one_question_answerer(new_question)
