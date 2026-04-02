# Claude-created
"""Script 2: Fine-tune encoder-only LLMs on the labelled data from label_sentences.py.

Trains three models on binary yes/no classification for each question:
  - ModernBERT-base-multilingual  (answerdotai/ModernBERT-base-multilingual)
  - mDeBERTa-v3-base              (microsoft/mdeberta-v3-base)
  - XLM-RoBERTa-base              (xlm-roberta-base)

Each model is fine-tuned separately for each question (NLI-style: input = question + sentence).
Results are written to a CSV and a summary table is printed to stdout.

Dependencies (install manually, not in pyproject.toml):
    pip install "transformers>=4.40" datasets torch accelerate scikit-learn

Usage:
    python scripts/encoder_experiment/finetune_encoder.py \\
        --input scripts/encoder_experiment/labelled_sentences.jsonl \\
        --output-dir scripts/encoder_experiment/results
"""

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

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


MODELS: dict[str, str] = {
    "ModernBERT-multilingual": "jhu-clsp/mmBERT-base",
    # "mDeBERTa-v3-base": "microsoft/mdeberta-v3-base",
    # "XLM-RoBERTa-base": "FacebookAI/xlm-roberta-base",
}

RANDOM_SEED = 42
MAX_LENGTH = 128
TEST_FRACTION = 0.2


@dataclass
class QuestionDataset:
    question: str
    inputs: list[str]
    labels: list[int]


@dataclass
class ModelResult:
    model_name: str
    question: str
    question_index: int
    n_train: int
    n_test: int
    accuracy: float
    f1_binary: float
    f1_macro: float
    precision: float
    recall: float
    train_seconds: float


def load_labelled_data(input_path: Path) -> list[dict]:
    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d labelled records from %s", len(records), input_path)
    return records


def build_question_datasets(
    records: list[dict],
    questions: list[str],
) -> list[QuestionDataset]:
    """
    For each question, build a QuestionDataset containing all the questions + labels (0/1).
    Input string = question + " " + sentence_text (tokeniser adds [CLS]/[SEP]).
    Label = int(answer) for 0.0 or 1.0; records with 0.5 (unsure) are filtered out.
    """
    datasets = []
    for question in questions:
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
        datasets.append(
            QuestionDataset(question=question, inputs=inputs, labels=labels)
        )
    return datasets


def split_dataset(
    qd: QuestionDataset,
    test_fraction: float = TEST_FRACTION,
    random_state: int = RANDOM_SEED,
) -> tuple[QuestionDataset, QuestionDataset] | None:
    """
    Stratified 80/20 train/test split.
    Returns None (and warns) if a class has fewer than 2 examples.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

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


def tokenise_dataset(qd: QuestionDataset, tokenizer) -> "datasets.Dataset":
    import datasets as hf_datasets

    tokenised = tokenizer(
        qd.inputs,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    data = {k: v for k, v in tokenised.items()}
    data["labels"] = qd.labels
    return hf_datasets.Dataset.from_dict(data)


def compute_metrics(eval_pred) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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


def train_one_model(
    model_key: str,
    model_id: str,
    train_ds,
    test_ds,
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


def run_experiment(
    question_datasets: list[QuestionDataset],
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    save_checkpoints: bool,
    questions: list[str],
    csv_path: Path,
) -> list[ModelResult]:
    from transformers import AutoTokenizer

    results: list[ModelResult] = []

    for q_idx, qd in enumerate(question_datasets):
        split = split_dataset(qd)
        if split is None:
            continue
        train_qd, test_qd = split
        question_label = f"q{q_idx:02d}"
        logger.info(
            "Question %d/%d: %r  (train=%d, test=%d)",
            q_idx + 1,
            len(question_datasets),
            qd.question[:60],
            len(train_qd.inputs),
            len(test_qd.inputs),
        )

        for model_key, model_id in MODELS.items():
            logger.info("  Training %s (%s)...", model_key, model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)

            train_ds = tokenise_dataset(train_qd, tokenizer)
            test_ds = tokenise_dataset(test_qd, tokenizer)

            try:
                metrics, elapsed = train_one_model(
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
                logger.error("  Failed for %s / question %d: %s", model_key, q_idx, e)
                continue

            result = ModelResult(
                model_name=model_key,
                question=qd.question,
                question_index=q_idx,
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
                "question_index": result.question_index,
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


def print_summary_table(results: list[ModelResult]) -> None:
    if not results:
        print("No results to summarise.")
        return

    print("\n" + "=" * 70)
    print("SUMMARY: Mean ± SD of binary F1 across all questions")
    print("=" * 70)
    print(f"{'Model':<30}  {'N questions':>11}  {'Mean F1':>8}  {'SD F1':>7}")
    print("-" * 70)

    for model_key in MODELS:
        model_results = [r for r in results if r.model_name == model_key]
        if not model_results:
            continue
        f1_scores = [r.f1_binary for r in model_results if not np.isnan(r.f1_binary)]
        if not f1_scores:
            print(f"{model_key:<30}  {'—':>11}  {'—':>8}  {'—':>7}")
            continue
        mean_f1 = np.mean(f1_scores)
        sd_f1 = np.std(f1_scores)
        print(f"{model_key:<30}  {len(f1_scores):>11}  {mean_f1:>8.3f}  {sd_f1:>7.3f}")

    print("=" * 70 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="scripts/encoder_experiment/labelled_sentences.jsonl",
        help="Path to labelled JSONL from label_sentences.py",
    )
    parser.add_argument(
        "--output-dir",
        default="scripts/encoder_experiment/results",
        help="Directory for model checkpoints and results CSV",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Per-device batch size (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving model checkpoints",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override: 'cpu', 'cuda', 'mps' (default: auto-detect)",
    )
    return parser.parse_args()


def main() -> None:
    # Defer heavy imports to here so --help works without torch installed
    try:
        import datasets  # noqa: F401
        import torch
        import transformers  # noqa: F401
    except ImportError as e:
        logger.error(
            "Missing dependency: %s\n"
            "Install with: pip install 'transformers>=4.40' datasets torch accelerate scikit-learn",
            e,
        )
        raise SystemExit(1)

    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        raise SystemExit(1)

    device = args.device or auto_detect_device()
    logger.info("Using device: %s", device)
    if device != "cpu":
        import os

        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    from questions import QUESTIONS

    records = load_labelled_data(input_path)
    question_datasets = build_question_datasets(records, QUESTIONS)

    csv_path = output_dir / "results.csv"
    init_results_csv(csv_path)

    results = run_experiment(
        question_datasets=question_datasets,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_checkpoints=not args.no_save,
        questions=QUESTIONS,
        csv_path=csv_path,
    )

    print_summary_table(results)


if __name__ == "__main__":
    main()
