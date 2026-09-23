"""Fine-tune the multi-head encoder that `PastelLocal` answers questions with.

The caller supplies labelled sentences; how they were gathered, split or
balanced belongs to the downstream task. The result is written where
`model_registry` looks for it, with the question -> head map beside it, so a
trained model is usable by `PastelLocal` without anything else being updated.

Needs the optional training dependencies:

    uv sync --extra train
"""

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pastel.local.model_registry import (
    MODEL_CATEGORY,
    MODEL_DIR_NAME,
    MODELS,
    record_heads,
)
from pastel.local.multi_head_encoder import IGNORE_LABEL, MultiHeadEncoder

_logger = logging.getLogger(__name__)

MAX_LENGTH = 128  # must match local_answerer.MAX_LENGTH
RANDOM_SEED = 42
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5

METRIC_NAMES = ("accuracy", "f1_binary", "f1_macro", "precision", "recall")


@dataclass
class Split:
    """One split: a sentence per row, with one label per head.

    `IGNORE_LABEL` marks a question that sentence has no answer for. The
    model's loss masks those out, so a partly-labelled sentence still trains
    the heads it does have answers for.
    """

    sentences: list[str]
    labels: list[list[int]]

    def __len__(self) -> int:
        return len(self.sentences)

    def answered(self, head: int) -> list[int]:
        """The labels this split has for one head."""
        return [row[head] for row in self.labels if row[head] != IGNORE_LABEL]

    def without_sentences_in(self, training: "Split") -> tuple["Split", int]:
        """This split minus any sentence the body was trained on, and how many
        were dropped.

        The body is shared, so a sentence used to train any head is one the
        model has seen whatever question it is later scored on. That is the one
        error that makes a model look *better* than it is, with no other
        symptom, so evaluation splits are worth passing through this even when
        the sentences were divided once for every question.
        """
        trained_on = set(training.sentences)
        keep = [
            (sentence, row)
            for sentence, row in zip(self.sentences, self.labels)
            if sentence not in trained_on
        ]
        kept = Split(
            sentences=[sentence for sentence, _ in keep],
            labels=[row for _, row in keep],
        )
        return kept, len(self) - len(kept)


@dataclass
class TrainedModel:
    checkpoint: Path
    metrics: list[dict[str, float]]  # one per head, in head order
    mean_f1_binary: float
    seconds: float


def auto_detect_device(requested: str | None = None) -> str:
    """The best device available, or the one asked for."""
    if requested:
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def binary_metrics(labels: Any, preds: Any) -> dict[str, float]:
    """Read `f1_binary`, not accuracy: on a skewed question a head that always
    answers 'no' still scores high accuracy, and f1_binary is what exposes it."""
    from sklearn.metrics import (  # type: ignore[import-untyped]
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    if len(labels) == 0:
        return dict.fromkeys(METRIC_NAMES, float("nan"))
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_binary": float(f1_score(labels, preds, average="binary", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "precision": float(
            precision_score(labels, preds, average="binary", zero_division=0)
        ),
        "recall": float(recall_score(labels, preds, average="binary", zero_division=0)),
    }


def _compute_metrics(n_heads: int) -> Any:
    """Metrics per head: one number over all of them would hide a head that has
    stopped working. Each head is scored only on the sentences it has labels
    for."""
    import numpy as np

    def compute(eval_pred: Any) -> dict[str, float]:
        logits, label_ids = eval_pred
        preds = np.argmax(logits, axis=-1)  # (sentences, heads)

        metrics: dict[str, float] = {}
        for head in range(n_heads):
            answered = label_ids[:, head] != IGNORE_LABEL
            for name, value in binary_metrics(
                label_ids[answered, head], preds[answered, head]
            ).items():
                metrics[f"h{head:02d}_{name}"] = value
        metrics["mean_f1_binary"] = float(
            np.nanmean([metrics[f"h{h:02d}_f1_binary"] for h in range(n_heads)])
        )
        return metrics

    return compute


def _torch_dataset(split: Split, tokenizer: Any) -> Any:
    """Tokenise one split into something Trainer will iterate. A plain torch
    Dataset rather than a HuggingFace one: a dozen lines saves the whole
    `datasets` dependency."""
    import torch

    encoded = tokenizer(
        split.sentences, padding="max_length", truncation=True, max_length=MAX_LENGTH
    )
    labels = split.labels

    class Encoded(torch.utils.data.Dataset[dict[str, Any]]):
        def __len__(self) -> int:
            return len(labels)

        def __getitem__(self, index: int) -> dict[str, Any]:
            item = {key: torch.tensor(values[index]) for key, values in encoded.items()}
            item["labels"] = torch.tensor(labels[index])
            return item

    return Encoded()


def train_multi_head(
    questions: list[str],
    train: Split,
    evaluation: Split,
    output_dir: Path,
    base_model_id: str | None = None,
    model_category: str = MODEL_CATEGORY,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    seed: int = RANDOM_SEED,
    device: str | None = None,
) -> TrainedModel:
    """Train one encoder with a head per question, and save it under
    `output_dir` where `PastelLocal` can load it.

    `questions` is in head order: question `i` is answered by head `i`, which
    is what gets written to `model_map.json`. The body is shared, so a retrain
    replaces the whole model - every question has to be trained together.
    """
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    device = auto_detect_device(device)
    base_model_id = base_model_id or MODELS[model_category]
    model_dir = output_dir / model_category / MODEL_DIR_NAME
    working_dir = model_dir / "_training"

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = MultiHeadEncoder.from_base_model(base_model_id, len(questions))

    steps_per_epoch = max(1, len(train) // batch_size)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(working_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            lr_scheduler_type="linear",
            warmup_steps=max(1, int(0.1 * steps_per_epoch * epochs)),
            eval_strategy="epoch",
            # Nothing saved per epoch: "the best epoch" would be one number
            # over every question, and each checkpoint is over a gigabyte.
            save_strategy="no",
            # Trainer otherwise takes the best accelerator it can find, which
            # makes device="cpu" a lie rather than a choice.
            use_cpu=device == "cpu",
            seed=seed,
            report_to="none",
            logging_steps=50,
        ),
        train_dataset=_torch_dataset(train, tokenizer),
        eval_dataset=_torch_dataset(evaluation, tokenizer),
        compute_metrics=_compute_metrics(len(questions)),
    )

    _logger.info(
        "Training %d head(s) from %s on %s", len(questions), base_model_id, device
    )
    started = time.time()
    trainer.train()
    elapsed = time.time() - started
    metrics = {
        key.removeprefix("eval_"): value for key, value in trainer.evaluate().items()
    }

    # A numbered checkpoint is the only form model_registry.checkpoints()
    # recognises. Trainer's working directory holds nothing worth keeping.
    checkpoint = model_dir / f"checkpoint-{int(trainer.state.global_step or 1)}"
    trainer.save_model(str(checkpoint))
    shutil.rmtree(working_dir, ignore_errors=True)
    record_heads(questions, models_dir=output_dir, model_category=model_category)

    per_head = [
        {
            name: metrics.get(f"h{head:02d}_{name}", float("nan"))
            for name in METRIC_NAMES
        }
        for head in range(len(questions))
    ]
    return TrainedModel(
        checkpoint=checkpoint,
        metrics=per_head,
        mean_f1_binary=metrics.get("mean_f1_binary", float("nan")),
        seconds=elapsed,
    )
