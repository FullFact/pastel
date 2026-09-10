"""One shared encoder body with a binary classification head per question.

Every question asks about the same sentence, so nine separately fine-tuned
encoders spent nine forward passes - and nine copies of a 307M-parameter body -
producing nine yes/no answers. One body with one small head per question
answers all of them in a single pass, for a ninth of the memory.

The cost is that the questions are no longer independent: the body is shared,
so adding or changing a question means retraining all of them together.

Unlike the rest of `pastel.local`, this module imports torch and transformers
at module scope - it cannot define a torch module otherwise. They are an
optional extra, so import this module from inside a function rather than at the
top of one that Gemini-only installs also import.
"""

from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig, AutoModel

N_CLASSES = 2  # every question is answered yes or no

# Marks "this sentence has no answer for this question" in a label vector. A
# sentence labelled for only some questions still trains the heads it does have
# answers for. -100 is torch's own default ignore_index.
IGNORE_LABEL = -100

# What transformers' Trainer names the weights it saves for a plain nn.Module.
WEIGHTS_FILENAME = "model.safetensors"

HEAD_PREFIX = "heads."


def _classification_head(hidden_size: int) -> nn.Module:
    """One question's head, mirroring the shape of the base model's own
    classification head so that a shared body starts from what already worked
    when each question had a whole model to itself."""
    return nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.GELU(),
        nn.LayerNorm(hidden_size),
        nn.Linear(hidden_size, N_CLASSES),
    )


class MultiHeadEncoder(nn.Module):
    """A pre-trained encoder with `n_heads` binary classification heads, head
    `i` answering the question the model map records against index `i`."""

    def __init__(self, encoder: Any, n_heads: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleList(
            _classification_head(encoder.config.hidden_size) for _ in range(n_heads)
        )
        # Which sentence representation the base model's own classifier uses.
        # Pooling the other way would quietly cost accuracy.
        self.pooling = getattr(encoder.config, "classifier_pooling", "cls")

    @property
    def n_heads(self) -> int:
        return len(self.heads)

    @classmethod
    def from_base_model(cls, base_model_id: str, n_heads: int) -> "MultiHeadEncoder":
        """A model ready to fine-tune: the pre-trained body, untrained heads."""
        return cls(AutoModel.from_pretrained(base_model_id), n_heads)

    @classmethod
    def from_checkpoint(
        cls, checkpoint: Path, base_model_id: str
    ) -> "MultiHeadEncoder":
        """A fine-tuned model, in evaluation mode. The body's shape comes from
        the base model's config and every weight from the checkpoint, so this
        does not download the base model's weights only to overwrite them.

        How many heads the checkpoint has is read from the checkpoint itself
        rather than from the model map, so that a map listing questions the
        model was not trained for fails where it can be explained.
        """
        state = load_file(checkpoint / WEIGHTS_FILENAME)
        n_heads = len(
            {
                key.split(".")[1]
                for key in state
                if key.startswith(HEAD_PREFIX) and key.split(".")[1].isdigit()
            }
        )
        config = AutoConfig.from_pretrained(base_model_id)
        encoder = AutoModel.from_config(config)  # type: ignore[no-untyped-call]
        model = cls(encoder, n_heads)
        model.load_state_dict(state)
        model.eval()
        return model

    def _pooled(self, input_ids: Any, attention_mask: Any) -> torch.Tensor:
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        if self.pooling == "mean":
            weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * weights).sum(dim=1) / weights.sum(dim=1)
        return hidden[:, 0]  # the [CLS] position

    def head_logits(self, input_ids: Any, attention_mask: Any) -> torch.Tensor:
        """Every head's logits for every sentence, shaped
        (sentences, heads, classes). One pass of the body answers every
        question; the heads themselves are too small to be worth batching."""
        pooled = self._pooled(input_ids, attention_mask)
        return torch.stack([head(pooled) for head in self.heads], dim=1)

    def forward(
        self,
        input_ids: Any,
        attention_mask: Any,
        labels: Any = None,
    ) -> dict[str, Any]:
        """The training entry point transformers' Trainer calls. `labels` holds
        one answer per head for each sentence, IGNORE_LABEL where that sentence
        has no answer for that question."""
        logits = self.head_logits(input_ids, attention_mask)
        if labels is None:
            return {"logits": logits}
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, N_CLASSES),
            labels.reshape(-1),
            ignore_index=IGNORE_LABEL,
        )
        return {"loss": loss, "logits": logits}
