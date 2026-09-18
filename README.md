# PASTEL

A concept from Sheffield University[1]: ask a fixed list of yes/no questions
about a piece of text, then combine the answers into a single score with a
linear regression model.

At Full Fact this identifies claims worth bringing to professional fact
checkers, with questions such as "Could believing this claim harm someone's
health?". The same approach could score text for propaganda, bias or
reliability — the library declares no questions of its own.

## Code overview

`pastel/pastel.py` defines `PastelModel`: the features-to-weights model itself,
saving and loading it, and turning answers into a score. It is abstract —
answering the questions is left to a backend:

* `PastelGemini` sends all of a model's questions to Gemini in one prompt per
  sentence.
* `PastelLocal` answers every question with one locally fine-tuned encoder,
  which has a classification head per question and so answers them all in a
  single pass. `pastel/local/` holds the registry, the loading code and the
  fine-tuning.

They are drop-in replacements, so pick one at runtime rather than by changing
imports:

```python
from pastel import get_backend

pastel = get_backend("local").load_model("my_model.json")   # or "gemini"
```

With no argument, `get_backend()` reads `$PASTEL_BACKEND` and falls back to
Gemini. Each demo script in `scripts/` takes the same choice as `--backend`.

`pastel/optimise_weights.py` fits the regression weights from a list of
sentences with checkworthy scores.

`training/cached_pastel.py` wraps any backend and caches its responses in a
local SQLite database, which saves a lot of time when re-analysing the same
sentences while experimenting. It is no use in production, where each sentence
is seen once. `training/crossvalidate_pastel.py` and `training/beam_search.py`
compare large numbers of candidate models to find a good combination of
questions; `beam_search` is heuristic and much faster. `data/sample_responses.db`
is a sample cache to initialise the `DatabaseManager` with.

### Pastel functions and claim types

`pastel_functions` holds functions returning a true/false value for a single
sentence, such as `is_claim_type_quantity`, so a model can score quantity-type
sentences higher or lower. Sentences must carry their claim types on the
`Sentence` class for these to work: a sentence without them is treated as
having none, which gives poor scores. Only use claim-type functions in models
deployed where claim types are available.

## Setup

Set `GEMINI_PROJECT`, `GEMINI_LOCATION` and `GEMINI_MODEL` unless you configure
Gemini yourself.

The Gemini backend needs nothing beyond the base install. The local backend
needs `transformers` and `torch`, and fine-tuning also needs `accelerate`:

```
uv sync --extra local     # to answer questions with a fine-tuned model
uv sync --extra train     # to fine-tune one
```

PASTEL runs the local models on CPU, so the CPU build of torch is installed.

### The local backend

One encoder answers every question, with a head per question. `model_map.json`
— written by training, read by inference — records which head answers which
question, and is the only thing that says what the backend can answer.

`PASTEL_LOCAL_MODELS_DIR` sets the directory holding the model, for training
and inference. It defaults to `data/local_models/models`, which is relative and
so only resolves from the repo root. The model itself sits under the category
name, in `multi_head/checkpoint-*`.

```
python -m pastel.local
```

reports each recorded question as OK or MISSING and prints where it looked. In
code, `available_questions()` gives the recorded questions once the model has
been trained, and `PastelLocal.from_available_questions()` builds a model from
exactly those.

`PASTEL_LOCAL_QUANTISE=1` quantises the model's linear layers to int8 as it
loads, worth roughly 20% of inference time on CPU. It changes the numerics, so
re-run your holdout evaluation before trusting a model with it on.

### Fine-tuning

`pastel.local.training.train_multi_head()` trains one encoder with a head per
question from labelled sentences, saves it where `PastelLocal` will find it,
and writes the model map beside it. Gathering, splitting and balancing the
labels belongs to the downstream task — see genai-checkworthy for an example.

Because the body is shared, a retrain replaces the whole model: adding or
rewording one question means training all of them again. Sentences need not be
labelled for every question — an unlabelled answer is masked out of the loss
for that head only.

The input is the bare sentence. Each head answers one fixed question, so
prefixing the question text would only spend the forward pass encoding a
constant. Training and inference must agree on this.

### Billing labels

`PastelGemini` takes an optional `labels` dict, attached to each Gemini call so
its spend can be separated out in Google Cloud billing:

```python
pastel = PastelGemini.from_dict(weights, labels={"task": "checkworthy_pastel"})
```

These merge with any `GENAI_LABEL_*` environment variables `genai_utils` picks
up, so a per-task label composes with a service-level one. Keys must start with
a lowercase letter; keys and values may only contain lowercase letters,
numbers, `-` and `_`, up to 63 characters. `genai_utils` drops an invalid label
with a warning rather than failing the call, so a typo means untagged spend.
`PastelLocal` makes no Gemini calls and takes no labels.

## Upgrading from 1.x

Splitting the backends removed `Pastel`, the class that used to do everything.
`PastelModel` is now the abstract base; use `PastelGemini` (or `get_backend()`)
wherever you used `Pastel`, with the same arguments — `from_dict`,
`load_model`, `from_feature_list` and `labels` are unchanged. `make_prompt` is
Gemini-specific and now `PastelGemini._make_prompt`. Everything else keeps its
name and signature.

## A note on data

`data/example_training_data.jsonl` lets the tests and demos run. It was
generated with Gemini and is not real news media, for copyright reasons —
please provide your own examples.

## Citation

[1] Leite, J. A., Razuvayevskaya, O., Bontcheva, K., & Scarton, C. (2025).
[Weakly supervised veracity classification with LLM-predicted credibility signals](https://arxiv.org/abs/2309.07601).
EPJ Data Science, 14(1), 16.
