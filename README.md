# PASTEL

This is a concept from Sheffield University[1], where the prompt consists of a series of yes/no questions. The answers to these questions, in the context of a piece of text, are then combined into a single score using a linear regression model. 

At Full Fact, this approach is used to help identify claims that are worth bringing to the attention of professional fact checkers. That model includes questions such as "Could believing this claim harm someone's health?" and "Is this sentence likely to be believed by many people?".

### Code overview

The `pastel/pastel.py` module defines `PastelModel`: the features-to-weights model itself, saving and loading it, and turning a set of answers into a single score. It is abstract - answering the questions is left to a backend, which is the only part that differs between them:

* `pastel/pastel_gemini.py` — `PastelGemini` sends all of a model's questions to Gemini in one prompt per sentence.
* `pastel/pastel_local.py` — `PastelLocal` answers every question with one locally fine-tuned encoder, which has a classification head per question and so answers them all in a single pass per sentence. `pastel/local/` holds the model registry and the loading code; training the model lives in `local_models/` (see its README) and is not part of the installable library.

Both are drop-in replacements for each other, so pick one at runtime with `pastel.get_backend()` rather than by changing imports:

```python
from pastel import get_backend

pastel = get_backend("local").load_model("my_model.json")   # or "gemini"
```

With no argument, `get_backend()` reads the `PASTEL_BACKEND` environment variable and falls back to Gemini. Each demo script in `scripts/` takes the same choice as a `--backend` flag.

`PastelGemini` also takes optional Vertex billing `labels`, attached to every Gemini call it makes so its spend can be separated out in Google Cloud billing. They are threaded through `from_dict`, `load_model` and `from_feature_list`, and survive the model copying that training does.

The `pastel/optimise_weights.py` module calculates the parameters of the regression model, and requires a list of sentences with associated checkworthy scores.

Currently, this is used by the genai-checkworthy repo but in the future, the same approach might be used to analyse text for other features such as propaganda, bias, reliability etc.

`training/cached_pastel.py` wraps any backend and uses a local SQLite database to cache its responses. This saves a lot of time and effort when re-analysing the same sentences over and over again, so is useful for experimenting with/optimising Pastel models, but should not be used in production. (It won't help there anyway, as each sentence is only ever seen once.) Similarly, `training/crossvalidate_pastel.py` and `training/beam_search.py` are scripts to compare a large number of Pastel models (potentially millions!) to help find a good combination of questions. `beam_search` uses heuristics and is a lot faster. There is a sample database of cached answers in `data/sample_responses.db` that can be used to initialise the DatabaseManager.

### Upgrading from 1.x

Splitting the backends renamed the class that used to do everything, so `Pastel`
no longer exists. `PastelModel` is the abstract base; pick the backend you want:

| 1.x | 2.x |
| --- | --- |
| `from pastel.pastel import Pastel` | `from pastel import PastelGemini` (or `get_backend()`) |
| `Pastel(model, labels)` | `PastelGemini(model, labels)` |
| `Pastel.from_dict(d, labels)` | `PastelGemini.from_dict(d, labels)` |
| `Pastel.load_model(path, labels)` | `PastelGemini.load_model(path, labels)` |
| `Pastel.from_feature_list(features, labels)` | `PastelGemini.from_feature_list(features, labels)` |
| `pastel.make_prompt(sentence)` | `PastelGemini._make_prompt(sentence)` — Gemini-specific, now internal |

Everything else keeps its name and signature, `labels` included: `make_predictions`,
`update_predictions`, `save_model`, `display_model`, `get_questions`,
`get_functions`, `get_bias`, `quantify_answers`, `get_scores_from_answers`, and
the `Sentence` / `ScoreAndAnswers` / `BiasType` models. So for Gemini users the
migration is the import and the class name.

### Pastel Functions and Claim Types

The `pastel_functions` module defines a set of functions that return a true/false value for a single sentence. One current use is for claim types with functions such as `is_claim_type_quantity`, which allows Pastel models to give higher (or lower) scores to quantity-type sentences. To make this work, sentences must specify the list of claim types as part of a Sentence class (see `pastel/models.py`). If sentences without claim types are used, then any claim type function in a Pastel model will treat the sentence as NOT having any claim types, which will lead to poor performance. So it's important to only use claim-type functions in Pastel models deployed to platforms that have claim-types added to each sentence.

## Setup

If you don't want to manually specify the config of Gemini, you should set the following environment variables:
* `GEMINI_PROJECT`: the GCP project you want to use Gemini in, e.g. "my-production-project-1"
* `GEMINI_LOCATION`: the GCP location you want to run Gemini on, e.g. "global"
* `GEMINI_MODEL`: the Gemini model you wish to use, e.g. "gemini-2.5-flash-lite"

Using the Gemini backend needs nothing beyond the base install. The local backend needs `transformers` and `torch`, which are an optional extra:

```
uv sync --extra local
```

and it needs to be able to find the fine-tuned models — set `PASTEL_LOCAL_MODELS_DIR` unless you are running from the repo root with the models under `data/local_models/models`.

### Billing labels

`PastelGemini` takes an optional `labels` dict, which is attached to each Gemini call the model makes so its spend can be separated out in Google Cloud billing:

```python
pastel = PastelGemini.from_dict(weights, labels={"task": "checkworthy_pastel"})
```

These are merged with any `GENAI_LABEL_*` environment variables that `genai_utils` picks up at import time (e.g. `GENAI_LABEL_SERVICE=claims-analysis-api` gives every call a `service` label), so a per-task label here composes with the service-level one rather than replacing it.

Keys must start with a lowercase letter; keys and values can only contain lowercase letters, numbers, `-` and `_`, and must be at most 63 characters. `genai_utils` drops any label that doesn't meet those rules (with a warning) rather than failing the call, so a typo means untagged spend rather than an error.

`PastelLocal` makes no Gemini calls, so it takes no labels.

### A note on data

An example data file, `data/example_training_data.jsonl` is provided so tests and demos can run.
Note that this was generated using Gemini and for copyright reasons is not real news media.
Please provide your own examples.

### Citation
[1] Leite, J. A., Razuvayevskaya, O., Bontcheva, K., & Scarton, C. (2025). [Weakly supervised veracity classification with LLM-predicted credibility signals](https://arxiv.org/abs/2309.07601). EPJ Data Science, 14(1), 16.
