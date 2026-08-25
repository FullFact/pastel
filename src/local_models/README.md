# Training local models for Pastel

Originally, the Pastel library relied on Gemini to answer a set of questions around each piece of text. While this produced good results, the rising price of Gemini and comparable tools encourages the use of small, local, fine-tuned encoder models.

When a new question is added, we need to create a training set and then use it to fine tune a new model. Each model will only answer one question. We use Gemini to create the training set as a one-off task.

This package is the *training* half of that, and is not part of the installable library. Everything needed to *use* the resulting models lives in `pastel/local/`: the question list, the registry saying which model answers which question, and loading them for inference.

## Choosing a backend

`PastelGemini` and `PastelLocal` are interchangeable implementations of `PastelModel`: both answer the model's questions, one by prompting Gemini and one with the fine-tuned encoders. Pick one at runtime rather than by editing imports:

```python
from pastel import get_backend

backend = get_backend("local")       # or "gemini", or None to read the environment
pastel = backend.load_model("my_model.json")
```

`get_backend(None)` reads the `PASTEL_BACKEND` environment variable and falls back to `gemini`. Every demo script takes a matching `--backend` flag.

`PastelLocal` only accepts questions listed in `pastel/local/questions.py`, because that is the set it is *meant* to have models for. It raises a `ValueError` on construction otherwise, rather than answering with the wrong model.

Installing the library does not pull in `transformers` and `torch` — they are an optional extra, so Gemini-only users don't pay for them:

```
uv sync --extra local          # to use the local models
uv sync --group ml-labeller    # to fine-tune new ones (includes the above)
```

### Which questions can actually be answered?

`questions.py` is a hand-maintained declaration. Whether a question's model has really been trained, and is where we expect to find it, is a separate matter:

```
python -m pastel.local
```

That reports each declared question as OK or MISSING, flags any trained model whose question was never added to `questions.py`, and prints where it looked. In code, `available_questions()` gives the declared questions with a trained model, and `PastelLocal.from_available_questions()` builds a model from exactly those — which is what the demo scripts use, so they work with a partly-trained set. Answering a declared-but-untrained question raises `FileNotFoundError` naming the question.

### Where the models live

`PASTEL_LOCAL_MODELS_DIR` sets the directory holding the fine-tuned models, for both training and inference. It defaults to `data/local_models/models`, which is relative and so only resolves when the working directory is the repo root — set it to an absolute path anywhere else, production included.

## Create a training set

`label_sentences.py` processes a single question. It loads a set of sentences that already have labels for some questions, passes the sentences and the new question to Gemini, and records the results as an extra set of labels in `labelled_sentences.jsonl`.

```
python -m local_models.label_sentences \
    --question "Is this sentence a joke or satirical?" \
    --input data/local_models/fullfact-2026-03-31-claims.json \
    --limit 200        # optional: a cheap smoke test before labelling everything
```

## Training a model

A new model can then be trained using `finetune_encoder.py`. Pass the question to `build_one_question_answerer()` and it will extract the question and labelled sentences from the training file and fine tune a model.

Each model is saved under a short id (`q00`, `q01`, ...). `pastel/local/model_registry.py` allocates those ids and records them in `model_map.json`, and inference looks them up in the same place — so training and inference always agree on which model answers which question. Add the new question to `pastel/local/questions.py` once its model exists, and it becomes available to `PastelLocal`.

When all new questions have been set up, a new Pastel model can be trained with `demo_beam_search.py --backend local`.

## Inference

After training, `pastel/local/local_answerer.py` uses a fine-tuned model to label new sentences. `PastelLocal.preload()` loads a model's encoders up front, which is worth doing before a long batch run or before timing anything — it also surfaces a missing model straight away rather than part-way through a batch.

# Known issues!

* The trained local models will need to be stored in a bucket and downloaded as required. Until then, `PASTEL_LOCAL_MODELS_DIR` has to point at a directory that already holds them.

* Evaluation of individual local models and the combined Pastel model.
