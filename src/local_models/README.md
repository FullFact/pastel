# Training local models for Pastel

Originally, the Pastel library relied on Gemini to answer a set of questions around each piece of text. While this produced good results, the rising price of Gemini and comparable tools encourages the use of small, local, fine-tuned encoder models.

When a new question is added, we need to create a training set and then use it to fine tune the model. One encoder answers every question, with a small classification head per question, so adding a question means retraining all of them together. We use Gemini to create the training set as a one-off task.

This package is the *training* half of that, and is not part of the installable library. Everything needed to *use* the resulting model lives in `pastel/local/`: the registry saying which head answers which question, and loading the model for inference.

The library declares no questions of its own. Which questions to ask, and the weights to combine their answers with, belong to the downstream task: pick them, train the weights, and save the result as a Pastel model JSON file.

## Choosing a backend

`PastelGemini` and `PastelLocal` are interchangeable implementations of `PastelModel`: both answer the model's questions, one by prompting Gemini and one with the fine-tuned encoder. Pick one at runtime rather than by editing imports:

```python
from pastel import get_backend

backend = get_backend("local")       # or "gemini", or None to read the environment
pastel = backend.load_model("my_model.json")
```

`get_backend(None)` reads the `PASTEL_BACKEND` environment variable and falls back to `gemini`. Every demo script takes a matching `--backend` flag.

`PastelLocal` only accepts questions the fine-tuned model on disk has a head for. It raises a `ValueError` on construction otherwise, rather than answering with the wrong head.

Installing the library does not pull in `transformers` and `torch` — they are an optional extra, so Gemini-only users don't pay for them:

```
uv sync --extra local          # to use the local models
uv sync --group ml-labeller    # to fine-tune new ones (includes the above)
```

### Which questions can actually be answered?

`model_map.json`, written by training and read by inference, records every question that has a head. Whether the model is really on disk where we expect it is a separate matter:

```
python -m pastel.local
```

That reports each recorded question as OK or MISSING and prints where it looked. In code, `available_questions()` gives the recorded questions once the model has been trained, and `PastelLocal.from_available_questions()` builds a model from exactly those — which is what the demo scripts use. The questions arrive together, since one model answers all of them; with nothing trained, `available_questions()` is empty and `require_available_questions()` raises `FileNotFoundError` saying where it looked.

### Where the models live

`PASTEL_LOCAL_MODELS_DIR` sets the directory holding the fine-tuned model, for both training and inference. It defaults to `data/local_models/models`, which is relative and so only resolves when the working directory is the repo root — set it to an absolute path anywhere else, production included. The model itself is saved under the base model's name, in `multi_head/checkpoint-*`.

## Create a training set

`label_sentences.py` processes a single question. It loads a set of sentences that already have labels for some questions, passes the sentences and the new question to Gemini, and records the results as an extra set of labels in `labelled_sentences.jsonl`.

```
python -m local_models.label_sentences \
    --question "Is this sentence a joke or satirical?" \
    --input data/local_models/fullfact-2026-03-31-claims.json \
    --limit 200        # optional: a cheap smoke test before labelling everything
```

## Training a model

```
python -m local_models.finetune_encoder
```

That trains one model to answer every question the labelled data has answers for. `train_answerer(questions)` does the same for a chosen list, and takes the epochs, batch size and learning rate if you want to change them.

The questions are trained together because the encoder body is shared and only the heads are per-question — nine separate encoders were nine times the memory and nine forward passes per sentence to answer the same nine questions about it. So a retrain replaces the whole model, and it has to cover every question in the map: leaving one out would leave it with a head index the new model has either not trained or trained for something else. `train_answerer()` refuses rather than let that happen; drop a question from `model_map.json` to stop answering it.

Each question gets a head index. `pastel/local/model_registry.py` allocates those and records them in `model_map.json`, and inference looks them up in the same place — so training and inference always agree on which head answers which question. Once training has written those entries, the questions are available to `PastelLocal`; nothing else needs updating.

Sentences don't have to be labelled for every question: a missing answer, and Gemini's unsure answer of `0.5`, are left out of the loss for that head only, so a sentence still trains the heads it does have answers for.

### What the model sees

The input is the sentence on its own. Each head answers one fixed question, so prefixing the question text only spends the forward pass encoding a constant. Training and inference have to agree on this: a model fine-tuned on `question + " " + sentence` has to be retrained before it is used.

### What the map looks like

`scripts/example_questions.json` is an example: the ten questions Full Fact's checkworthiness model uses, mapped to the heads they were trained into. It is there to show the format, not to be used — training writes the real file itself, and a model always ships with its own.

When all new questions have been set up, a new Pastel model can be trained with `demo_beam_search.py --backend local`.

## Inference

After training, `pastel/local/local_answerer.py` uses the fine-tuned model to label new sentences. `answer_questions()` answers every question in one pass of the encoder, so asking all of them costs little more than asking one; `answer_question()` is the single-question form of it. `PastelLocal.preload()` loads the encoder up front, which is worth doing before a long batch run or before timing anything — it also surfaces a missing model, or a question the model has no head for, straight away rather than part-way through a batch.

### Quantising the model

Setting `PASTEL_LOCAL_QUANTISE=1` quantises the model's linear layers to int8 as it is loaded, which is worth roughly 20% of inference time on a CPU with no GPU. It changes the numerics, so it is off by default — re-run the holdout evaluation before trusting a model with it on. It uses [torchao](https://github.com/pytorch/ao), which comes with the `local` extra.

# Known issues!

* The trained local model will need to be stored in a bucket and downloaded as required. Until then, `PASTEL_LOCAL_MODELS_DIR` has to point at a directory that already holds it.

* Evaluation of the individual heads and the combined Pastel model.

* `_load_model` never places the model on a GPU, so inference is CPU-only whatever hardware it runs on.

* Exporting to ONNX Runtime or OpenVINO would probably beat `PASTEL_LOCAL_QUANTISE` on CPU — neither has been measured.
