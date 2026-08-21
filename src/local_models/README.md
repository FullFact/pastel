# Using local models for Pastel

Originally, the Pastel library relied on Gemini to answer a set of questions around each piece of text. While this produced good results, the rising price of Gemini and comparable tools encourages the use of small, local, fine-tuned encoder models.

When a new question is added, we need to create a training set and then use it to fine tune a new model. Each model will only answer one question. We use Gemini to create the training set as a one-off task.

## Choosing a backend

`PastelGemini` and `PastelLocal` are interchangeable implementations of `PastelModel`: both answer the model's questions, one by prompting Gemini and one with the fine-tuned encoders. Pick one at runtime rather than by editing imports:

```python
from pastel import get_backend

backend = get_backend("local")       # or "gemini", or None to read the environment
pastel = backend.load_model("my_model.json")
```

`get_backend(None)` reads the `PASTEL_BACKEND` environment variable and falls back to `gemini`. Every demo script takes a matching `--backend` flag.

`PastelLocal` only accepts questions listed in `questions.py`, because that is the set it has fine-tuned models for. It raises a `ValueError` on construction otherwise, rather than answering with the wrong model.

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

Each model is saved under a short id (`q00`, `q01`, ...). `model_registry.py` allocates those ids and records them in `model_map.json`, and inference looks them up in the same place — so training and inference always agree on which model answers which question. Add the new question to `questions.py` once its model exists, and it becomes available to `PastelLocal`.

When all new questions have been set up, a new Pastel model can be trained with `demo_beam_search.py --backend local`.

## Inference

After training, `local_answerer.py` uses a fine-tuned model to label new sentences. `PastelLocal.preload()` loads a model's encoders up front, which is worth doing before a long batch run or before timing anything.

# Known issues!

* The trained local models will need to be stored in a bucket and downloaded as required. Until then, `model_registry.MODELS_DIR` expects them under `data/local_models/models/`.

* Evaluation of individual local models and the combined Pastel model.

* `questions.py` is a hand-maintained list that has to be kept in step with the models that actually exist on disk. `PastelLocal` will fail at inference time (`FileNotFoundError`) for a question that is listed but never trained.
