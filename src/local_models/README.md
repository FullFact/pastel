# Using local models for Pastel

Originally, the Pastel library relied on Gemini to answer a set of questions around each piece of text. While this produced good results, the rising price of Gemini and comparable tools encourages the use of small, local, fine-tuned encoder models.

When a new question is added, we need to create a training set and then use it to fine tune a new model. Each model will only answer one question. We use Gemini to create the training set as a one-off task.

## Create a training set

The `label_sentences.py` module `main()` function processes a single question. It loads a set of sentences that already have labels for some questions. It then passes the sentences and the new question to Gemini and records the results. The end result is an extra set of labels in the `labelled_sentences.jsonl` file.

## Training a model

A new model can then be trained using `finetune_encode.py`. Pass the question to `build_one_question_answerer()` and it will extract the question and labelled sentences from the training file and fine tune a model.

When all new questions have been set up, a new Pastel model can be trained by adding the questions to `demo_beam_search.py`. 

## Inference

After training, `local_answerer.py` uses a fine-tuned model to label new sentences.

# Known issues!

* The function `pastel.py / _get_answers_for_single_sentence()` can either pass the questions to Gemini (needed when creating a training set) or to the new fine-tuned model (for inference). Currently, this switch is done by commenting out bits of code! In the long run, in production, we'll want to only use Gemini during training. However, the library is open source, so it might want to keep both options with a flag set in the environment to indicate 'gemini or local'.

* The trained local models will need to be stored in a bucket and downloaded as required.

* Evaluation of individual local models and the combined Pastel model.

