# PASTEL

This is a concept from Sheffield University, where the prompt consists of a series of yes/no questions. The answers to these questions, in the context of a piece of text, are then combined into a single score using a linear regression model.

The `pastel/optimise_weights.py` module calculates the parameters of the regression model, and requires a list of sentences with associated checkworthy scores. 

The `pastel/pastel.py` module passes the text and questions to a genAI model and uses the regression model to calculate a single score.

Currently, this is used by the genai-checkworthy repo but in the future, the same approach might be used to analyse text for other features such as propaganda, bias, reliability etc.

`training/cached_pastel.py` uses a local SQLite database to cache Gemini's responses. This saves a lot of time and effort when re-analysing the same sentences over and over again, so is useful for experimenting with/optimising Pastel models, but should not be used in production. (It won't help there anyway, as each sentence is only ever seen once.) Similarly, `training/crossvalidate_pastel.py` and `training/beam_search.py` are scripts to compare a large number of Pastel models (potentially millions!) to help find a good combination of questions. `beam_search` uses heuristics and is a lot faster. There is a sample database of cached answers in `data/sample_responses.db` that can be used to initialise the DatabaseManager.

### Pastel Functions and Claim Types

The `pastel_functions` module defines a set of functions that return a true/false value for a single sentence. One current use is for claim types with functions such as `is_claim_type_quantity`, which allows Pastel models to give higher (or lower) scores to quantity-type sentences. To make this work, sentences must specify the list of claim types as part of a Sentence class (see `pastel/models.py`). If sentences without claim types are used, then any claim type function in a Pastel model will treat the sentence as NOT having any claim types, which will lead to poor performance. So it's important to only use claim-type functions in Pastel models deployed to platforms that have claim-types added to each sentence.

## Setup

If you don't want to manually specify the config of Gemini, you should set the following environment variables:
* `GEMINI_PROJECT`: the GCP project you want to use Gemini in, e.g. "my-production-project-1"
* `GEMINI_LOCATION`: the GCP location you want to run Gemini on, e.g. "global"
* `GEMINI_MODEL`: the Gemini model you wish to use, e.g. "gemini-2.5-flash-lite"

### A note on data

An example data file, `data/example_training_data.jsonl` is provided so tests and demos can run.
Note that this was generated using Gemini and for copyright reasons is not real news media.
Please provide your own examples.