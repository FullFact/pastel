# Uses local, pre-trained encoder models to answer questions about sentences.

from pathlib import Path

from local_models.questions import QUESTIONS

MODELS: dict[str, str] = {
    "ModernBERT-multilingual": "jhu-clsp/mmBERT-base",
    "mDeBERTa-v3-base": "microsoft/mdeberta-v3-base",
    "XLM-RoBERTa-base": "FacebookAI/xlm-roberta-base",
}

MODEL_CATEGORY = "ModernBERT-multilingual"  # only using this one for now
RESULTS_DIR = Path(__file__).parent / "results"
MODELS_DIR = Path("data/local_models/models")
MAX_LENGTH = 128
BATCH_SIZE = 32

_model_cache: dict[int, tuple] = {}


def _load_model_for_question(question_index: int) -> tuple:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    question_label = f"q{question_index:02d}"
    checkpoint_dir = MODELS_DIR / MODEL_CATEGORY / question_label

    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    latest = checkpoints[-1]
    print(f"Loading model from {latest}")

    model_id = MODELS[MODEL_CATEGORY]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(str(latest))
    model.eval()
    return model, tokenizer


def preload_models():
    """Load all models into cache to save time later"""
    for question_index in range(0, len(QUESTIONS)):
        _model_cache[question_index] = _load_model_for_question(question_index)


def answer_question(question: str, sentences: list[str]) -> list[float]:
    """
    Answers the question for the given list of sentences.
    Returns one score per sentence, in the same order as the input.
    """
    import torch

    try:
        question_index = QUESTIONS.index(question)
    except ValueError:
        print("Error: unknown question ", question)
        return [0.0] * len(sentences)

    if question_index not in _model_cache:
        _model_cache[question_index] = _load_model_for_question(question_index)

    model, tokenizer = _model_cache[question_index]

    input_text = [question + " " + sentence for sentence in sentences]

    answers: list[float] = []
    for start in range(0, len(input_text), BATCH_SIZE):
        batch = input_text[start : start + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = model(**inputs).logits

        answers.extend(torch.argmax(logits, dim=-1).float().tolist())

    return answers


if __name__ == "__main__":

    preload_models()
    sentences = [
        "Scientists have shown that tamoxifen patients are more likely to develop deadly blood clots and cancer.",
        "Rubbing olive oil onto a lump under your skin will make it disappear in a few days.",
    ]
    answers = {question: answer_question(question, sentences) for question in QUESTIONS}

    for idx, sentence in enumerate(sentences):
        print(f"\n{'*' * 80}\n{sentence}\n")
        for question in QUESTIONS:
            print(f"{question[:60]:60s}  {answers[question][idx]}")
