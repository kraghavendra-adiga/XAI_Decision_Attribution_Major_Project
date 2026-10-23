"""Lazy DistilBERT evidence-relevance inference for text attribution."""

import threading
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
FINETUNED_MODEL = (
    Path(__file__).resolve().parent
    / "finetuning_artifacts"
    / "evidence_alignment"
    / "model"
)
MAX_LENGTH = 512
INFERENCE_BATCH_SIZE = 16

_MODELS: dict[str, tuple[object, object, torch.device]] = {}
_MODEL_LOCK = threading.RLock()


def finetuned_model_available() -> bool:
    return (FINETUNED_MODEL / "config.json").is_file() and (
        (FINETUNED_MODEL / "model.safetensors").is_file()
        or (FINETUNED_MODEL / "pytorch_model.bin").is_file()
    )


def require_finetuned_model() -> None:
    if not finetuned_model_available():
        raise FileNotFoundError(
            "The fine-tuned evidence model is not installed yet. Run "
            "`venv/bin/python finetune_evidence.py` from the xai folder."
        )


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model(model_type: str):
    if model_type not in {"pretrained_sst2", "finetuned_sst2"}:
        raise ValueError(f"Unknown attribution model: {model_type}")
    if model_type == "finetuned_sst2":
        require_finetuned_model()
        source = str(FINETUNED_MODEL)
    else:
        source = BASE_MODEL

    with _MODEL_LOCK:
        loaded = _MODELS.get(model_type)
        if loaded is None:
            device = _device()
            tokenizer = AutoTokenizer.from_pretrained(source)
            model = AutoModelForSequenceClassification.from_pretrained(source)
            model.to(device)
            model.eval()
            loaded = (tokenizer, model, device)
            _MODELS[model_type] = loaded
        return loaded


def score_evidence_pairs(
    candidates: list[str],
    reference: str,
    *,
    model_type: str,
    report=None,
) -> np.ndarray:
    """Return P(relevant evidence) for each source-text/reference pair."""
    if model_type == "finetuned_sst2":
        require_finetuned_model()
    if model_type not in {"pretrained_sst2", "finetuned_sst2"}:
        raise ValueError(f"Unknown classifier attribution model: {model_type}")

    scores = np.zeros(len(candidates), dtype=float)
    valid_indices = [i for i, text in enumerate(candidates) if str(text or "").strip()]
    if not valid_indices:
        return scores

    tokenizer, model, device = _load_model(model_type)
    with _MODEL_LOCK, torch.inference_mode():
        for start in range(0, len(valid_indices), INFERENCE_BATCH_SIZE):
            indexes = valid_indices[start : start + INFERENCE_BATCH_SIZE]
            texts = [str(candidates[index]) for index in indexes]
            inputs = tokenizer(
                texts,
                [reference] * len(texts),
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            positive = torch.softmax(model(**inputs).logits, dim=-1)[:, 1]
            scores[indexes] = positive.detach().cpu().numpy()
            if report:
                report(min(1.0, (start + len(indexes)) / len(valid_indices)))
    return np.clip(scores, 0.0, 1.0)


def attribution_model_label(model_type: str) -> str:
    labels = {
        "tfidf": "Current TF-IDF",
        "pretrained_sst2": "Pretrained DistilBERT SST-2",
        "finetuned_sst2": "Fine-tuned DistilBERT evidence model",
    }
    try:
        return labels[model_type]
    except KeyError as exc:
        raise ValueError(f"Unknown attribution model: {model_type}") from exc
