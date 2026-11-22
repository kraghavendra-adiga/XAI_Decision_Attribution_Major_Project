"""Fine-tune the existing DistilBERT SST-2 checkpoint for evidence relevance."""

import argparse
import json
import random
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from .evidence_benchmark import (
        SCIFACT_URL,
        binary_metrics,
        load_scifact_archive,
        make_evidence_pairs,
        tfidf_similarity_scores,
    )
except ImportError:
    from evidence_benchmark import (
        SCIFACT_URL,
        binary_metrics,
        load_scifact_archive,
        make_evidence_pairs,
        tfidf_similarity_scores,
    )


BASE_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "finetuning_artifacts" / "evidence_alignment"
MAX_LENGTH = 512


def _download_dataset(archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        return
    temporary_path = archive_path.with_suffix(archive_path.suffix + ".part")
    try:
        print(f"Downloading SciFact from {SCIFACT_URL}")
        urllib.request.urlretrieve(SCIFACT_URL, temporary_path)
        temporary_path.replace(archive_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _model_scores(
    model, tokenizer, pairs, device, batch_size: int, label: str
) -> list[float]:
    model.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            inputs = tokenizer(
                [pair.sentence for pair in batch],
                [pair.claim for pair in batch],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            logits = model(**inputs).logits
            scores.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
            if start % (batch_size * 25) == 0:
                print(
                    f"{label}: {min(start + len(batch), len(pairs))}/{len(pairs)} pairs",
                    flush=True,
                )
    return scores


def _train_epoch(model, tokenizer, pairs, optimizer, device, batch_size, rng):
    order = list(range(len(pairs)))
    rng.shuffle(order)
    model.train()
    running_loss = 0.0
    batches = 0
    for start in range(0, len(order), batch_size):
        batch = [pairs[index] for index in order[start : start + batch_size]]
        inputs = tokenizer(
            [pair.sentence for pair in batch],
            [pair.claim for pair in batch],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        labels = torch.tensor([pair.label for pair in batch], device=device)
        optimizer.zero_grad(set_to_none=True)
        output = model(**inputs, labels=labels)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += float(output.loss.detach().cpu())
        batches += 1
    return running_loss / max(1, batches)


def run_experiment(
    *,
    archive_path: Path,
    output_dir: Path,
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    max_train_pairs: int | None = None,
    max_negatives_per_claim: int = 4,
    device_name: str = "auto",
) -> dict:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    _download_dataset(archive_path)
    corpus, splits = load_scifact_archive(archive_path)
    train_pairs = make_evidence_pairs(
        splits["train"],
        corpus,
        max_negatives_per_claim=max_negatives_per_claim,
        seed=42,
    )
    dev_pairs = make_evidence_pairs(
        splits["dev"],
        corpus,
        max_negatives_per_claim=max_negatives_per_claim,
        seed=43,
    )
    if max_train_pairs:
        train_pairs = train_pairs[:max_train_pairs]

    device = (
        torch.device("cuda" if torch.cuda.is_available() else
                     "mps" if torch.backends.mps.is_available() else "cpu")
        if device_name == "auto"
        else torch.device(device_name)
    )
    print(
        f"Training on {device}; {len(train_pairs)} train and "
        f"{len(dev_pairs)} dev pairs.",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL)
    if model.config.num_labels != 2:
        raise ValueError(f"Expected a binary checkpoint, got {model.config.num_labels} labels.")
    model.config.id2label = {0: "NOT_EVIDENCE", 1: "EVIDENCE"}
    model.config.label2id = {"NOT_EVIDENCE": 0, "EVIDENCE": 1}
    model.to(device)

    train_labels = [pair.label for pair in train_pairs]
    train_tfidf_scores = tfidf_similarity_scores(train_pairs)
    threshold = max(
        set(train_tfidf_scores),
        key=lambda value: f1_score(
            train_labels,
            [score >= value for score in train_tfidf_scores],
            zero_division=0,
        ),
    )
    dev_tfidf_scores = tfidf_similarity_scores(dev_pairs)
    tfidf_result = binary_metrics(
        [pair.label for pair in dev_pairs],
        dev_tfidf_scores,
        threshold=float(threshold),
    )
    tfidf_result.update({
        "threshold": float(threshold),
        "threshold_source": "Training split; maximum F1",
    })

    dev_labels = [pair.label for pair in dev_pairs]
    print("Evaluating pretrained SST-2 checkpoint...", flush=True)
    pretrained_scores = _model_scores(
        model, tokenizer, dev_pairs, device, batch_size, "Pretrained evaluation"
    )
    pretrained_result = binary_metrics(dev_labels, pretrained_scores)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    rng = random.Random(42)
    epoch_losses = []
    for epoch in range(epochs):
        print(f"Training epoch {epoch + 1}/{epochs}...", flush=True)
        loss = _train_epoch(
            model, tokenizer, train_pairs, optimizer, device, batch_size, rng
        )
        epoch_losses.append(loss)
        print(f"Epoch {epoch + 1}/{epochs}: loss={loss:.4f}", flush=True)

    print("Evaluating fine-tuned checkpoint...", flush=True)
    finetuned_scores = _model_scores(
        model, tokenizer, dev_pairs, device, batch_size, "Fine-tuned evaluation"
    )
    finetuned_result = binary_metrics(dev_labels, finetuned_scores)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    categorized_samples = {
        "corrected": [],
        "unchanged": [],
        "regressed": [],
        "both_incorrect": [],
    }
    for pair, tfidf_score, before, after in zip(
        dev_pairs, dev_tfidf_scores, pretrained_scores, finetuned_scores
    ):
        before_prediction = int(before >= 0.5)
        after_prediction = int(after >= 0.5)
        if before_prediction != pair.label and after_prediction == pair.label:
            category = "corrected"
        elif before_prediction == pair.label and after_prediction != pair.label:
            category = "regressed"
        elif before_prediction == pair.label and after_prediction == pair.label:
            category = "unchanged"
        else:
            category = "both_incorrect"
        categorized_samples[category].append({
            "claim": pair.claim,
            "sentence": pair.sentence,
            "label": "relevant evidence" if pair.label else "not annotated as evidence",
            "label_value": pair.label,
            "tfidf_score": round(float(tfidf_score), 4),
            "pretrained_score": round(float(before), 4),
            "finetuned_score": round(float(after), 4),
        })
    samples = []
    for category in ("corrected", "unchanged", "regressed", "both_incorrect"):
        samples.extend(categorized_samples[category][:10])

    results = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "SciFact",
            "url": "https://github.com/allenai/scifact",
            "description": (
                "Expert-written scientific claims with annotated supporting or "
                "contradicting evidence sentences."
            ),
            "train_pairs": len(train_pairs),
            "validation_pairs": len(dev_pairs),
            "validation_claims": len(splits["dev"]),
            "labels": {
                "positive": "Annotated support/contradiction rationale: relevant evidence",
                "negative": "Sentence in a cited abstract without an evidence annotation",
                "training_examples": (
                    "Evidence sentences paired with their scientific claim are positive. "
                    "Other sentences in cited abstracts paired with the claim are negative."
                ),
            },
        },
        "base_model": BASE_MODEL,
        "finetuned_model": str(model_dir.relative_to(output_dir)),
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": str(device),
            "epoch_losses": epoch_losses,
            "max_length": MAX_LENGTH,
            "max_negatives_per_claim": max_negatives_per_claim,
            "seed": 42,
        },
        "metrics": {
            "production_tfidf": tfidf_result,
            "pretrained_sst2": pretrained_result,
            "finetuned_evidence": finetuned_result,
        },
        "samples": samples,
        "limitations": [
            "SciFact is a small, scientific-claims dataset; it does not establish "
            "general improvement on every subject or input modality.",
            "DistilBERT accepts at most 512 tokens per pair; longer candidate passages "
            "are truncated by the tokenizer during scoring.",
            "SciFact test labels are not public; metrics use its labeled dev split.",
            "The app lets users compare the pretrained and fine-tuned DistilBERT "
            "evidence scorers for text summaries and Q&A.",
            "The fine-tuned classifier scores source-text/reference pairs; it changes "
            "attribution, not the generated summary or answer.",
            "An unannotated abstract sentence may still be relevant; labels reflect "
            "the dataset's evidence annotations, not universal truth.",
        ],
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved model to {model_dir}")
    print(f"Saved benchmark results to {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(__file__).resolve().parent / "finetuning_artifacts" / "scifact.tar.gz",
        help="Local SciFact archive path (downloaded automatically if absent).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-train-pairs", type=int)
    parser.add_argument("--max-negatives-per-claim", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    args = parser.parse_args()
    run_experiment(
        archive_path=args.archive,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_train_pairs=args.max_train_pairs,
        max_negatives_per_claim=args.max_negatives_per_claim,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
