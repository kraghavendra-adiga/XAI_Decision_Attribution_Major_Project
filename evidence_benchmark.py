"""Data loading and evaluation helpers for the evidence-alignment experiment."""

import json
import random
import tarfile
from pathlib import Path
from typing import NamedTuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"


class EvidencePair(NamedTuple):
    claim: str
    sentence: str
    label: int


def read_jsonl(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_scifact_archive(archive_path: Path) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """Read only the corpus and labeled claims from the official SciFact archive."""
    wanted = {"corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl"}
    files: dict[str, str] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if Path(member.name).name not in wanted or not member.isfile():
                continue
            stream = archive.extractfile(member)
            if stream is None:
                continue
            files[Path(member.name).name] = stream.read().decode("utf-8")

    missing = wanted - files.keys()
    if missing:
        raise ValueError(f"SciFact archive is missing required files: {sorted(missing)}")

    corpus_rows = read_jsonl(files["corpus.jsonl"])
    corpus = {str(row["doc_id"]): row for row in corpus_rows}
    claims = {
        "train": read_jsonl(files["claims_train.jsonl"]),
        "dev": read_jsonl(files["claims_dev.jsonl"]),
    }
    return corpus, claims


def make_evidence_pairs(
    claims: list[dict],
    corpus: dict[str, dict],
    *,
    max_negatives_per_claim: int = 4,
    seed: int = 42,
) -> list[EvidencePair]:
    """Create support-evidence vs non-evidence sentence pairs from SciFact labels.

    Both SUPPORT and CONTRADICT rationales are positive: each is relevant evidence
    for evaluating the claim. Non-rationale sentences in cited abstracts are
    negative examples. Negatives are sampled deterministically to limit imbalance.
    """
    rng = random.Random(seed)
    pairs: list[EvidencePair] = []
    for record in claims:
        claim = str(record.get("claim", "")).strip()
        if not claim:
            continue

        evidence: dict[str, set[int]] = {}
        for doc_id, rationales in (record.get("evidence") or {}).items():
            evidence[str(doc_id)] = {
                int(index)
                for rationale in rationales
                for index in rationale.get("sentences", [])
            }

        negative_candidates: list[str] = []
        for doc_id in record.get("cited_doc_ids", []):
            document = corpus.get(str(doc_id))
            if not document:
                continue
            sentences = document.get("abstract") or []
            positive_indices = evidence.get(str(doc_id), set())
            for index, sentence in enumerate(sentences):
                sentence = str(sentence).strip()
                if not sentence:
                    continue
                if index in positive_indices:
                    pairs.append(EvidencePair(claim, sentence, 1))
                else:
                    negative_candidates.append(sentence)

        rng.shuffle(negative_candidates)
        for sentence in negative_candidates[:max_negatives_per_claim]:
            pairs.append(EvidencePair(claim, sentence, 0))

    if not pairs or not any(pair.label == 1 for pair in pairs) or not any(
        pair.label == 0 for pair in pairs
    ):
        raise ValueError("SciFact did not produce both positive and negative pairs.")
    return pairs


def binary_metrics(
    labels: list[int], scores: list[float], *, threshold: float = 0.5
) -> dict[str, float]:
    """Return comparable binary metrics and ROC AUC for continuous scores."""
    y_true = np.asarray(labels, dtype=int)
    y_score = np.asarray(scores, dtype=float)
    if len(y_true) == 0 or len(y_true) != len(y_score):
        raise ValueError("Expected equally sized, non-empty labels and scores.")
    if len(np.unique(y_true)) != 2:
        raise ValueError("Metrics require both evidence and non-evidence examples.")
    predictions = y_score >= threshold
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "count": int(len(y_true)),
    }


def tfidf_similarity_scores(pairs: list[EvidencePair]) -> list[float]:
    """Score claim/evidence relevance using the production explainer's TF-IDF basis."""
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    texts = [pair.sentence for pair in pairs]
    claims = [pair.claim for pair in pairs]
    matrix = vectorizer.fit_transform(texts + claims)
    sentence_matrix = matrix[: len(pairs)]
    claim_matrix = matrix[len(pairs) :]
    numerator = np.asarray(sentence_matrix.multiply(claim_matrix).sum(axis=1)).ravel()
    sentence_norm = np.sqrt(np.asarray(sentence_matrix.multiply(sentence_matrix).sum(axis=1)).ravel())
    claim_norm = np.sqrt(np.asarray(claim_matrix.multiply(claim_matrix).sum(axis=1)).ravel())
    denominator = sentence_norm * claim_norm
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0,
    ).tolist()
