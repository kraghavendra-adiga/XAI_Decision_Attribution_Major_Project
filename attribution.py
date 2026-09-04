"""LIME, SHAP, and TF-IDF attribution explainers for text summarization and question answering.

All three explainers answer the same question: which parts of the input does the
output the user is looking at actually rest on? The explained quantity is
therefore the alignment between a candidate input and the produced summary (or
answer), measured as TF-IDF cosine similarity.

That matters because the panels previously explained a sentiment classifier
(distilbert SST-2) which has nothing to do with summarization: removing words
barely moved its POSITIVE probability, so every weight came back around ±0.005
and the ranking was noise. Perturbing the real summarizer instead would need one
distilbart decode per LIME sample, which is minutes of CPU per request, so the
alignment score stands in for it — a surrogate, but one that responds to exactly
the words the summary was built from.
"""

import numpy as np
import shap
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .text_cleaning import (
        clean_attribution_feature,
        clean_educational_text,
        split_sentences,
    )
except ImportError:
    from text_cleaning import (
        clean_attribution_feature,
        clean_educational_text,
        split_sentences,
    )

_DISPLAY_STOPWORDS = ENGLISH_STOP_WORDS | {
    "ai",
    "response",
    "responses",
    "mistake",
    "mistakes",
    "source",
    "sources",
}
_TOKEN_SPLIT = r"\W+"

# LIME asks for every perturbation in a single call, so the scorer slices the
# batch itself to have something honest to report progress against. 60 keeps the
# bar moving about twenty times across a 1200-sample run without making the
# vectoriser transform calls small enough to lose their batching advantage.
_PROGRESS_CHUNK = 60

# shap's own default for a partition explainer; named here so the progress
# fraction has a real denominator.
_SHAP_MAX_EVALS = 500

__all__ = [
    "_is_display_feature",
    "_alignment_scorer",
    "_aggregate_by_word",
    "get_lime_explanation",
    "get_shap_explanation",
    "get_sentence_attribution",
]


def _is_display_feature(text: str) -> bool:
    feature = text.lower()
    return bool(feature) and not feature.isdigit() and feature not in _DISPLAY_STOPWORDS


def _alignment_scorer(text: str, reference: str):
    """Score how strongly a candidate input supports `reference`.

    The vectorizer is fit once, on the original text and the reference only, and
    never refit. LIME and SHAP both feed thousands of word-masked variants
    through the returned function, and refitting per batch would move the
    vocabulary underneath them and invalidate the explanation.
    """
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    try:
        vec.fit([reference, text])
        ref_matrix = vec.transform([reference])
    except ValueError:
        # Nothing but stop words on one side; no alignment can be measured.
        return None
    if ref_matrix.nnz == 0:
        return None

    def score(candidates):
        matrix = vec.transform([str(c) for c in candidates])
        sims = cosine_similarity(matrix, ref_matrix).ravel()
        return np.clip(sims, 0.0, 1.0)

    return score


def _aggregate_by_word(pairs, limit=60):
    """Sum contributions per distinct word, strongest first.

    SHAP emits one row per token occurrence, so "hole" appearing four times used
    to render as four separate chips competing for the reader's attention.
    """
    totals = {}
    display = {}
    for raw_word, value in pairs:
        word = clean_attribution_feature(str(raw_word))
        if not _is_display_feature(word):
            continue
        key = word.lower()
        totals[key] = totals.get(key, 0.0) + float(value)
        display.setdefault(key, word)
    ranked = sorted(totals.items(), key=lambda kv: -abs(kv[1]))
    return [(display[key], round(total, 4)) for key, total in ranked[:limit]]


def get_lime_explanation(
    text: str, reference: str = "", num_features: int = 10, report=None
):
    """Word-level LIME weights for the input's support of `reference`.

    `report(fraction)` is called with true progress: LIME hands the whole sample
    batch to the scorer in one call, so the scorer walks it in chunks and reports
    how many perturbations it has actually finished rather than guessing.
    """
    try:
        clean_text = clean_educational_text(text)
        clean_reference = clean_educational_text(reference) or clean_text
        if not clean_text:
            return []

        score = _alignment_scorer(clean_text, clean_reference)
        if score is None:
            return []

        num_samples = 1200

        def predict_proba(texts):
            texts = list(texts)
            chunks = []
            for start in range(0, len(texts), _PROGRESS_CHUNK):
                chunks.append(score(texts[start:start + _PROGRESS_CHUNK]))
                if report:
                    report(min(1.0, (start + _PROGRESS_CHUNK) / max(1, len(texts))))
            aligned = np.concatenate(chunks) if chunks else np.zeros(0)
            return np.column_stack([1.0 - aligned, aligned])

        explainer = LimeTextExplainer(
            class_names=["UNUSED", "SUPPORTS_OUTPUT"],
            split_expression=_TOKEN_SPLIT,
            bow=True,
            random_state=0,
        )
        # The whole input is explained, not the first 512 characters: the summary
        # draws on the tail of the text as much as the head. Samples are cheap
        # here (one TF-IDF transform each), so the surrogate gets enough of them
        # to produce stable weights.
        exp = explainer.explain_instance(
            clean_text,
            predict_proba,
            labels=(1,),
            num_features=max(num_features * 3, 30),
            num_samples=num_samples,
        )
        rows = _aggregate_by_word(exp.as_list(label=1), limit=num_features)
        if report:
            report(1.0)
        return [{"word": word, "weight": weight} for word, weight in rows]
    except Exception as e:
        return [{"word": f"[LIME error: {e}]", "weight": 0.0}]


def get_shap_explanation(text: str, reference: str = "", limit: int = 40, report=None):
    """Per-word SHAP values for the input's support of `reference`.

    `report(fraction)` tracks evaluations against `_SHAP_MAX_EVALS`, which is
    passed explicitly so the denominator is known. 500 is also shap's own default
    for a partition explainer, so naming it changes the budget for nobody.
    """
    try:
        clean_text = clean_educational_text(text)
        clean_reference = clean_educational_text(reference) or clean_text
        if not clean_text:
            return []

        score = _alignment_scorer(clean_text, clean_reference)
        if score is None:
            return []

        seen = {"n": 0}

        def counted(candidates):
            values = score(candidates)
            seen["n"] += len(values)
            if report:
                report(min(0.99, seen["n"] / _SHAP_MAX_EVALS))
            return values

        # Default mask token ("..."): TF-IDF's word pattern ignores it, so masking
        # a word is equivalent to deleting it. An empty mask token would be more
        # direct but shap 0.49 cannot build its token hierarchy from one.
        masker = shap.maskers.Text(r"\W+")
        explainer = shap.Explainer(counted, masker, silent=True)
        shap_values = explainer(
            [clean_text], max_evals=_SHAP_MAX_EVALS, silent=True
        )
        tokens = shap_values.data[0]
        values = np.asarray(shap_values.values[0], dtype=float).ravel()
        rows = _aggregate_by_word(zip(tokens, values), limit=limit)
        if report:
            report(1.0)
        return [{"token": token, "value": value} for token, value in rows]
    except Exception as e:
        return [{"token": f"[SHAP error: {e}]", "value": 0.0}]


def get_sentence_attribution(text: str, reference: str = ""):
    """Rank input sentences by how much of `reference` they account for.

    Scores are raw cosine similarities, not values renormalised so the best
    sentence reads 100%. The previous version summed each sentence's TF-IDF row,
    which grows with sentence length and never looked at the output at all, so
    the longest run of text always won regardless of what the summary said.
    """
    clean_text = clean_educational_text(text)
    sentences = split_sentences(clean_text)
    if not sentences:
        return []
    if len(sentences) < 2:
        return [{"sentence": sentences[0], "score": 1.0}]

    try:
        clean_reference = clean_educational_text(reference)
        if clean_reference:
            vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
            matrix = vec.fit_transform([clean_reference] + sentences)
            scores = cosine_similarity(matrix[1:], matrix[0]).ravel()
            scores = np.clip(scores, 0.0, 1.0)
        else:
            # No output to attribute to: fall back to per-sentence TF-IDF mass,
            # normalised so the ranking is still readable.
            vec = TfidfVectorizer(stop_words="english")
            mass = vec.fit_transform(sentences).toarray().sum(axis=1)
            length = np.array([max(1, len(s.split())) for s in sentences], dtype=float)
            scores = mass / np.sqrt(length)
            scores = scores / (scores.max() or 1.0)

        result = [
            {"sentence": sent, "score": round(float(s), 4)}
            for sent, s in zip(sentences, scores)
        ]
        result.sort(key=lambda x: x["score"], reverse=True)
        return result
    except Exception as e:
        return [{"sentence": f"[Attribution error: {e}]", "score": 0.0}]
