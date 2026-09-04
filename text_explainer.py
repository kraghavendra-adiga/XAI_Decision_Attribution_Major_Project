"""Text QA that answers in whole statements, and a heat map over the passage.

Two problems with answering from `answer_question()` alone:

* A span is often the wrong shape for the question. "What is a glycolytic
  product" came back as "A mitochondrion" at 1.7% confidence, because the span
  with the best start/end logits does not have to be about the question at all,
  and nothing downstream noticed that 1.7% means "no idea".
* Even a correct span ("adenosine triphosphate") is not an answer anybody can
  read; the sentence it sits in is.

So the answer is composed from statements: Gemini first, the same way the image
tab does it, and a TF-IDF retrieval over the passage's own sentences when Gemini
is unavailable — the free tier allows 20 calls a day, so the fallback is a normal
condition, not an error path. Either way the cited sources are mapped back onto
character ranges of the passage, which is what `build_source_heat()` turns into
the same blue -> red ramp `_build_heatmap()` paints on images.
"""

import re
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .model_utils import answer_question
    from .text_cleaning import clean_educational_text, split_sentences_spans
    from .utils import (
        INTERROGATIVES,
        _extract_keywords,
        describe_service_error,
        gemini_available,
        gemini_model,
        is_exhausted_for_today,
        not_answerable_rules,
        parse_json_block,
        question_is_off_topic,
        reply_is_unsupported,
        unsupported_reply,
    )
except ImportError:  # pragma: no cover - fallback for direct script execution
    from model_utils import answer_question
    from text_cleaning import clean_educational_text, split_sentences_spans
    from utils import (
        INTERROGATIVES,
        _extract_keywords,
        describe_service_error,
        gemini_available,
        gemini_model,
        is_exhausted_for_today,
        not_answerable_rules,
        parse_json_block,
        question_is_off_topic,
        reply_is_unsupported,
        unsupported_reply,
    )

__all__ = [
    "explain_text_qa",
    "build_source_heat",
    "heat_from_sentence_scores",
]

# Question words carry no retrieval signal but do match "what" in the passage,
# so they are dropped before sentences are scored against the question. Shared
# with the other tabs, which need the same list to tell what a question is about.
_INTERROGATIVES = INTERROGATIVES

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*")

# "What is X", "define X", "what does X mean" — questions that want the sentence
# where X is introduced, not the sentence that mentions X most often.
_DEFINITIONAL_RE = re.compile(
    r"^\s*(?:what|which)\s+(?:is|are|was|were)\b"
    r"|^\s*(?:define|describe|explain)\b"
    r"|\bmean(?:s|ing)?\b",
    re.IGNORECASE,
)

# Gemini sees the cleaned passage, so its quotes line up with the text the heat
# map paints. Long passages are capped rather than rejected.
_MAX_PROMPT_CHARS = 20000

# A statement shorter than this is a heading or a fragment, not an answer.
_MIN_ANSWER_WORDS = 5

_PROMPT = """You are answering a question using ONLY the passage below.

Return ONLY a valid JSON object, no markdown, no extra text:
{{
  "answerable": true,
  "answer": "your complete answer here",
  "sources": [
    {{
      "text": "exact sentence copied verbatim from the passage",
      "contribution": 0.6,
      "reason": "why this sentence supports the answer"
    }},
    {{
      "text": "another verbatim sentence from the passage",
      "contribution": 0.4,
      "reason": "why this one contributed"
    }}
  ]
}}

Rules:
- The answer must be a complete, self-contained explanation of two to five
  sentences, not a two-word extract. Define the term the question asks about,
  then state what the passage says about it.
- If the question asks for several things, cover all of them.
- "text" must be copied verbatim from the passage. Never paraphrase inside it.
- contribution values must add up to 1.0. Between 2 and 5 sources.
- Never add information that is not in the passage.
{not_answerable}

PASSAGE:
{passage}

QUESTION: {question}
"""


def _words_with_spans(text: str) -> list:
    return [(m.group(0).lower(), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def _question_terms(question: str) -> set:
    return {t for t in _extract_keywords(question) if t not in _INTERROGATIVES}


def _containment(sentence: str, quote: str) -> float:
    """Share of the quote's content words that this sentence carries.

    Containment, not Jaccard: a two-word quote inside a thirty-word sentence is
    a full match, and Jaccard would score it 0.06 and drop the only line the
    answer actually used.
    """
    quote_terms = _extract_keywords(quote)
    if not quote_terms:
        return 0.0
    sentence_terms = _extract_keywords(sentence)
    return len(sentence_terms & quote_terms) / len(quote_terms)


def _align_quote(sentence: str, quote: str):
    """Character range of `quote` inside `sentence`, or None.

    Gemini quotes verbatim most of the time but not always, so a plain substring
    search misses. The window that shares the most words with the quote is taken
    instead, then trimmed back to its first and last matching word so the
    highlight does not run into the punctuation on either side.
    """
    sentence_words = _words_with_spans(sentence)
    quote_words = [word for word, _, _ in _words_with_spans(quote)]
    if not sentence_words or not quote_words:
        return None

    wanted = set(quote_words)
    size = min(len(quote_words), len(sentence_words))
    best_hits, best_start = 0, 0
    for start in range(len(sentence_words) - size + 1):
        hits = sum(1 for word, _, _ in sentence_words[start : start + size] if word in wanted)
        if hits > best_hits:
            best_hits, best_start = hits, start
    if best_hits < max(1, int(0.6 * size)):
        return None

    left, right = best_start, best_start + size
    while left < right and sentence_words[left][0] not in wanted:
        left += 1
    while right > left and sentence_words[right - 1][0] not in wanted:
        right -= 1
    if left >= right:
        return None
    return sentence_words[left][1], sentence_words[right - 1][2]


def _runs(char_heat: np.ndarray) -> list:
    """Run-length encode per-character heat into spans the frontend can wrap.

    Quantised to 5% steps first: without it a blurred score produces one <span>
    per character and the passage renders as tens of thousands of nodes.
    """
    levels = np.rint(np.clip(char_heat, 0.0, 1.0) * 20).astype(np.int16)
    if levels.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(levels)) + 1
    edges = np.concatenate(([0], breaks, [levels.size]))
    return [
        {
            "start": int(edges[i]),
            "end": int(edges[i + 1]),
            "score": round(float(levels[edges[i]]) / 20.0, 3),
        }
        for i in range(len(edges) - 1)
    ]


def build_source_heat(text: str, sources: list) -> dict:
    """Map cited sources onto character ranges of `text`.

    Heat is graded against the largest contribution rather than against 1.0, the
    same way the image path does it: five sources cited at 0.2 each would
    otherwise all sit at the blue end of the ramp and the map would read as
    uniformly weak.

    The whole statement gets a base glow and the matched phrase inside it gets
    the full score, so a reader can see both which sentence was used and which
    part of it carried the answer.
    """
    passage = str(text or "")
    spans = split_sentences_spans(passage)
    if not passage or not spans:
        return {"text": passage, "spans": [], "sentences": []}

    contributions = [float(s.get("contribution", 0.0) or 0.0) for s in (sources or [])]
    # Contributions arrive as percentages from the API layer and as fractions
    # straight from Gemini; both are normalised against their own maximum.
    strongest = max(contributions, default=0.0) or 1.0

    char_heat = np.zeros(len(passage), dtype=np.float32)
    sentence_heat = np.zeros(len(spans), dtype=np.float32)

    for source, contribution in zip(sources or [], contributions):
        quote = str(source.get("text", "") or "")
        if not quote.strip():
            continue
        weight = 0.45 + 0.55 * min(1.0, contribution / strongest)

        matches = [_containment(sentence, quote) for sentence, _, _ in spans]
        best = max(matches, default=0.0)
        if best <= 0.0:
            continue
        # Keep the clear matches plus anything close to the best one: a source
        # that spans a two-sentence definition must light both halves.
        chosen = [i for i, ratio in enumerate(matches) if ratio >= max(0.5, 0.75 * best)]
        if not chosen:
            chosen = [int(np.argmax(matches))]

        for index in chosen:
            sentence, start, end = spans[index]
            score = min(1.0, matches[index]) * weight
            sentence_heat[index] = max(float(sentence_heat[index]), score)

            phrase = _align_quote(sentence, quote)
            if phrase is None:
                char_heat[start:end] = np.maximum(char_heat[start:end], score)
            else:
                char_heat[start:end] = np.maximum(char_heat[start:end], score * 0.5)
                left, right = start + phrase[0], start + phrase[1]
                char_heat[left:right] = np.maximum(char_heat[left:right], score)

    return {
        "text": passage,
        "spans": _runs(char_heat),
        "sentences": [
            {
                "text": sentence,
                "start": start,
                "end": end,
                "score": round(float(sentence_heat[i]), 4),
            }
            for i, (sentence, start, end) in enumerate(spans)
        ],
    }


def heat_from_sentence_scores(text: str, attribution: list) -> dict:
    """Heat map for the summarize path, from TF-IDF sentence attribution.

    get_sentence_attribution() deliberately reports raw cosine similarities, so
    the best sentence in a long passage often reads 0.28. Those numbers stay raw
    in the sentence list, but the ramp is normalised against the strongest one —
    an unnormalised map would be entirely dark blue and say nothing.
    """
    passage = str(text or "")
    spans = split_sentences_spans(passage)
    if not passage or not spans:
        return {"text": passage, "spans": [], "sentences": []}

    scored = {
        str(row.get("sentence", "")).strip(): float(row.get("score", 0.0) or 0.0)
        for row in (attribution or [])
    }
    peak = max(scored.values(), default=0.0) or 1.0

    char_heat = np.zeros(len(passage), dtype=np.float32)
    sentences = []
    for sentence, start, end in spans:
        score = min(1.0, scored.get(sentence, 0.0) / peak)
        char_heat[start:end] = score
        sentences.append(
            {"text": sentence, "start": start, "end": end, "score": round(score, 4)}
        )

    return {"text": passage, "spans": _runs(char_heat), "sentences": sentences}


def _readable_note(error) -> str:
    """Turn a hosted-model failure into a line the reader can act on.

    The common case is not a bug — the hosted tier has a request allowance, so a
    quota message is expected. The exception text goes to the log; the sentence
    shown to the reader comes from the shared classifier in utils, so every tab
    words the same failure the same way.
    """
    return f"{describe_service_error(error)} This answer came from the passage itself."


def _gemini_answer(question: str, passage: str):
    """Ask Gemini for an answer plus verbatim sources. None when unavailable."""
    if not gemini_available():
        return None, (
            "The hosted analysis service is not configured, so this answer came "
            "from the passage itself."
        )

    prompt = _PROMPT.format(
        passage=passage[:_MAX_PROMPT_CHARS],
        question=question,
        not_answerable=not_answerable_rules("passage"),
    )
    model = gemini_model()
    last_error = None
    for attempt in range(2):
        try:
            response = model.generate_content(prompt)
            return parse_json_block(response.text), None
        except Exception as exc:  # noqa: BLE001 - any failure falls back to local
            last_error = exc
            message = str(exc).lower()
            # A per-minute quota clears on its own; the per-day one does not, and
            # sleeping on it only makes the user wait for the same 429. The daily
            # marker has no space in it, so the shared test squashes punctuation
            # out rather than looking for the words "per day".
            transient = "429" in message or "quota" in message
            if not transient or is_exhausted_for_today(exc) or attempt == 1:
                break
            time.sleep(5)
    print(f"[Gemini text] unavailable: {type(last_error).__name__}: {last_error}")
    return None, _readable_note(last_error)


def _is_definitional(question: str) -> bool:
    return bool(_DEFINITIONAL_RE.search(str(question or "")))


def _defines(sentence: str, terms: set) -> bool:
    """Does this sentence read like a definition of one of the question's terms?

    "What is ATP" ranked "the mitochondrion has many other functions in addition
    to the production of ATP" first on TF-IDF alone: short, and every content
    word matches. The sentence that actually defines ATP — "...generate adenosine
    triphosphate (ATP), which is used throughout the cell as a source of chemical
    energy" — is longer and scores worse, so the defining construction has to be
    recognised explicitly rather than left to term frequency.
    """
    lowered = sentence.lower()
    for term in terms:
        escaped = re.escape(term)
        patterns = (
            rf"\b{escaped}\b\s*(?:\([^)]*\))?\s*(?:is|are|was|were|refers? to|means?|denotes?)\b",
            rf"\b(?:called|known as|termed|nicknamed)\s+(?:the\s+|a\s+|an\s+)?[\"“']?{escaped}\b",
            rf"\b{escaped}\s*\(",  # a term glossed in parentheses right after it
            rf"\(\s*{escaped}\s*\)",  # ... or the acronym for the phrase before it
            rf"\b{escaped}\b\s*,\s*(?:which|who|a|an|the)\b",
        )
        if any(re.search(pattern, lowered) for pattern in patterns):
            return True
    return False


def _rank_sentences(question: str, spans: list, span_text: str = "") -> np.ndarray:
    """Score every sentence for how well it answers the question.

    Cosine alone is dominated by sentence length; coverage alone ignores how rare
    a term is. "glycolytic" is rare and decisive, so it needs the TF-IDF half;
    "products" is common, so it needs the coverage half to not outvote it. Two
    bonuses ride on top: a defining construction for the term being asked about,
    and agreement with the extractive model when that model is actually sure.
    """
    sentences = [sentence for sentence, _, _ in spans]
    cosine = np.zeros(len(sentences), dtype=np.float32)
    try:
        vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        matrix = vectorizer.fit_transform([question] + sentences)
        cosine = cosine_similarity(matrix[1:], matrix[0]).ravel().astype(np.float32)
    except ValueError:
        # Nothing but stop words in the question; coverage carries the ranking.
        pass

    terms = _question_terms(question)
    if terms:
        coverage = np.array(
            [len(_extract_keywords(s) & terms) / len(terms) for s in sentences],
            dtype=np.float32,
        )
    else:
        coverage = np.zeros(len(sentences), dtype=np.float32)

    score = 0.6 * cosine + 0.4 * coverage

    if terms and _is_definitional(question):
        score += 0.25 * np.array(
            [1.0 if _defines(s, terms) else 0.0 for s in sentences], dtype=np.float32
        )

    if span_text:
        score += 0.2 * np.array(
            [_containment(s, span_text) for s in sentences], dtype=np.float32
        )

    # Section headings ("Structure", "Outer membrane") and stray fragments match
    # a keyword perfectly and state nothing, so they must not be quotable as an
    # answer. They stay in the passage and in the heat map, just never as sources.
    viable = np.array(
        [len(_WORD_RE.findall(s)) >= _MIN_ANSWER_WORDS for s in sentences],
        dtype=np.float32,
    )
    if viable.any():
        score = score * viable

    return np.clip(score, 0.0, 1.0)


def _local_answer(question: str, passage: str) -> dict:
    """Retrieve the statements that answer the question, best first.

    The extractive span is kept as a short answer when it is both confident and
    actually inside the retrieved statements — it is the part a reader wants
    first for "who discovered mitochondria" — but it never stands alone any more.
    """
    spans = split_sentences_spans(passage)
    if not spans:
        return {
            "answer": "",
            "short_answer": "",
            "sources": [],
            "confidence": 0.0,
            "unsupported": False,
        }

    # A question the passage says nothing about has no sources to rank, and
    # ranking anyway is how "what is a dog" ended up citing sentences on
    # mitochondria. Decided before the models run, so it costs nothing.
    if question_is_off_topic(question, passage):
        return {
            "answer": unsupported_reply("passage", question, passage),
            "short_answer": "",
            "sources": [],
            "confidence": 0.0,
            "unsupported": True,
        }

    # Below ~0.15 the span carries no information: the model answered "what is a
    # glycolytic product" with "A mitochondrion" at 0.7%, and "what is ATP" with
    # the definition of an organelle at 4%. Only a confident span gets a vote.
    span, span_score = answer_question(question, passage)
    if not span or span.startswith("[QA error"):
        span, span_score = "", 0.0
    trusted_span = span if span_score >= 0.15 else ""

    relevance = _rank_sentences(question, spans, trusted_span)
    order = np.argsort(-relevance)
    best = float(relevance[order[0]])
    if best <= 0.0:
        # Not one sentence shares a term with the question. There is nothing to
        # cite, so the first sentence of the passage is not "the source" — it is
        # whatever happened to be at the top.
        return {
            "answer": unsupported_reply("passage", question, passage),
            "short_answer": "",
            "sources": [],
            "confidence": 0.0,
            "unsupported": True,
        }
    chosen = [int(order[0])] + [
        int(i) for i in order[1:3] if float(relevance[i]) >= 0.6 * best
    ]
    chosen.sort()

    answer = " ".join(spans[i][0] for i in chosen)
    total = sum(float(relevance[i]) for i in chosen) or 1.0
    sources = [
        {
            "text": spans[i][0],
            "contribution": round(float(relevance[i]) / total * 100, 1),
            "reason": "Highest TF-IDF and keyword overlap with the question.",
        }
        for i in sorted(chosen, key=lambda i: -float(relevance[i]))
    ]

    # The span leads the answer only when the model was sure of it and the
    # retrieved statements actually contain it.
    short = ""
    if trusted_span and len(trusted_span) <= 80 and _containment(answer, trusted_span) >= 0.9:
        short = trusted_span

    return {
        "answer": answer,
        "short_answer": short,
        "sources": sources,
        "confidence": round(best, 4),
        "unsupported": False,
    }


def explain_text_qa(question: str, context: str) -> dict:
    """Answer `question` from `context` and report what the answer rests on."""
    passage = clean_educational_text(context)
    query = str(question or "").strip()
    if not passage or not query:
        return {
            "answer": "[QA error: empty question or context]",
            "short_answer": "",
            "engine": "local",
            "engine_note": None,
            "confidence": 0.0,
            "confidence_label": "Passage match",
            "cleaned_text": passage,
            "sources": [],
            "unsupported": False,
            "heatmap": {"text": passage, "spans": [], "sentences": []},
        }

    parsed, note = _gemini_answer(query, passage)
    answer = ""
    sources = []
    if parsed:
        answer = str(parsed.get("answer", "") or "").strip()
        for source in parsed.get("sources", []) or []:
            if not isinstance(source, dict):
                continue
            quote = str(source.get("text", "") or "").strip()
            if not quote:
                continue
            try:
                contribution = float(source.get("contribution", 0.0) or 0.0)
            except (TypeError, ValueError):
                contribution = 0.0
            sources.append(
                {
                    "text": quote,
                    "contribution": round(contribution * 100, 1),
                    "reason": str(source.get("reason", "") or "").strip(),
                }
            )

    # A declined answer is a finished answer, so it returns here rather than
    # falling through to the retrieval path — which would only re-derive the same
    # "nothing matches" conclusion. Sources and heat map go out empty on purpose:
    # there is no evidence to point at, and 50/50 filler citations were the bug.
    if answer and reply_is_unsupported(parsed or {}, answer):
        return {
            "answer": answer,
            "short_answer": "",
            "engine": "gemini",
            "engine_note": None,
            "confidence": 0.0,
            "confidence_label": "Grounding",
            "cleaned_text": passage,
            "sources": [],
            "unsupported": True,
            "heatmap": {"text": passage, "spans": [], "sentences": []},
        }

    if answer and sources:
        # Grounding stands in for a confidence score Gemini does not report:
        # the share of each quote that is really in the passage, weighted by the
        # contribution claimed for it. A paraphrased "quote" drags it down.
        weights = [s["contribution"] for s in sources]
        total = sum(weights) or 1.0
        grounding = sum(
            _containment(passage, s["text"]) * w for s, w in zip(sources, weights)
        ) / total
        return {
            "answer": answer,
            "short_answer": "",
            "engine": "gemini",
            "engine_note": None,
            "confidence": round(min(1.0, grounding), 4),
            "confidence_label": "Grounding",
            "cleaned_text": passage,
            "sources": sources,
            "unsupported": False,
            "heatmap": build_source_heat(passage, sources),
        }

    if parsed and not note:
        # Named neutrally, like every other note the reader sees: which vendor
        # sits behind the hosted tier is not something the page discloses.
        note = (
            "The hosted analysis service returned no usable sources — this "
            "answer came from the passage itself."
        )

    local = _local_answer(query, passage)
    return {
        "answer": local["answer"] or "[QA error: no answer found]",
        "short_answer": local["short_answer"],
        "engine": "local",
        "engine_note": note,
        "confidence": local["confidence"],
        "confidence_label": "Passage match",
        "cleaned_text": passage,
        "sources": local["sources"],
        "unsupported": bool(local["unsupported"]),
        # An unsupported answer gets the same empty heat map the hosted path
        # sends, so the payload does not depend on which engine declined.
        "heatmap": (
            {"text": passage, "spans": [], "sentences": []}
            if local["unsupported"]
            else build_source_heat(passage, local["sources"])
        ),
    }
