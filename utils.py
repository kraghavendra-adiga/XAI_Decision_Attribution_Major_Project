"""Shared utility functions used across explainers."""

import base64
import difflib
import os
import re
from io import BytesIO

from PIL import Image
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ── Gemini setup ─────────────────────────────────────────────────────
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

load_dotenv()

# Single source of truth for the model name; retired ids fail with a 404 at
# call time, so both the image and audio paths must stay in sync.
GEMINI_MODEL = "gemini-3.6-flash"

__all__ = [
    "GEMINI_MODEL",
    "ServiceOverloaded",
    "gemini_available",
    "gemini_model",
    "describe_service_error",
    "is_exhausted_for_today",
    "parse_json_block",
    "INTERROGATIVES",
    "not_answerable_rules",
    "question_topic_terms",
    "unmentioned_terms",
    "question_is_off_topic",
    "reply_is_unsupported",
    "text_is_unsupported",
    "unsupported_reply",
    "source_topics",
    "_image_to_b64",
    "_get_nltk_stopwords",
    "_extract_keywords",
]


class ServiceOverloaded(RuntimeError):
    """A hosted-model call failed for a reason the user can act on.

    Carries a message that is already fit to display, so the endpoint can pass
    `str(exc)` straight to the client instead of leaking a provider stack trace.
    """


# Raw provider errors are unreadable ("429 You exceeded your current quota,
# please check your plan and billing details. [violations { quota_metric: ... }]")
# and name a vendor the UI deliberately does not mention. Each pattern maps to
# one sentence saying what happened and what the user can do about it.
#
# The messages say nothing about falling back to the on-device engine, because
# only the text and image paths have one — audio has to fail outright. Whoever
# does fall back appends its own sentence saying so.
_ERROR_PATTERNS = (
    (
        ("429", "quota", "resource_exhausted", "rate limit", "too many requests"),
        "The analysis service is overloaded right now and turned this request "
        "away. It was retried automatically without luck — waiting a moment and "
        "running it again usually gets through.",
    ),
    (
        ("503", "unavailable", "overloaded", "server is busy", "capacity"),
        "The analysis service is temporarily overloaded and could not take this "
        "request. Trying again in a minute usually clears it.",
    ),
    (
        ("504", "deadline", "timeout", "timed out"),
        "The analysis service took too long to respond and the request was "
        "dropped. A shorter input, or a retry, normally goes through.",
    ),
    (
        ("api key", "api_key", "unauthenticated", "401", "permission_denied", "403"),
        "The analysis service rejected this request as unauthorised, so it could "
        "not be completed.",
    ),
    (
        ("404", "not found", "is not supported", "does not exist"),
        "The analysis service reported that the requested model is unavailable.",
    ),
    (
        ("connection", "network", "dns", "getaddrinfo", "ssl", "unreachable"),
        "The analysis service could not be reached — this usually means the "
        "network dropped. Check the connection and run it again.",
    ),
    (
        ("json", "parse", "malformed"),
        "The analysis service returned a reply that could not be read. Running "
        "it again usually produces a clean one.",
    ),
)

_GENERIC_SERVICE_ERROR = (
    "The analysis service is unavailable at the moment. Trying again shortly "
    "usually works."
)

_DAILY_QUOTA_ERROR = (
    "The analysis service has used up its request allowance for today. That "
    "resets on a 24-hour cycle, so retrying now will hit the same limit."
)

# The daily allowance is the one refusal a retry cannot ride out, so it gets its
# own test rather than a needle in the table above. The marker arrives as a
# quota_id rather than prose — "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
# — with no space inside "PerDay", so punctuation is squashed out before looking
# for it. Matching on "per day" alone silently misses every real payload.
_DAILY_MARKERS = ("perday", "dailylimit", "requestsperday")


def is_exhausted_for_today(exc: BaseException | str) -> bool:
    """Has the allowance for the day been spent, rather than for the minute?

    A per-minute quota clears on its own and a short sleep rides it out. A
    per-day one does not, so retrying only makes the caller wait for the same
    refusal — every retry loop checks this before sleeping.
    """
    raw = exc if isinstance(exc, str) else f"{type(exc).__name__} {exc}"
    squashed = re.sub(r"[^a-z0-9]", "", raw.lower())
    return any(marker in squashed for marker in _DAILY_MARKERS)


def describe_service_error(exc: BaseException) -> str:
    """Turn a provider exception into one sentence worth showing a user.

    The daily-quota case is settled before the table is consulted, because a
    spent daily allowance does not clear by waiting a minute and telling someone
    to retry would be wrong.
    """
    if is_exhausted_for_today(exc):
        return _DAILY_QUOTA_ERROR
    haystack = f"{type(exc).__name__} {exc}".lower()
    for needles, message in _ERROR_PATTERNS:
        if any(needle in haystack for needle in needles):
            return message
    return _GENERIC_SERVICE_ERROR


def gemini_available() -> bool:
    """Is a key configured at all? Skips a call that can only 400."""
    return bool(os.getenv("GEMINI_API_KEY"))


# ── The provider SDK, behind one adapter ─────────────────────────────
#
# `google.generativeai` reached end of life — no fixes, no new models — so the
# calls go through `google.genai` instead. The two SDKs differ in shape: the old
# one bound the model name to an object and took loose dicts for inline media,
# the new one passes the name per call and wants typed Parts.
#
# Rather than spread that difference over five call sites in four explainers,
# it is absorbed here. `gemini_model().generate_content(...)` accepts exactly
# what the old object accepted — a string, or a list mixing strings with
# {"mime_type", "data"} blobs — so swapping providers again stays a one-file job.

_client = None


def _gemini_client():
    """The shared client, built on first use.

    Lazily, because importing this module must not depend on a key being
    present: every caller already asks `gemini_available()` first and falls back
    to a local engine, and a missing key should reach them as that fallback
    rather than as an ImportError.
    """
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _client


def _as_part(item):
    """One piece of a request, in the form the SDK now expects."""
    if isinstance(item, dict) and "data" in item:
        return genai_types.Part.from_bytes(
            data=item["data"], mime_type=item.get("mime_type", "application/octet-stream")
        )
    return item


class _GeminiModel:
    """Presents the old `GenerativeModel.generate_content` surface."""

    def __init__(self, model: str) -> None:
        self._model = model

    def generate_content(self, contents):
        parts = [_as_part(i) for i in contents] if isinstance(contents, list) else contents
        return _gemini_client().models.generate_content(
            model=self._model, contents=parts
        )


def gemini_model(model: str = GEMINI_MODEL) -> _GeminiModel:
    """A handle to call the hosted model with. Cheap — holds no connection."""
    return _GeminiModel(model)


def parse_json_block(raw: str) -> dict:
    """Pull the JSON object out of a Gemini reply.

    Gemini wraps its JSON in ```json fences often enough, and prefixes it with a
    sentence of preamble occasionally, that both have to be stripped before
    json.loads() will look at it.
    """
    import json

    text = str(raw or "").strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    if start > 0:
        text = text[start:]
    end = text.rfind("}")
    if end != -1 and end < len(text) - 1:
        text = text[: end + 1]

    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("The hosted analysis service returned JSON that is not an object")
    return parsed


def _image_to_b64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _get_nltk_stopwords() -> set[str]:
    """Lazily load NLTK English stopwords."""
    try:
        from nltk.corpus import stopwords as _sw  # type: ignore[import-untyped]
        return set(_sw.words("english"))
    except Exception:
        # Fallback to sklearn stopwords if NLTK data missing
        return set(ENGLISH_STOP_WORDS)


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from text, removing stopwords and punctuation."""
    stops = _get_nltk_stopwords()
    words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text)
    return {
        w.lower() for w in words
        if len(w) > 1 and w.lower() not in stops and not w.isdigit()
    }


# ── Questions the source cannot answer ──────────────────────────────────────
#
# Every prompt here used to demand "minimum 2 sources", so asking a mitochondria
# recording "what is a dog" produced the right answer — the transcript does not
# discuss dogs — bolted onto two invented citations at 50% each, a donut split
# evenly between them, and timestamps pointing at whichever lines were least
# unrelated. The model's own `reason` strings said as much ("no mention of dogs").
# That is not attribution: there is nothing to attribute.
#
# So each prompt now asks for an `answerable` flag and an empty source list when
# it is false, and every path drops its sources, chart, evidence and heat map in
# that case. The local retrieval paths have no model to ask, so they decide it
# from whether the source mentions the question's subject at all.

# Question words carry no subject matter: "what is a dog" is asking about dogs,
# and matching "what" against the source would make every question look answered.
INTERROGATIVES = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "does", "do", "did", "is", "are", "was", "were", "can", "could", "would",
    "explain", "describe", "define", "tell", "give", "list", "name", "mean",
    "meaning", "means", "about",
}


def not_answerable_rules(noun: str) -> str:
    """The prompt rules that let a model decline instead of inventing sources.

    Worded once here so the five tabs behave identically; `noun` is what the
    model is looking at ("transcript", "image", "passage", "pages").
    """
    return (
        f'- Set "answerable" to false when the {noun} does not discuss what the '
        f'question is about. In that case "sources" MUST be an empty list, and '
        f'"answer" must be one or two sentences stating plainly that the {noun} '
        f'contains no information about it and naming what it covers instead.\n'
        f'- Only cite sources when "answerable" is true. Never cite the closest '
        f'related line as a stand-in for evidence that is not there.'
    )


# Prose is the second channel for the flag: a model that ignores the schema still
# writes "the transcript does not contain any information about dogs" in the
# answer, and citations attached to that sentence are filler by definition. Both
# halves must match within one sentence — a bare negation ("the cell does not
# contain a nucleus") is a fact about the subject, not a refusal — so a source
# noun has to be present as well as a negated verb of *carrying content*.
_SOURCE_NOUN_RE = re.compile(
    r"\b(?:transcript|passage|document|text|image|picture|photo|video|audio"
    r"|recording|page|pages|slide|slides|context|pdf|presentation|file|excerpt"
    r"|content|material|source|diagram|chart|figure)\b",
    re.IGNORECASE,
)

_NO_CONTENT_RE = re.compile(
    r"\b(?:does not|doesn't|do not|don't|did not|didn't|is not|isn't|are not"
    r"|aren't|cannot|can't|could not|couldn't|no|not|never|without|lacks"
    r"|lack|omits|omit)\b"
    r"[^.?!]{0,40}?"
    r"\b(?:contain|contains|mention|mentions|discuss|discusses|cover|covers"
    r"|address|addresses|include|includes|refer|refers|provide|provides"
    r"|describe|describes|state|states|say|says|answer|answers|relate"
    r"|relates|show|shows|information|detail|details|reference|data)\b",
    re.IGNORECASE,
)


def text_is_unsupported(answer: str) -> bool:
    """Does this answer read as "the source does not cover that"?

    Only the opening two sentences are examined. A refusal leads with itself; a
    caveat three sentences into a real answer ("the passage does not name the
    enzyme, but...") is qualifying evidence that does exist, and suppressing its
    sources would throw away a good answer.
    """
    sentences = re.split(r"(?<=[.!?])\s+", str(answer or "").strip())
    return any(
        _SOURCE_NOUN_RE.search(sentence) and _NO_CONTENT_RE.search(sentence)
        for sentence in sentences[:2]
    )


def reply_is_unsupported(parsed: dict, answer: str = "") -> bool:
    """Did the model decline to answer from the source?

    An explicit `answerable` flag settles it either way — a model that says the
    question *is* answerable is trusted over any phrasing in its prose. The text
    test only runs when the field is missing, which is the case for every reply
    from a model that ignored the schema.
    """
    flag = parsed.get("answerable") if isinstance(parsed, dict) else None
    if isinstance(flag, bool):
        return not flag
    if isinstance(flag, str) and flag.strip().lower() in {
        "true", "false", "yes", "no"
    }:
        return flag.strip().lower() in {"false", "no"}
    return text_is_unsupported(answer)


def question_topic_terms(question: str) -> set[str]:
    """The content words of a question — what it is actually asking about."""
    return {t for t in _extract_keywords(question) if t not in INTERROGATIVES}


def _root(word: str) -> str:
    """Crudely singularise a word so "dogs" and "dog" compare equal."""
    w = re.sub(r"[^a-z0-9]", "", str(word or "").lower())
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("es") and not w.endswith(("ses", "ces", "ges")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def _mentions(term: str, source_terms: set, source_roots: set) -> bool:
    """Is `term` in the source, allowing for plurals and Latin endings?

    Irregular plurals matter here — a question about a "mitochondrion" is about
    the "mitochondria" the source names, and calling that unmentioned would
    refuse a question the source answers well. A similarity test handles them,
    restricted to candidates sharing a prefix so "cat" never matches "catalyst".
    """
    root = _root(term)
    if term in source_terms or root in source_roots:
        return True
    head = root[:3]
    return any(
        candidate.startswith(head)
        and difflib.SequenceMatcher(None, root, candidate).ratio() >= 0.82
        for candidate in source_roots
    )


def unmentioned_terms(question: str, source_text: str) -> list[str]:
    """Which of the question's subject words appear nowhere in the source."""
    terms = question_topic_terms(question)
    if not terms:
        return []
    source_terms = _extract_keywords(source_text)
    source_roots = {_root(t) for t in source_terms}
    return sorted(t for t in terms if not _mentions(t, source_terms, source_roots))


def question_is_off_topic(question: str, source_text: str) -> bool:
    """Does the source mention nothing at all that the question asks about?

    Deliberately the strictest possible test: every subject word missing, not
    most of them. A question sharing even one term with the source is answered as
    before, because a weak answer to a real question is a smaller failure than
    refusing one the source does address. Off-topic questions score zero here —
    "dog" against a mitochondria transcript — so the strict rule still catches
    the case that produced the filler citations.
    """
    terms = question_topic_terms(question)
    if not terms:
        return False
    return len(unmentioned_terms(question, source_text)) == len(terms)


def _join_terms(terms: list) -> str:
    """"a", "a and b", "a, b and c"."""
    items = [str(t) for t in terms if str(t).strip()]
    if not items:
        return "that"
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


def source_topics(source_text: str, limit: int = 3) -> list[str]:
    """The words the source keeps returning to, for naming what it is about.

    Term frequency, nothing cleverer: this only finishes a sentence that has
    already said the source is about something else, and "mainly about" is a
    claim raw counts can support. Words used once are dropped when anything is
    used more — on a short passage every count is 1 and the list would otherwise
    be whatever came first alphabetically.

    Three, not more: counts rank the top of the list well and the tail badly. A
    fourth word was reliably the weakest — a bare modifier ("inner"), or on OCR'd
    text a run-together artifact ("andcellular") — and it sat in the most visible
    position, last.
    """
    stops = _get_nltk_stopwords()
    counts: dict[str, int] = {}
    for word in re.findall(r"[A-Za-z][A-Za-z0-9'-]+", str(source_text or "")):
        lowered = word.lower()
        if len(lowered) < 3 or lowered in stops or lowered in INTERROGATIVES:
            continue
        counts[lowered] = counts.get(lowered, 0) + 1
    repeated = {w: c for w, c in counts.items() if c > 1}
    pool = repeated if len(repeated) >= 2 else counts
    # Longest first among equal counts: "mitochondria" says more than "area".
    ranked = sorted(pool, key=lambda w: (-pool[w], -len(w), w))
    # One topic per root, or the list reads "mitochondrion, mitochondria".
    topics: list[str] = []
    seen: set[str] = set()
    for word in ranked:
        root = _root(word)
        if root in seen or any(
            difflib.SequenceMatcher(None, root, other).ratio() >= 0.82
            for other in seen
        ):
            continue
        seen.add(root)
        topics.append(word)
        if len(topics) >= limit:
            break
    return topics


def _question_subject(question: str, missing: list) -> str:
    """How to name what was asked about, in the question's own words.

    The missing terms come back normalised, so "what is a dog" would read as
    "information about dog". When one term is missing and the question put an
    article in front of it, that article is kept — the sentence is going to be
    read by a person.
    """
    surface = _join_terms(missing)
    if len(missing) != 1:
        return surface
    match = re.search(
        rf"\b(a|an|the)\s+({re.escape(missing[0])}\w*)\b", question, re.IGNORECASE
    )
    return f"{match.group(1).lower()} {match.group(2)}" if match else surface


def unsupported_reply(noun: str, question: str, source_text: str) -> str:
    """The plain statement that stands in for an answer the source cannot give.

    Same shape the hosted model produces, so a reader cannot tell which engine
    declined: what was asked about is absent, and here is what is there instead.
    """
    missing = unmentioned_terms(question, source_text) or sorted(
        question_topic_terms(question)
    )
    # "is there a dog in this video" is asking about a dog. The medium the reader
    # named is not part of the subject, though it does stay in the decision above:
    # a term the source happens to use is evidence the question is on topic.
    subject = [t for t in missing if not _SOURCE_NOUN_RE.fullmatch(t)] or missing
    sentence = (
        f"The provided {noun} does not contain any information about "
        f"{_question_subject(question, subject[:3])}."
    )
    topics = source_topics(source_text)
    if topics:
        sentence += f" It is mainly about {_join_terms(topics)}."
    return sentence


