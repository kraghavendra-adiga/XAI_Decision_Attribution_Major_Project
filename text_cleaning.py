import re


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_CITATION_RE = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")
_REFERENCE_LINE_RE = re.compile(r"^\s*\[\d+\]\s*\S+.*$", re.MULTILINE)
_SOURCE_WARNING_RE = re.compile(r"\bAI responses may include mistakes\.?", re.IGNORECASE)
_BULLET_RE = re.compile(r"^\s*(?:•|â€¢|[?*-])\s+", re.MULTILINE)
_MATH_INLINE_RE = re.compile(r"\$([^$]+)\$")
_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")

# Pasted search results glue a heading onto the next sentence ("agoYouTube",
# "ScientistMain", "MythsThey"). Requiring three lowercase letters before a
# Titlecase word keeps real intra-word capitals intact: iPhone, eBay, macOS and
# McDonald all have fewer than three lowercase letters before the capital.
_GLUED_WORDS_RE = re.compile(r"(?<=[a-z]{3})(?=[A-Z][a-z])")

# The same paste drops the space after terminal punctuation ("singularity.Watch",
# "A*).Key", "hole?22K"). A lowercase letter or closing bracket before the mark
# excludes initialisms (U.S.A) and decimals (1.44), which must not be split.
_GLUED_SENTENCE_RE = re.compile(r"(?<=[a-z)][.!?])(?=[A-Z0-9])")

# Chrome glues to the words around it too ("space:1:44What"). Ungluing digits
# from a following Titlecase word lets the duration match its own \b boundary.
_GLUED_NUMBER_RE = re.compile(r"(?<=\d)(?=[A-Z][a-z])|(?<=[a-z]):(?=[A-Z])")

# Video/search chrome that carries no lesson content but dominates token-level
# attribution once it is pasted in: durations, view counts, relative dates.
_MEDIA_CHROME_RE = re.compile(
    r"\b\d{1,2}:\d{2}(?::\d{2})?\b"
    r"|\b\d[\d.,]*\s*[KMB]?\s*views?\b"
    r"|\b\d+\s*(?:sec|second|min|minute|hour|day|week|month|year)s?\s+ago\b"
    r"|\b(?:Subscribe|Watch later|Share|Save|Shorts|Sponsored|Ad)\b",
    re.IGNORECASE,
)

# A platform name, optionally with the channel or author credited after it.
# Capped at two Titlecase words: a longer run starts swallowing the lesson's own
# headings ("YouTube - New Scientist" must not extend into "Main Types of ...").
_PLATFORM_CREDIT_RE = re.compile(
    r"\b(?:YouTube|Vimeo|Instagram|TikTok|Facebook|Twitter|Reddit|Quora|"
    r"LinkedIn|Medium|Wikipedia)\b"
    r"(?:\s*[·•|:-]\s*[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*)?)?"
)
_MIDDOT_RE = re.compile(r"\s*[·•]\s*")

# Abbreviations whose period must never end a sentence.
_ABBREVIATIONS = (
    "e.g", "i.e", "etc", "vs", "cf", "approx", "fig", "eq", "no",
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr",
    # Parenthetical glosses in reference prose: "A mitochondrion (pl.
    # mitochondria) is an organelle..." was being cut in half at "pl.", which
    # left "A mitochondrion (pl." to be ranked and cited as if it were a claim.
    "pl", "sing", "al", "vol", "ed", "pp", "ca", "sp", "spp",
)
_ABBREV_RE = re.compile(
    r"(?:\b(?:" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")"
    r"|\b(?:[A-Za-z]\.){1,4}[A-Za-z])\.$",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])[ \t]+|\n+")


def clean_educational_text(text: str) -> str:
    """Remove pasted AI-search citation noise while preserving the lesson text."""
    clean = str(text or "").replace(" ", " ")
    clean = _REFERENCE_LINE_RE.sub("", clean)
    clean = _SOURCE_WARNING_RE.sub("", clean)
    clean = _URL_RE.sub("", clean)
    clean = _CITATION_RE.sub("", clean)
    clean = _MATH_INLINE_RE.sub(r"\1", clean)
    clean = _BULLET_RE.sub("", clean)
    clean = clean.replace("\r\n", "\n").replace("\r", "\n")

    # Unglue before stripping chrome: "agoYouTube" has to become two words
    # before the platform credit is recognisable.
    clean = _GLUED_WORDS_RE.sub(" ", clean)
    clean = _GLUED_SENTENCE_RE.sub(" ", clean)
    clean = _GLUED_NUMBER_RE.sub(" ", clean)
    clean = _PLATFORM_CREDIT_RE.sub("", clean)
    clean = _MEDIA_CHROME_RE.sub("", clean)
    clean = _MIDDOT_RE.sub(" ", clean)

    # Stripping chrome mid-sentence leaves dangling punctuation behind it.
    clean = re.sub(r"[ \t]+([,.;:!?])", r"\1", clean)
    clean = re.sub(r"([:;,])\s*(?=[.!?])", "", clean)
    clean = _WHITESPACE_RE.sub(" ", clean)
    clean = _BLANK_LINES_RE.sub("\n\n", clean)
    return clean.strip()


def split_sentences_spans(text: str) -> list:
    """Split cleaned text into (sentence, start, end) triples.

    clean_educational_text() restores the space after terminal punctuation, so a
    whitespace split is enough here; the abbreviation pass then undoes the few
    splits that landed inside "e.g." or "Dr.".

    The offsets are into `text` exactly as passed in, which is what lets the
    heat map paint character ranges of the passage the user is looking at. They
    are kept here, next to the split rule, so the two cannot drift apart.
    """
    raw = str(text or "")
    bounds = []
    cursor = 0
    for match in _SENTENCE_SPLIT_RE.finditer(raw):
        bounds.append((cursor, match.start()))
        cursor = match.end()
    bounds.append((cursor, len(raw)))

    spans = []
    for start, end in bounds:
        piece = raw[start:end]
        sentence = piece.strip()
        if not sentence:
            continue
        begin = start + (len(piece) - len(piece.lstrip()))
        finish = begin + len(sentence)
        if spans and _ABBREV_RE.search(spans[-1][0]):
            _, prev_start, _ = spans[-1]
            # Sliced from the source rather than joined with a space, so the
            # sentence text and its offsets can never disagree.
            spans[-1] = (raw[prev_start:finish], prev_start, finish)
        else:
            spans.append((sentence, begin, finish))
    return spans


def split_sentences(text: str) -> list:
    """Split cleaned text into sentences, rejoining abbreviations."""
    return [sentence for sentence, _, _ in split_sentences_spans(text)]


def clean_attribution_feature(feature: str) -> str:
    """Normalize one displayed LIME/SHAP feature and hide citation leftovers."""
    clean = clean_educational_text(feature)
    clean = clean.strip(" \t\n\r.,;:()[]{}\"'")
    return clean
