"""Image analysis and Grad-CAM heatmap explainability."""

import json
import os
import re
import time
from io import BytesIO

import numpy as np
import pytesseract
from PIL import Image, ImageFilter, ImageOps
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

try:
    from .model_utils import answer_question
    from .utils import (
        GEMINI_MODEL,
        _extract_keywords,
        _image_to_b64,
        describe_service_error,
        gemini_model,
        is_exhausted_for_today,
        not_answerable_rules,
        question_is_off_topic,
        reply_is_unsupported,
        unsupported_reply,
    )
except ImportError:
    from model_utils import answer_question
    from utils import (
        GEMINI_MODEL,
        _extract_keywords,
        _image_to_b64,
        describe_service_error,
        gemini_model,
        is_exhausted_for_today,
        not_answerable_rules,
        question_is_off_topic,
        reply_is_unsupported,
        unsupported_reply,
    )

# Configure Tesseract binary path for Windows
import shutil as _shutil
if not _shutil.which("tesseract"):
    _tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.isfile(_tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = _tesseract_path

# Re-exported so the explainer.py compatibility shim keeps exposing the shared
# helpers under their original module path.
__all__ = [
    "GEMINI_MODEL",
    "IMAGE_STAGES",
    "_image_to_b64",
    "_extract_keywords",
    "_clean_ocr_line",
    "_dedupe_lines",
    "_group_ocr_by_line",
    "_build_heatmap",
    "preprocess_for_ocr",
    "explain_image_gemini",
    "explain_image_vqa",
    "_stem",
    "_content_terms",
    "_line_text",
    "_prose_indices",
    "_is_usable_line",
    "_is_heading",
    "_columns_of",
    "_group_into_blocks",
    "_score_ocr_lines",
    "_select_relevant",
    "_read_image_text",
    "_compose_answer",
]


def _clean_ocr_line(line: str) -> str:
    line = re.sub(r"\s+", " ", str(line or "")).strip()
    return line.strip(" |")


def _dedupe_lines(lines: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for line in lines:
        clean_line = _clean_ocr_line(line)
        key = re.sub(r"[^a-z0-9]+", " ", clean_line.lower()).strip()
        if len(clean_line) < 3 or not key or key in seen:
            continue
        seen.add(key)
        deduped.append(clean_line)
    return deduped


def _group_ocr_by_line(ocr_data: dict, scale_x: float = 1.0, scale_y: float = 1.0) -> dict:
    """Group OCR tokens into lines using block_num, par_num, line_num."""
    lines: dict[tuple, dict] = {}
    texts = ocr_data.get("text", [])
    for idx, raw_word in enumerate(texts):
        word = str(raw_word or "").strip()
        if not word:
            continue
        try:
            conf = float(ocr_data["conf"][idx])
        except (TypeError, ValueError, IndexError):
            conf = 0.0
        if conf < 0:
            continue

        key = (
            int(ocr_data.get("block_num", [0])[idx]),
            int(ocr_data.get("par_num", [0])[idx]),
            int(ocr_data.get("line_num", [0])[idx]),
        )
        left = int(int(ocr_data["left"][idx]) * scale_x)
        top = int(int(ocr_data["top"][idx]) * scale_y)
        right = left + int(int(ocr_data["width"][idx]) * scale_x)
        bottom = top + int(int(ocr_data["height"][idx]) * scale_y)

        if key not in lines:
            lines[key] = {"words": [], "boxes": []}
        line = lines[key]
        line["words"].append(word)
        line["boxes"].append((left, top, right, bottom))

    # The bounding box is taken from the prose only. A callout numeral swept into
    # the row would otherwise stretch the box — and so the highlight — across
    # blank space the text never occupied.
    for line in lines.values():
        keep = _prose_indices(line["words"], line["boxes"])
        line["words"] = [line["words"][i] for i in keep]
        boxes = [line["boxes"][i] for i in keep]
        line["boxes"] = boxes
        line["left"] = min(box[0] for box in boxes)
        line["top"] = min(box[1] for box in boxes)
        line["right"] = max(box[2] for box in boxes)
        line["bottom"] = max(box[3] for box in boxes)

    return lines


def _build_heatmap(
    image: Image.Image,
    ocr_lines: dict,
    line_scores: dict,
    threshold: float = 0.0,
) -> Image.Image:
    """Paint each OCR line with an intensity proportional to its relevance.

    The previous version filled every matched box with the same flat red and
    tinted everything else blue, so a weak keyword coincidence looked exactly
    as important as the line the answer actually came from. Scores now drive
    both opacity and hue, and the falloff is drawn per-line so a bright region
    does not bleed across the neighbouring text.
    """
    base = image.convert("RGBA")
    heat = np.zeros((base.height, base.width), dtype=np.float32)

    for key, score in line_scores.items():
        if score <= threshold or key not in ocr_lines:
            continue
        line = ocr_lines[key]
        pad = 3
        x0 = max(0, line["left"] - pad)
        y0 = max(0, line["top"] - pad)
        x1 = min(base.width, line["right"] + pad)
        y1 = min(base.height, line["bottom"] + pad)
        if x1 <= x0 or y1 <= y0:
            continue
        # np.maximum, not +=, so overlapping boxes do not sum into a hotter
        # region than either line earned on its own.
        heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1], float(score))

    if not heat.any():
        return image.convert("RGB")

    # Blur in float space for a soft edge, then renormalise: blurring alone
    # would dim every peak and wash out the strongest evidence.
    blurred = np.array(
        Image.fromarray((heat * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=9)
        ),
        dtype=np.float32,
    ) / 255.0
    peak = float(blurred.max()) or 1.0
    blurred = (blurred / peak) * float(heat.max())

    # Blue -> cyan -> yellow -> red, so intermediate relevance is visible
    # rather than collapsing to "red or nothing".
    stops = np.array(
        [[40, 90, 210], [40, 190, 200], [250, 210, 60], [235, 45, 45]],
        dtype=np.float32,
    )
    positions = np.clip(blurred, 0.0, 1.0) * (len(stops) - 1)
    lower = np.floor(positions).astype(np.int32)
    upper = np.clip(lower + 1, 0, len(stops) - 1)
    frac = (positions - lower)[..., None]
    colour = stops[lower] * (1 - frac) + stops[upper] * frac

    # Alpha ramps with the score, so low-relevance text stays legible.
    alpha = np.clip(blurred, 0.0, 1.0) ** 0.75 * 165.0

    overlay = np.concatenate(
        [colour, alpha[..., None]], axis=-1
    ).astype(np.uint8)
    result = Image.alpha_composite(
        base, Image.fromarray(overlay, mode="RGBA")
    )
    return result.convert("RGB")


# ── Question-aware relevance scoring ────────────────────────────────────────

# Kept deliberately small: aggressive stemming ("studies" -> "studi") produced
# false matches, while these suffixes cover the plural/tense drift that made
# "dog" miss "dogs".
def _stem(word: str) -> str:
    """Fold a word to a comparable root ("dogs" -> "dog", "studies" -> "study")."""
    word = re.sub(r"[^a-z0-9]", "", str(word or "").lower())
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    for suffix in ("sses", "shes", "ches", "xes"):
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


_QUESTION_WORDS = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "will",
    "would", "should", "shall", "may", "might", "the", "a", "an", "tell", "me",
    "about", "list", "give", "show", "explain", "describe", "please",
}


def _content_terms(text: str) -> set[str]:
    """Stemmed content words, with interrogatives and stopwords removed."""
    words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", str(text or ""))
    terms = set()
    for word in words:
        lowered = word.lower()
        if lowered in _QUESTION_WORDS or lowered in ENGLISH_STOP_WORDS:
            continue
        if len(lowered) < 2 or lowered.isdigit():
            continue
        stemmed = _stem(word)
        if stemmed:
            terms.add(stemmed)
    return terms


def _prose_indices(words: list, boxes: list) -> list:
    """Which words of a row are prose, dropping callout numbers at either end.

    Infographics set a big "6" beside each bullet. Tesseract folded it into the
    adjacent row, so the evidence read "Pavlov's experiments on dogs gave 6".
    A decorative number is separated from the prose by a much wider gap than the
    word spacing, which is what identifies it — a date written inline ("in
    1957") sits at normal spacing and is kept.
    """
    keep = list(range(len(words)))
    if len(words) < 3 or len(boxes) != len(words):
        return keep

    gaps = [boxes[i + 1][0] - boxes[i][2] for i in range(len(boxes) - 1)]
    positive = sorted(gap for gap in gaps if gap > 0)
    if not positive:
        return keep
    detached = max(20, positive[len(positive) // 2] * 3)

    # Only the ends can be strays; a number mid-sentence is part of the text.
    while len(keep) > 2 and re.fullmatch(r"\d{1,2}", words[keep[-1]]) \
            and gaps[keep[-1] - 1] >= detached:
        keep.pop()
    while len(keep) > 2 and re.fullmatch(r"\d{1,2}", words[keep[0]]) \
            and gaps[keep[0]] >= detached:
        keep.pop(0)
    return keep


def _line_text(line: dict) -> str:
    """Readable text of an OCR line, or of a merged block if already joined."""
    if line.get("text"):
        return _clean_ocr_line(str(line["text"]))
    return _clean_ocr_line(" ".join(line.get("words", [])))


def _is_usable_line(text: str) -> bool:
    """Reject OCR debris so it cannot be cited as evidence.

    Tesseract emits stray marks as one- or two-character "words" ("N", "y",
    "AB T"). Quoting those as supporting evidence is what made the list look
    random, so they are dropped before anything is scored.
    """
    text = str(text or "").strip()
    if len(text) < 6:
        return False
    letters = re.findall(r"[A-Za-z]", text)
    if len(letters) < 4:
        return False
    # Mostly-symbol lines are decoration or misreads, not sentences.
    if len(letters) / max(1, len(text.replace(" ", ""))) < 0.55:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*", text)
    real_words = [word for word in words if len(word) >= 3]
    return len(real_words) >= 2


def _is_heading(text: str) -> bool:
    """Is this a title or logo rather than a statement of fact?

    Headings are set in caps in virtually every layout, and they assert nothing —
    quoting "UNDERSTANDING ANIMAL RESEARCH" as the evidence behind an answer is
    exactly the "random info" complaint. They stay eligible, just outranked by
    any real sentence.
    """
    letters = re.sub(r"[^A-Za-z]", "", str(text or ""))
    return len(letters) >= 4 and letters.isupper()


def _score_ocr_lines(
    ocr_lines: dict,
    question: str,
    answer: str = "",
) -> dict:
    """Score every OCR line 0..1 for how well it answers the question.

    Counting shared keywords made every line on a page about dogs score alike.
    Three signals are combined instead:
      * IDF weighting, so a term repeated on every line carries little weight
      * cosine similarity against the question, for overall topical fit
      * overlap with the drafted answer, which pins the specific line used
    """
    keys = sorted(ocr_lines)
    if not keys:
        return {}

    # Score only lines that read as text; debris would otherwise win on rarity,
    # since a misread fragment is by definition unique on the page.
    keys = [key for key in keys if _is_usable_line(_line_text(ocr_lines[key]))]
    if not keys:
        return {}

    texts = [_line_text(ocr_lines[key]) for key in keys]
    question_terms = _content_terms(question)
    answer_terms = _content_terms(answer)

    # IDF over the lines of this image: a term on every line is uninformative.
    line_terms = [_content_terms(text) for text in texts]
    document_count = len(line_terms) or 1
    frequencies: dict[str, int] = {}
    for terms in line_terms:
        for term in terms:
            frequencies[term] = frequencies.get(term, 0) + 1

    def idf(term: str) -> float:
        """Weight a term by how *rare* it is on this page.

        Smoothed IDF still gave a term present on every line a weight near 1.0,
        so on a page about dogs the word "dogs" dominated and every line scored
        alike. Squaring the raw ratio drives a page-wide term to ~0, which is
        what makes the remaining words decide the ranking.
        """
        seen = frequencies.get(term, 0)
        if seen <= 0:
            return 1.0
        rarity = 1.0 - (seen / document_count)
        return float(max(0.02, rarity ** 2))

    # A broad question ("what is dog" on a page where every line says dogs)
    # carries no discriminating power on its own. Measure that up front and
    # lean on the answer instead when it happens.
    discriminating = [
        term for term in question_terms
        if frequencies.get(term, 0) < document_count * 0.6
    ]
    question_is_broad = bool(question_terms) and not discriminating

    question_weight = sum(idf(term) for term in question_terms) or 1.0
    answer_weight = sum(idf(term) for term in answer_terms) or 1.0

    # Cosine similarity gives a smooth topical score that survives wording
    # differences between the question and the text on the page.
    cosine = {key: 0.0 for key in keys}
    if question_terms:
        try:
            # No stop_words here: a callable analyzer ignores it (and warns),
            # and _content_terms has already removed them.
            vectorizer = TfidfVectorizer(
                sublinear_tf=True, analyzer=_content_terms
            )
            matrix = vectorizer.fit_transform(texts + [question])
            page, query = matrix[:-1], matrix[-1]
            norms = np.sqrt(page.multiply(page).sum(axis=1)).A1
            query_norm = float(np.sqrt(query.multiply(query).sum())) or 1.0
            similarity = (page @ query.T).toarray().ravel() / (
                np.where(norms == 0, 1.0, norms) * query_norm
            )
            cosine = {key: float(value) for key, value in zip(keys, similarity)}
        except Exception as exc:  # pragma: no cover - vectoriser edge cases
            print(f"[Relevance] cosine unavailable: {type(exc).__name__}: {exc}")

    # When the question cannot separate the lines, the answer decides; ranking
    # by a term every line shares produced the flat ~12% spread in the UI.
    if question_is_broad and answer_terms:
        weights = (0.10, 0.75, 0.15)
    elif answer_terms:
        weights = (0.45, 0.35, 0.20)
    else:
        weights = (0.75, 0.0, 0.25)

    scores = {}
    for key, text, terms in zip(keys, texts, line_terms):
        if not terms:
            scores[key] = 0.0
            continue
        q_overlap = sum(idf(t) for t in terms & question_terms) / question_weight
        a_overlap = (
            sum(idf(t) for t in terms & answer_terms) / answer_weight
            if answer_terms else 0.0
        )
        # Reward covering the answer's wording, so the line the answer was
        # taken from outranks one that merely shares its rarest word.
        if answer_terms:
            a_overlap = 0.5 * a_overlap + 0.5 * (
                len(terms & answer_terms) / len(answer_terms)
            )
        combined = (
            # The question is amplified because its wording legitimately differs
            # from the page's. The answer is not: it was extracted from one of
            # these blocks, so the true source scores ~1.0 unaided, and
            # amplifying lifted lines sharing just two of its words to 0.5.
            weights[0] * min(1.0, q_overlap * 2.0)
            + weights[1] * min(1.0, a_overlap)
            + weights[2] * min(1.0, cosine.get(key, 0.0) * 1.5)
        )
        if _is_heading(text):
            combined *= 0.35
        scores[key] = round(float(min(1.0, combined)), 4)

    # Normalise against the strongest line so the heatmap always has a clear
    # peak, whatever the absolute overlap happened to be.
    top = max(scores.values(), default=0.0)
    if top > 0:
        scores = {key: round(value / top, 4) for key, value in scores.items()}
    return scores


def _select_relevant(scores: dict, floor: float = 0.30, limit: int = 4) -> set:
    """Keep only the statements that genuinely stand out.

    A fixed keyword count admitted anything sharing a common word, which listed
    eight near-identical contributions. Cutting at the widest score gap swung the
    other way: scores are normalised against the top line, so the widest gap sits
    immediately below it and the rule collapsed to a single 100% source. A
    threshold relative to the best score, plus a hard limit, does the job without
    either failure.
    """
    ranked = sorted(
        ((score, key) for key, score in scores.items() if score > 0),
        reverse=True,
    )
    if not ranked:
        return set()

    cutoff = max(floor, ranked[0][0] * 0.35)
    return {key for score, key in ranked[:limit] if score >= cutoff}



# ── OCR ─────────────────────────────────────────────────────────────────────

def preprocess_for_ocr(image: Image.Image, invert: bool = False) -> Image.Image:
    """Prepare an image for Tesseract.

    A single global threshold at 150 turned light-on-dark text into a solid
    block, which is how headings and captions on coloured panels went missing.
    An adaptive local threshold handles mixed backgrounds, and `invert` lets the
    caller try the light-on-dark reading too.
    """
    from PIL import ImageEnhance

    image = image.resize(
        (image.width * 2, image.height * 2), Image.LANCZOS
    ).convert("L")
    image = ImageEnhance.Contrast(image).enhance(1.8)
    image = image.filter(ImageFilter.SHARPEN)
    if invert:
        image = ImageOps.invert(image)

    # Compare each pixel with a blurred local mean instead of a fixed cutoff,
    # so a dark panel and a white margin can both threshold correctly.
    pixels = np.asarray(image, dtype=np.float32)
    background = np.asarray(
        image.filter(ImageFilter.GaussianBlur(radius=12)), dtype=np.float32
    )
    binarized = np.where(pixels < background - 8, 0, 255).astype(np.uint8)
    return Image.fromarray(binarized)


def _read_image_text(image: Image.Image) -> tuple[dict, list[str], Image.Image]:
    """OCR an image, trying both polarities and keeping the better read.

    Returns (ocr_data, clean_lines, the variant that won) so bounding boxes and
    text come from the same pass.
    """
    best = None
    for invert in (False, True):
        variant = preprocess_for_ocr(image, invert=invert)
        try:
            # psm 3 (full auto page segmentation). psm 6 assumes one uniform
            # block and shredded multi-column layouts into interleaved noise.
            data = pytesseract.image_to_data(
                variant, config=r"--oem 3 --psm 3",
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:
            print(f"[OCR error] {type(exc).__name__}: {exc}")
            continue

        # Score by how much confident text was recovered, not raw character
        # count: a bad polarity yields many low-confidence fragments.
        words, confidence = [], []
        for index, word in enumerate(data.get("text", [])):
            token = str(word or "").strip()
            if not token:
                continue
            try:
                conf = float(data["conf"][index])
            except (TypeError, ValueError, IndexError):
                conf = 0.0
            if conf >= 45:
                words.append(token)
                confidence.append(conf)
        score = sum(len(w) for w in words) * (
            (sum(confidence) / len(confidence) / 100.0) if confidence else 0.0
        )
        if best is None or score > best[0]:
            best = (score, data, variant)

    if best is None:
        return {}, [], image

    _, data, variant = best
    # Report whole statements rather than visual rows: the UI shows these as the
    # extracted text, and a wrapped bullet split across rows reads as gibberish.
    lines = _group_ocr_by_line(data)
    clean_lines = _dedupe_lines(
        [block["text"] for block in _group_into_blocks(lines, variant.width).values()]
    )
    return data, clean_lines, variant


def _compose_answer(question: str, blocks: list[str]) -> tuple[str, float]:
    """Answer from the image text using extractive QA over statement blocks.

    The previous fallback handed a written-out instruction prompt to a
    summarisation model, which echoed the instructions back as the answer —
    the "list all of them as a numbered list" text seen in the UI. Extractive
    QA cannot invent text, so whatever it returns is on the page.
    """
    blocks = [_clean_ocr_line(block) for block in blocks if _clean_ocr_line(block)]
    if not blocks:
        return "No readable text was found in the image.", 0.0

    # Bullets on a page carry no terminal punctuation, so a bare join let a QA
    # span run from the middle of one statement into the next — which is how the
    # answer came back as a page-long run-on. A full stop bounds each block.
    context = " ".join(
        block if block.endswith((".", "!", "?")) else block + "."
        for block in blocks
    )

    span, confidence = answer_question(question, context)
    if not span or span.startswith("[QA error"):
        return "The image text does not appear to answer that question.", 0.0

    # Prefer the block containing the span: a bare span ("strictly regulated")
    # reads as a fragment, the sentence around it reads as an answer.
    span_normalized = re.sub(r"\s+", " ", span.lower()).strip()
    for block in blocks:
        if span_normalized and span_normalized in re.sub(r"\s+", " ", block.lower()):
            return block, float(confidence)

    span_terms = _content_terms(span)
    if span_terms:
        best, overlap = None, 0
        for block in blocks:
            shared = len(span_terms & _content_terms(block))
            if shared > overlap:
                best, overlap = block, shared
        if best and overlap >= max(1, len(span_terms) // 2):
            return best, float(confidence)

    return span, float(confidence)


# Stage weights, as shares of one request. Measured, not guessed: on a 2079x756
# graphic and a 1240x1754 text page the hosted vision call took 15.6s while the
# whole local remainder took 1.4s and 2.6s — 86-92% against 8-12% for the OCR
# pass and under 2% for scoring and the heatmap. The first table gave the vision
# call 46%, which left the bar sitting at 4% for 15 of an 18-second request.
IMAGE_STAGES = [
    ("Reading image", 1),
    ("Analysing image", 87),
    ("Reading text in image", 9),
    ("Scoring evidence", 1),
    ("Building heatmap", 2),
]


def explain_image_gemini(image: Image.Image, question: str, reporter=None):
    """Gemini-powered image VQA with OCR heatmap and source attribution.

    `reporter` is optional and only needs `.stage()` (see progress.Reporter); its
    labels are the ones in `IMAGE_STAGES`. Stages are reported at the real substep
    boundaries below, so the bar moves when work actually completes.
    """
    def stage(label):
        if reporter:
            reporter.stage(label)

    try:
        stage("Reading image")
        image = image.convert("RGB")
        original_image = image.copy()  # keep for display + heatmap
        question = question.strip()
        if not question:
            raise ValueError("Question is required.")

        # SUBSTEP A — Convert image to base64 for Gemini
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        # SUBSTEP B — Build the strict JSON prompt
        prompt = """
        Look at this image carefully and answer the question below.

        Return ONLY a valid JSON object, no markdown, no extra text:
        {
          "answerable": true,
          "answer": "your complete generalized answer here",
          "sources": [
            {
              "text": "exact text or region from image you used",
              "contribution": 0.6,
              "reason": "why this part contributed to the answer"
            },
            {
              "text": "another piece of text from image",
              "contribution": 0.4,
              "reason": "why this part contributed"
            }
          ]
        }

        Rules:
        - Answer must be complete and generalized, not just extracted
        - If question asks for multiple points, list all of them
        - Sources must be exact text visible in the image
        - Contribution values must add up to 1.0
        - Between 2 and 5 sources when the image can answer the question
        - Never add information not present in the image
""" + not_answerable_rules("image") + """

        Question: """ + question

        # SUBSTEP C — Call Gemini API. A per-minute quota clears on its own, so a
        # short retry rides that out instead of dropping straight to the weaker
        # local path. A spent daily allowance is different: it will refuse the
        # retry too, and sleeping on it cost 18 seconds before the fallback even
        # started.
        stage("Analysing image")
        model = gemini_model()
        response = None
        last_error = None
        for attempt in range(3):
            try:
                response = model.generate_content([
                    {"mime_type": "image/png", "data": image_bytes},
                    prompt
                ])
                break
            except Exception as exc:
                last_error = exc
                transient = "429" in str(exc) or "quota" in str(exc).lower()
                retryable = transient and not is_exhausted_for_today(exc)
                if not retryable or attempt == 2:
                    raise
                delay = 6 * (attempt + 1)
                print(f"[Gemini] quota hit, retrying in {delay}s")
                time.sleep(delay)
        if response is None:  # pragma: no cover - defensive
            raise last_error or RuntimeError(
                "The hosted analysis service returned no response"
            )

        raw = response.text.strip()
        
        # Remove markdown fences
        raw = re.sub(r"^```json\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"^```\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"```$", "", raw).strip()
        
        # Strip any preamble before the first {
        brace_start = raw.find("{")
        if brace_start > 0:
            raw = raw[brace_start:]
        
        # Strip anything after the last }
        brace_end = raw.rfind("}")
        if brace_end != -1 and brace_end < len(raw) - 1:
            raw = raw[:brace_end + 1]
        
        # Parse
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[Image JSON error] {e}")
            print(f"[Raw Gemini output] {raw[:500]}")
            raise ValueError(
                f"The hosted analysis service returned invalid JSON: {str(e)}"
            )
        
        answer = parsed.get("answer", "").strip()
        sources = parsed.get("sources", [])

        # The image has nothing to say about what was asked. OCR, scoring and the
        # heatmap all exist to point at evidence, and there is none — running them
        # is how "what is a dog" ended up lighting up whichever lines were least
        # unrelated. The picture still goes back so the reader sees what they sent.
        if answer and reply_is_unsupported(parsed, answer):
            stage("Building heatmap")
            return {
                "answer": answer,
                "sources": [],
                "chart_data": [],
                "highlighted_lines": [],
                "ocr_text": [],
                "original_b64": _image_to_b64(original_image),
                "heatmap_b64": None,
                "unsupported": True,
            }

        # SUBSTEP D — OCR the original image for bounding boxes and text.
        # Boxes are scaled back to the original resolution so the heatmap lines
        # up with the picture the user sees.
        stage("Reading text in image")
        ocr_data, clean_lines, variant = _read_image_text(original_image)
        scale_x = original_image.width / max(1, variant.width)
        scale_y = original_image.height / max(1, variant.height)
        ocr_lines = _group_ocr_by_line(ocr_data, scale_x, scale_y)

        # Match against whole statements, not visual rows: a row-level match
        # quoted half a clause and lit only one line of a three-line bullet.
        blocks = _group_into_blocks(ocr_lines, original_image.width)

        # SUBSTEP E — Score each source against the OCR text it came from.
        # Scoring by relevance rather than by a shared-keyword count stops
        # every line on the page from matching when they share a common word.
        stage("Scoring evidence")
        source_match_scores = []
        block_scores: dict = {}

        # Heat is graded against the largest contribution, not against 1.0.
        # Weighting by the raw value left five equally-cited sources at 0.2 each,
        # so the strongest evidence on the page never rose past the blue end of
        # the ramp and the map read as uniformly weak.
        strongest = max(
            (float(s.get("contribution", 0.0)) for s in sources), default=0.0
        ) or 1.0

        for source in sources:
            source_text = source.get("text", "")
            contribution = float(source.get("contribution", 0.0))
            reason = source.get("reason", "")

            scores = _score_ocr_lines(blocks, source_text, answer)
            selected = _select_relevant(scores, floor=0.45)

            # Weight this source's heat by its share of the largest
            # contribution, so the heatmap emphasises what drove the answer.
            weight = 0.45 + 0.55 * min(1.0, contribution / strongest)
            for key in selected:
                block_scores[key] = max(
                    block_scores.get(key, 0.0), scores.get(key, 0.0) * weight
                )

            source_match_scores.append({
                "text": source_text,
                "contribution": round(contribution * 100, 1),
                "reason": reason,
                "matched_line_keys": [
                    list(row) for key in sorted(selected)
                    for row in blocks[key]["keys"]
                ],
                "matched_blocks": [key for key in sorted(selected)],
            })

        # Fall back to question relevance if no source matched any text, so the
        # heatmap is never blank just because Gemini paraphrased the image.
        if not block_scores:
            scores = _score_ocr_lines(blocks, question, answer)
            block_scores = {
                key: scores[key] for key in _select_relevant(scores)
            }

        # SUBSTEP F — Build the graded heatmap. Neighbouring lines are no longer
        # pulled in: expanding ±1 line highlighted text the answer never used.
        # A block's score covers every row it wraps across, so multi-line
        # evidence is highlighted in full rather than only on its first row.
        stage("Building heatmap")
        heat = {}
        for key, score in block_scores.items():
            for row in blocks.get(key, {}).get("keys", [key]):
                heat[row] = max(heat.get(row, 0.0), score)
        heatmap_image = _build_heatmap(original_image, ocr_lines, heat)

        # SUBSTEP G — Evidence text, strongest statement first, deduplicated.
        highlighted_lines = []
        for source in source_match_scores:
            ranked = sorted(
                source.pop("matched_blocks", []),
                key=lambda k: block_scores.get(k, 0.0),
                reverse=True,
            )
            for key in ranked:
                if key in blocks:
                    highlighted_lines.append(blocks[key]["text"])
        highlighted_lines = _dedupe_lines(highlighted_lines)

        # Full phrase, not a 40-char prefix: the legend needs to be
        # readable and to match the evidence card word for word.
        chart_data = [
            {
                "label": source["text"],
                "full_text": source["text"],
                "contribution": source["contribution"],
                "reason": source["reason"],
                "color": ""  # frontend will assign colors
            }
            for source in source_match_scores
        ]

        # SUBSTEP H — Return
        return {
            "answer": answer,
            "sources": source_match_scores,
            "chart_data": chart_data,
            "highlighted_lines": highlighted_lines,
            "ocr_text": clean_lines,
            "original_b64": _image_to_b64(original_image),
            "heatmap_b64": _image_to_b64(heatmap_image),
            "unsupported": False,
        }

    except Exception as exc:
        # SUBSTEP I — Fallback to the local OCR path
        print(f"[Gemini error] {type(exc).__name__}: {exc}")
        # The local path keeps reporting inside the "Analysing image" span rather
        # than advancing to the later stages: from the reader's side the step
        # still running is "work out what this image says", and the stages after
        # it are ones this path never reaches. Without this the bar froze for the
        # whole fallback — which is most of the request once the quota is spent.
        result = explain_image_vqa(
            image,
            question,
            report=reporter.sub("Analysing image") if reporter else None,
        )
        # The raw provider text ("429 You exceeded your current quota ...
        # quota_metric: generativelanguage.googleapis.com/...") used to be printed
        # straight into the UI. The reader gets one actionable sentence instead,
        # plus a note that what they are looking at came from the local path; the
        # detail stays in the log above.
        result["error"] = (
            f"{describe_service_error(exc)} The answer below was read from the "
            "image on this machine instead."
        )
        return result



def _columns_of(ocr_lines: dict, keys: list, page_width: int) -> list[list]:
    """Split rows into text columns by clustering their left edges.

    A two-column infographic interleaves vertically, so rows have to be read
    column by column. Tesseract's own block order put the second half of a
    bullet before its first half, which truncated the citation mid-clause.
    """
    if not keys:
        return []
    ordered = sorted(keys, key=lambda k: ocr_lines[k]["left"])
    separation = max(60, int(page_width * 0.15))
    columns = [[ordered[0]]]
    for key in ordered[1:]:
        previous = ocr_lines[columns[-1][-1]]["left"]
        if ocr_lines[key]["left"] - previous > separation:
            columns.append([key])
        else:
            columns[-1].append(key)
    return [
        sorted(column, key=lambda k: (ocr_lines[k]["top"], ocr_lines[k]["left"]))
        for column in columns
    ]


def _group_into_blocks(ocr_lines: dict, page_width: int = 0) -> dict:
    """Merge wrapped OCR rows into statement-level blocks, keeping their boxes.

    Scoring individual rows meant a citation could stop mid-clause and the
    heatmap could light only the first line of a three-line bullet. A block keeps
    every row it spans, so both the quote and the highlight are complete.

    Merging is decided geometrically. Testing a row's first character instead
    ("does it start lowercase?") split "...in Great" from "Britain." and joined a
    decorative numeral into the prose; line spacing separates a wrapped line from
    the next bullet far more reliably.
    """
    if not ocr_lines:
        return {}
    page_width = page_width or max(line["right"] for line in ocr_lines.values())

    blocks: dict = {}
    for column in _columns_of(ocr_lines, list(ocr_lines), page_width):
        current_key = None
        for key in column:
            row = ocr_lines[key]
            text = _line_text(row)
            if not text:
                continue
            current = blocks.get(current_key)
            if current is not None and _continues_block(current, row, text):
                current["text"] = f"{current['text']} {text}"
                current["keys"].append(key)
                current["left"] = min(current["left"], row["left"])
                current["top"] = min(current["top"], row["top"])
                current["right"] = max(current["right"], row["right"])
                current["bottom"] = max(current["bottom"], row["bottom"])
                continue
            # Decoration that continues nothing is skipped rather than started as
            # a block: a callout numeral sitting between two rows of one bullet
            # otherwise breaks the chain and swallows the rest of the sentence.
            if not _is_usable_line(text):
                continue
            blocks[key] = {
                "text": text,
                "keys": [key],
                "left": row["left"], "top": row["top"],
                "right": row["right"], "bottom": row["bottom"],
            }
            current_key = key

    # Only blocks that read as prose can be cited; the rest are decoration.
    return {
        key: block for key, block in blocks.items()
        if _is_usable_line(block["text"])
    }


def _continues_block(block: dict, row: dict, text: str) -> bool:
    """Is this row a wrapped continuation of the block above it?"""
    if block["text"].endswith((".", "!", "?", ":", ";")):
        return False
    # A wrapped line is never outdented past its own column margin, which is
    # what tells prose apart from a callout number set off to the side.
    if row["left"] < block["left"] - 12:
        return False
    if abs(row["left"] - block["left"]) > max(
        40, (block["right"] - block["left"]) * 0.35
    ):
        return False
    # Line spacing within a paragraph is a fraction of the line height; the gap
    # to the next bullet is several times larger.
    line_height = max(8, row["bottom"] - row["top"])
    return row["top"] - block["bottom"] <= max(6, line_height * 0.55)


# ── OCR-only path (used when Gemini is unavailable) ─────────────────────────

# The two replies _compose_answer gives when the page cannot answer at all.
# Matching on them is what lets the caller tell "here is the statement that
# answers it" apart from "there isn't one" — both come back as a plain string.
_NO_ANSWER_REPLIES = {
    "The image text does not appear to answer that question.",
    "No readable text was found in the image.",
}


def _image_declined(answer: str, image: Image.Image, clean_lines: list) -> dict:
    """The image path's result when there is no evidence to attribute.

    Sources, chart and heat map go out empty on purpose. The extracted text
    stays: it is what the reader can check the refusal against, and it is not a
    claim about the answer. A null heatmap tells the browser to show the
    original picture with a note rather than a map of nothing.
    """
    return {
        "answer": answer,
        "confidence": 0.0,
        "sources": [],
        "chart_data": [],
        "highlighted_lines": [],
        "ocr_text": clean_lines,
        "original_b64": _image_to_b64(image),
        "heatmap_b64": None,
        "unsupported": True,
    }


def explain_image_vqa(image: Image.Image, question: str, report=None):
    """Answer from OCR text alone, with a relevance-graded heatmap.

    This path runs whenever Gemini is unreachable — including when the free-tier
    quota is spent — so it has to stand on its own. It reports the same shape as
    the Gemini path (sources, chart_data) so the UI renders identically.

    `report` is an optional callable taking 0.0..1.0, called at the three real
    substep boundaries: text read off the page, answer composed, heatmap drawn.
    The OCR pass and the extractive answer are the two costs here, so those are
    the boundaries worth reporting.
    """
    def progress(fraction):
        if report:
            report(fraction)

    image = image.convert("RGB")
    question = question.strip()
    if not question:
        raise ValueError("Question is required.")

    ocr_data, clean_lines, variant = _read_image_text(image)
    scale_x = image.width / max(1, variant.width)
    scale_y = image.height / max(1, variant.height)
    ocr_lines = _group_ocr_by_line(ocr_data, scale_x, scale_y)

    # Score statements, not visual rows, so a citation is never a half-clause.
    blocks = _group_into_blocks(ocr_lines, image.width)
    progress(0.35)

    # Nothing the page says has anything to do with what was asked, so there is
    # no evidence to rank and no heat to paint. Settled from the text alone,
    # before the extractive reader runs — it has to return *some* span, and the
    # least unrelated line on the page is not an answer.
    page_text = " ".join(clean_lines)
    if question_is_off_topic(question, page_text):
        progress(1.0)
        return _image_declined(
            unsupported_reply("image", question, page_text), image, clean_lines
        )

    # Answer from the statements on the page. Titles and logos are page
    # furniture — reading the answer out of one gave "UNDERSTANDING ANIMAL
    # RESEARCH" as the reply to a question about dogs.
    statements = [
        block["text"] for block in blocks.values() if not _is_heading(block["text"])
    ]
    answer, confidence = _compose_answer(
        question, statements or [block["text"] for block in blocks.values()]
    )
    progress(0.7)

    block_scores = _score_ocr_lines(blocks, question, answer)
    selected = _select_relevant(block_scores)

    # Either the reader found no span worth returning, or not one statement on
    # the page scored above zero against the question and the answer. Both mean
    # there is nothing to cite, and citing anyway is what produced two filler
    # sources at 50% each.
    if answer in _NO_ANSWER_REPLIES or not selected:
        progress(1.0)
        reply = (
            unsupported_reply("image", question, page_text)
            if page_text
            else "No readable text was found in the image."
        )
        return _image_declined(reply, image, clean_lines)

    ranked = sorted(selected, key=lambda k: block_scores.get(k, 0.0), reverse=True)
    highlighted_lines = _dedupe_lines(
        [blocks[key]["text"] for key in ranked if key in blocks]
    )

    # Spread each block's score over every row it spans so multi-line evidence
    # is highlighted in full.
    heatmap_scores = {}
    for key in selected:
        for row in blocks[key]["keys"]:
            heatmap_scores[row] = block_scores.get(key, 0.0)
    heatmap_image = _build_heatmap(image, ocr_lines, heatmap_scores)
    progress(1.0)

    # Express each cited block's share of the total relevance, so the chart and
    # the contribution bars mean the same thing they do on the Gemini path.
    total = sum(max(0.0, block_scores.get(key, 0.0)) for key in ranked) or 1.0
    sources = []
    for key in ranked:
        text = blocks[key]["text"]
        if not text:
            continue
        share = max(0.0, block_scores.get(key, 0.0)) / total
        sources.append({
            "text": text,
            "contribution": round(share * 100, 1),
            "reason": (
                f"Relevance {block_scores.get(key, 0.0):.2f} to the question"
                + (" and to the extracted answer" if answer else "")
            ),
            "matched_line_keys": [list(row) for row in blocks[key]["keys"]],
        })

    chart_data = [
        {
            "label": source["text"],
            "full_text": source["text"],
            "contribution": source["contribution"],
            "reason": source["reason"],
            "color": "",
        }
        for source in sources
    ]

    return {
        "answer": answer,
        "confidence": confidence,
        "sources": sources,
        "chart_data": chart_data,
        "highlighted_lines": highlighted_lines,
        "ocr_text": clean_lines,
        "original_b64": _image_to_b64(image),
        "heatmap_b64": _image_to_b64(heatmap_image),
        "unsupported": False,
    }



# ── LIME / SHAP / Sentence Attribution (unchanged) ─────────────────────────

