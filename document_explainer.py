"""PDF and slide-deck analysis: per-page attribution, with handwriting support.

The image tab already answers a question about one picture and reports what the
answer rests on — a graded heatmap, quoted evidence, per-source contributions.
None of that machinery is actually specific to photographs: it needs text that
carries a bounding box, and nothing more. So this module does not reimplement any
of it. It converts a PDF or a .pptx into a list of *pages*, each carrying boxed
lines in exactly the shape `_group_ocr_by_line()` produces, and then hands those
to the same scorer and the same heatmap painter the image tab uses.

Three things a document needs that a single image does not:

* **Page identity.** A source has to be able to say "Slide 3", so every quote is
  tagged with the page it came from and each page gets its own heatmap.
* **Quota discipline.** The free tier allows 20 calls a day, so a 25-page PDF
  cannot become 25 calls. Pages are ranked locally first and only the best few
  are sent, in a single request.
* **Handwriting.** Tesseract reads print well and cursive not at all. Two free
  offline engines are wired in behind a per-run choice. EasyOCR recognises words
  and is quick; TrOCR recognises a whole line and is slower. Measured on a
  handwritten test page, EasyOCR recovered every word but split three of six
  lines into fragments it then ordered wrongly, while TrOCR transcribed all six
  exactly — which is the case for offering both rather than picking one. Both are
  optional: if neither imports, the page still analyses on Tesseract and the
  response says so, because a missing model is not a reason to fail an upload.
"""

import os
import re
import time
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    from .image_explainer import (
        _NO_ANSWER_REPLIES as _IMAGE_NO_ANSWER_REPLIES,
        _build_heatmap,
        _compose_answer,
        _dedupe_lines,
        _group_into_blocks,
        _group_ocr_by_line,
        _is_heading,
        _read_image_text,
        _score_ocr_lines,
        _select_relevant,
    )
    from .text_explainer import _containment
    from .utils import (
        _image_to_b64,
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
    from image_explainer import (
        _NO_ANSWER_REPLIES as _IMAGE_NO_ANSWER_REPLIES,
        _build_heatmap,
        _compose_answer,
        _dedupe_lines,
        _group_into_blocks,
        _group_ocr_by_line,
        _is_heading,
        _read_image_text,
        _score_ocr_lines,
        _select_relevant,
    )
    from text_explainer import _containment
    from utils import (
        _image_to_b64,
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
    "DOCUMENT_STAGES",
    "explain_document",
    "explain_document_local",
    "read_document_pages",
    "DocumentFormatError",
]


class DocumentFormatError(ValueError):
    """The upload cannot be read, with a message already fit to show a user."""


# Shares of one request, from timing the real work. Reading pages dominates
# whenever OCR is involved — a rendered page is fast, recognising ink on it is
# not — and it is the only stage that reports its position internally, which it
# can do honestly because it knows the page count up front.
DOCUMENT_STAGES = [
    ("Reading pages", 58),
    ("Ranking pages", 6),
    ("Answering question", 24),
    ("Building heatmaps", 12),
]


# A long PDF is legitimate input, but every page costs a render plus (sometimes)
# an OCR pass, and every page ships two PNGs back to the browser. 25 keeps a
# lecture deck or a chapter scan whole while bounding both.
_MAX_PAGES = int(os.getenv("DOC_MAX_PAGES", "25"))

# Only this many page images are sent for the hosted answer. Ranking happens
# locally first, so these are the pages most likely to contain the answer.
_GEMINI_PAGES = int(os.getenv("DOC_GEMINI_PAGES", "3"))

# 150 dpi is the point where Tesseract stops losing 9pt body text; going higher
# mostly costs render time.
_PDF_DPI = int(os.getenv("DOC_PDF_DPI", "150"))

# Page images are shipped to the browser as base64, so they are downscaled for
# display. The heatmap is painted at this size too, which keeps the two aligned.
_DISPLAY_MAX_WIDTH = int(os.getenv("DOC_DISPLAY_WIDTH", "1000"))

# Below this much embedded text a PDF page is a scan, not a digital page, and has
# to be read with OCR rather than trusted.
_DIGITAL_TEXT_FLOOR = 40

_TROCR_MODEL = os.getenv("DOC_TROCR_MODEL", "microsoft/trocr-small-handwritten")

# Line crops are pushed through TrOCR together. One at a time costs roughly
# 0.3s per line; a batch amortises the encoder and roughly halves a full page.
_TROCR_BATCH = int(os.getenv("DOC_TROCR_BATCH", "8"))

# Mean Tesseract confidence below this, over a page with real word count, is the
# signature of handwriting rather than of a bad scan of print.
_HANDWRITING_CONF = 60.0

_SLIDE_WIDTH_PX = 1280


# ── The uniform OCR record ──────────────────────────────────────────────────
#
# Every ingestion path — digital PDF text, EasyOCR, TrOCR, Tesseract, drawn
# slide text — funnels through _add_word() so it produces the same dict that
# pytesseract.image_to_data() does. That is the one contract that lets
# _group_ocr_by_line(), _group_into_blocks(), _score_ocr_lines() and
# _build_heatmap() be reused without a single change.

_OCR_FIELDS = (
    "text", "conf", "left", "top", "width", "height",
    "block_num", "par_num", "line_num",
)


def _blank_ocr_data() -> dict:
    return {field: [] for field in _OCR_FIELDS}


def _add_word(
    data: dict,
    text: str,
    box: tuple,
    block: int = 0,
    par: int = 0,
    line: int = 0,
    conf: float = 95.0,
) -> None:
    """Append one word, in image pixel coordinates, to an OCR record."""
    word = str(text or "").strip()
    if not word:
        return
    left, top, right, bottom = (float(v) for v in box)
    data["text"].append(word)
    data["conf"].append(float(conf))
    data["left"].append(int(round(left)))
    data["top"].append(int(round(top)))
    data["width"].append(max(1, int(round(right - left))))
    data["height"].append(max(1, int(round(bottom - top))))
    data["block_num"].append(int(block))
    data["par_num"].append(int(par))
    data["line_num"].append(int(line))


def _spread_words(
    data: dict,
    text: str,
    box: tuple,
    block: int = 0,
    par: int = 0,
    line: int = 0,
    conf: float = 90.0,
) -> None:
    """Split a recognised line into words, interpolating a box for each.

    EasyOCR and TrOCR both return a whole line at once. Storing that as a single
    "word" would defeat `_prose_indices()`, which drops callout numerals by
    looking at the gaps *between* words, and would coarsen the heatmap to
    whole-line rectangles. Character offsets are a good enough proxy for where
    each word sits inside a line box.
    """
    words = str(text or "").split()
    if not words:
        return
    left, top, right, bottom = (float(v) for v in box)
    span = max(1.0, right - left)
    total = sum(len(word) for word in words) + max(0, len(words) - 1)
    cursor = 0
    for word in words:
        start = left + span * (cursor / total)
        cursor += len(word)
        end = left + span * (cursor / total)
        cursor += 1  # the space that follows
        _add_word(data, word, (start, top, end, bottom), block, par, line, conf)


def _mean_confidence(data: dict) -> tuple:
    """(mean confidence, word count) over words a reader would call words."""
    scores = []
    for index, raw in enumerate(data.get("text", [])):
        word = str(raw or "").strip()
        if len(word) < 2:
            continue
        try:
            conf = float(data["conf"][index])
        except (TypeError, ValueError, IndexError):
            continue
        if conf >= 0:
            scores.append(conf)
    if not scores:
        return 0.0, 0
    return sum(scores) / len(scores), len(scores)


def _looks_handwritten(data: dict) -> bool:
    """Did Tesseract find text it could not read?

    Tesseract locates handwritten lines competently and then reports very low
    confidence on the words inside them. That gap is the useful signal: plenty of
    boxes, poor confidence, means an HTR engine should get a look at the page.
    """
    mean, count = _mean_confidence(data)
    return count >= 5 and mean < _HANDWRITING_CONF


# ── Optional engines, imported lazily ───────────────────────────────────────
#
# Both are cached after the first success and after the first failure, so a
# machine without the model downloaded pays the import error once rather than on
# every page of every upload.

_easyocr_reader = None
_easyocr_broken = False
_trocr_pair = None
_trocr_broken = False


def _get_easyocr():
    """The EasyOCR reader, or None if it is unavailable on this machine."""
    global _easyocr_reader, _easyocr_broken
    if _easyocr_reader is not None or _easyocr_broken:
        return _easyocr_reader
    try:
        import easyocr

        # gpu=False is explicit: the default probes CUDA and logs a warning on a
        # CPU-only install, which reads like an error in the server log.
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    except Exception as exc:  # noqa: BLE001 - any failure degrades to Tesseract
        print(f"[EasyOCR] unavailable: {type(exc).__name__}: {exc}")
        _easyocr_broken = True
    return _easyocr_reader


def _get_trocr():
    """(processor, model) for handwriting, or None if unavailable."""
    global _trocr_pair, _trocr_broken
    if _trocr_pair is not None or _trocr_broken:
        return _trocr_pair
    try:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        processor = TrOCRProcessor.from_pretrained(_TROCR_MODEL)
        model = VisionEncoderDecoderModel.from_pretrained(_TROCR_MODEL)
        model.eval()
        torch.set_grad_enabled(False)
        _trocr_pair = (processor, model)
    except Exception as exc:  # noqa: BLE001 - any failure degrades to Tesseract
        print(f"[TrOCR] unavailable: {type(exc).__name__}: {exc}")
        _trocr_broken = True
    return _trocr_pair


def _trocr_read(crops: list) -> list:
    """Read a list of line-crop images, in batches. Empty list if unavailable."""
    pair = _get_trocr()
    if pair is None or not crops:
        return []
    processor, model = pair
    import torch

    texts = []
    for start in range(0, len(crops), _TROCR_BATCH):
        batch = crops[start:start + _TROCR_BATCH]
        try:
            pixel_values = processor(
                images=batch, return_tensors="pt"
            ).pixel_values
            with torch.no_grad():
                generated = model.generate(pixel_values, max_new_tokens=64)
            texts.extend(
                processor.batch_decode(generated, skip_special_tokens=True)
            )
        except Exception as exc:  # noqa: BLE001 - a bad batch must not lose the page
            print(f"[TrOCR] batch failed: {type(exc).__name__}: {exc}")
            texts.extend([""] * len(batch))
    return texts


# ── Reading ink off a page image ────────────────────────────────────────────

def _tesseract_ink(image: Image.Image) -> tuple:
    """Tesseract's read of a page, in the page's own pixel coordinates.

    `_read_image_text()` works on an upscaled, binarised variant, so its boxes
    are in that variant's space. They are scaled back here, because the heatmap
    is painted on the image the user actually sees.
    """
    data, _clean, variant = _read_image_text(image)
    if not data.get("text"):
        return _blank_ocr_data(), {}
    scale_x = image.width / max(1, variant.width)
    scale_y = image.height / max(1, variant.height)

    rescaled = _blank_ocr_data()
    for index, raw in enumerate(data.get("text", [])):
        word = str(raw or "").strip()
        if not word:
            continue
        try:
            conf = float(data["conf"][index])
        except (TypeError, ValueError, IndexError):
            conf = 0.0
        if conf < 0:
            continue
        left = float(data["left"][index]) * scale_x
        top = float(data["top"][index]) * scale_y
        right = left + float(data["width"][index]) * scale_x
        bottom = top + float(data["height"][index]) * scale_y
        _add_word(
            rescaled, word, (left, top, right, bottom),
            data.get("block_num", [0])[index],
            data.get("par_num", [0])[index],
            data.get("line_num", [0])[index],
            conf,
        )
    return rescaled, _group_ocr_by_line(rescaled)


def _easyocr_ink(image: Image.Image) -> dict:
    """EasyOCR's read: detection boxes and text in a single pass."""
    reader = _get_easyocr()
    if reader is None:
        return _blank_ocr_data()
    try:
        detections = reader.readtext(np.asarray(image.convert("RGB")))
    except Exception as exc:  # noqa: BLE001
        print(f"[EasyOCR] read failed: {type(exc).__name__}: {exc}")
        return _blank_ocr_data()

    data = _blank_ocr_data()
    # Reading order: top to bottom, then left to right. EasyOCR returns
    # detections in its own detector order, which interleaves columns.
    ordered = sorted(
        detections,
        key=lambda d: (min(p[1] for p in d[0]), min(p[0] for p in d[0])),
    )
    for line_index, detection in enumerate(ordered):
        box, text, confidence = detection[0], detection[1], detection[2]
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        _spread_words(
            data, text, (min(xs), min(ys), max(xs), max(ys)),
            block=0, par=0, line=line_index,
            conf=float(confidence) * 100.0,
        )
    return data


def _merge_rows(lines: dict, tolerance: float = 0.6) -> list:
    """Merge boxes that sit on the same row into one box per line of writing.

    A word detector reports fragments: on the test page EasyOCR returned
    "Counted eleven herons", "on" and "the east bank" as three separate boxes on
    one row. TrOCR reads a whole line far better than it reads a two-word scrap,
    so fragments that overlap vertically are combined before cropping.
    """
    boxes = sorted(
        (
            (row["top"], row["bottom"], row["left"], row["right"])
            for row in lines.values()
        ),
        key=lambda box: box[0],
    )
    rows: list = []
    for top, bottom, left, right in boxes:
        height = max(1.0, bottom - top)
        for row in rows:
            overlap = min(row[1], bottom) - max(row[0], top)
            if overlap > tolerance * min(height, row[1] - row[0]):
                row[0] = min(row[0], top)
                row[1] = max(row[1], bottom)
                row[2] = min(row[2], left)
                row[3] = max(row[3], right)
                break
        else:
            rows.append([top, bottom, left, right])
    return [(left, top, right, bottom) for top, bottom, left, right in rows]


def _line_geometry(image: Image.Image) -> tuple:
    """Find the lines of writing on a page. Returns (boxes, source).

    Tesseract was the first choice here on the assumption that its layout
    analysis holds up on handwriting even when its recognition does not. Measured
    on a handwritten test page, it did not: it located 4 of 6 lines and missed two
    outright, so two lines of the page could never have been read no matter how
    good the recogniser. EasyOCR's CRAFT detector found all six, so it leads and
    Tesseract is the fallback for when EasyOCR is unavailable.
    """
    detections = _easyocr_ink(image)
    if detections["text"]:
        rows = _merge_rows(_group_ocr_by_line(detections))
        if rows:
            return rows, "craft"

    _data, lines = _tesseract_ink(image)
    return _merge_rows(lines), "tesseract"


def _for_trocr(crop: Image.Image) -> Image.Image:
    """Prepare a line crop for TrOCR: greyscale, three channels.

    This one line is the difference between a transcript and gibberish. TrOCR
    normalises each channel against mean 0.5, std 0.5, and was trained on
    greyscale scans copied across all three; blue-black ink on cream paper puts
    the channels in places the encoder has never seen, and it answers by
    generating fluent text unrelated to the image. Measured on the same crop:
    RGB gave "standard percentages and redundant in the time of the time",
    greyscale gave "Sediment looked heavier than last visit." — the exact words.

    Greyscale, not a threshold: TrOCR reads stroke weight, so binarising loses
    information. Autocontrast scored identically on the test page and is left out
    as an unearned complication.
    """
    return ImageOps.grayscale(crop).convert("RGB")


def _htr_ink(image: Image.Image) -> tuple:
    """A detector finds the lines; TrOCR reads them. Returns (data, engine).

    TrOCR is a line recogniser with no layout model of its own, so it has to be
    told where the lines are. Taking the boxes from a detector has the useful side
    effect that the geometry the heatmap needs comes free. The engine label comes
    back with the data because the fallbacks below change who did the reading, and
    the browser reports that.
    """
    boxes, source = _line_geometry(image)
    if not boxes:
        return _blank_ocr_data(), "none"

    kept, crops = [], []
    for box in boxes:
        pad = 6
        left = max(0, int(box[0]) - pad)
        top = max(0, int(box[1]) - pad)
        right = min(image.width, int(box[2]) + pad)
        bottom = min(image.height, int(box[3]) + pad)
        # A crop only a few pixels tall is a speck, and TrOCR will hallucinate a
        # word for it rather than return nothing.
        if right - left < 12 or bottom - top < 8:
            continue
        kept.append((left, top, right, bottom))
        crops.append(_for_trocr(image.crop((left, top, right, bottom))))

    texts = _trocr_read(crops)
    if texts:
        rewritten = _blank_ocr_data()
        for index, (box, text) in enumerate(zip(kept, texts)):
            cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
            if not cleaned:
                continue
            _spread_words(
                rewritten, cleaned, box, block=0, par=0, line=index, conf=90.0
            )
        if rewritten["text"]:
            return rewritten, "trocr"

    # TrOCR is unavailable or read nothing. Whatever found the lines can also
    # report the text, and _line_geometry has already paid for that read.
    if source == "craft":
        return _easyocr_ink(image), "easyocr"
    return _tesseract_ink(image)[0], "tesseract"


def _read_page_ink(image: Image.Image, mode: str) -> tuple:
    """Read a page image with the engine the user asked for.

    Returns (ocr_data, engine label). The label is reported back to the browser,
    because "this was read by Tesseract after all" is something the reader needs
    to know when the text looks wrong.
    """
    if mode == "accurate":
        data, engine = _htr_ink(image)
        if data["text"]:
            return data, engine
        return _tesseract_ink(image)[0], "tesseract"

    data = _easyocr_ink(image)
    if data["text"]:
        return data, "easyocr"

    # EasyOCR is missing or found nothing. Tesseract is the floor, but if it
    # reports the low-confidence-with-many-boxes pattern the page is handwritten,
    # and handing back its guess would be worse than spending a moment on HTR.
    data, _lines = _tesseract_ink(image)
    if _looks_handwritten(data):
        upgraded, engine = _htr_ink(image)
        if upgraded["text"] and engine == "trocr":
            return upgraded, engine
    return data, "tesseract"


# ── PDF ─────────────────────────────────────────────────────────────────────

def _display_copy(image: Image.Image) -> Image.Image:
    """Downscale a page for the browser, preserving aspect ratio."""
    if image.width <= _DISPLAY_MAX_WIDTH:
        return image.convert("RGB")
    height = max(1, round(image.height * _DISPLAY_MAX_WIDTH / image.width))
    return image.convert("RGB").resize(
        (_DISPLAY_MAX_WIDTH, height), Image.LANCZOS
    )


def _pdf_pages(data: bytes, mode: str, report=None) -> tuple:
    """Render each PDF page and collect its text with boxes.

    A digital PDF already knows where every word sits, so OCR would only add
    error. `get_text("words")` is used when the page carries real text, and the
    rendered image is only OCR'd when the page turns out to be a scan.
    """
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - install-time problem
        raise DocumentFormatError(
            "PDF support needs the pymupdf package: pip install pymupdf"
        ) from exc

    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise DocumentFormatError(
            f"This file could not be opened as a PDF ({type(exc).__name__})."
        ) from exc

    if document.needs_pass:
        document.close()
        raise DocumentFormatError(
            "This PDF is password protected. Remove the password and try again."
        )

    pages = []
    try:
        total = document.page_count
        limit = min(total, _MAX_PAGES)
        for index in range(limit):
            page = document[index]
            pixmap = page.get_pixmap(dpi=_PDF_DPI)
            # tobytes("png") rather than frombytes(pix.samples): a page with an
            # alpha channel or a CMYK image changes pix.n, and reading the raw
            # buffer as RGB would shear the render.
            rendered = Image.open(BytesIO(pixmap.tobytes("png"))).convert("RGB")
            image = _display_copy(rendered)
            scale = image.width / max(1, rendered.width)

            words = page.get_text("words") or []
            embedded = "".join(str(word[4] or "") for word in words).strip()
            if len(embedded) >= _DIGITAL_TEXT_FLOOR:
                ocr = _blank_ocr_data()
                # PDF user space is 72 dpi; the render is at _PDF_DPI and then
                # downscaled for display, so both factors apply.
                factor = (_PDF_DPI / 72.0) * scale
                for x0, y0, x1, y1, word, block, line, _word_no in words:
                    _add_word(
                        ocr, word,
                        (x0 * factor, y0 * factor, x1 * factor, y1 * factor),
                        block, 0, line,
                    )
                engine = "digital"
            else:
                ocr, engine = _read_page_ink(image, mode)

            pages.append(
                _page_record(index, f"Page {index + 1}", image, ocr, engine)
            )
            if report:
                report((index + 1) / limit)
    finally:
        document.close()

    return pages, total


# ── PPTX ────────────────────────────────────────────────────────────────────
#
# Slides are redrawn from shape geometry rather than rasterised by an office
# suite. python-pptx reports every shape's text *and* its position, which is
# everything the heatmap needs, and it keeps the whole feature a pip install.

_SLIDE_BG = (18, 18, 27)
_SLIDE_FG = (232, 232, 240)
_SLIDE_ACCENT = (129, 140, 248)
_SLIDE_FRAME = (42, 42, 61)

_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)

_font_cache: dict = {}


def _slide_font(size_px: int):
    """A truetype face at this pixel size, or the bitmap default."""
    size_px = max(9, min(96, int(size_px)))
    if size_px in _font_cache:
        return _font_cache[size_px]
    face = None
    for path in _FONT_CANDIDATES:
        if os.path.isfile(path):
            try:
                face = ImageFont.truetype(path, size_px)
                break
            except Exception:  # noqa: BLE001 - try the next candidate
                continue
    if face is None:
        # load_default() ignores the size, so slide text will be uniformly small
        # rather than absent. Boxes still come out right, so attribution works.
        face = ImageFont.load_default()
    _font_cache[size_px] = face
    return face


def _wrap_to_width(text: str, font, max_width: float) -> list:
    """Greedy word wrap using the font's own measurements."""
    words = str(text or "").split()
    if not words:
        return []
    lines, current = [], words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if font.getlength(candidate) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_text_block(
    draw: ImageDraw.ImageDraw,
    ocr: dict,
    text: str,
    box: tuple,
    font,
    block: int,
    par: int,
    line_offset: int,
    colour: tuple,
) -> int:
    """Draw wrapped text inside a box and record a box for every word.

    Words are measured as they are drawn, so the recorded boxes are exact rather
    than estimated — which makes a slide heatmap tighter than an OCR'd one.
    """
    left, top, right, bottom = box
    max_width = max(20.0, right - left)
    wrapped = _wrap_to_width(text, font, max_width)
    if not wrapped:
        return line_offset

    ascent, descent = font.getmetrics()
    line_height = (ascent + descent) * 1.25
    y = top
    line_index = line_offset
    space = font.getlength(" ")

    for visual_line in wrapped:
        if y + line_height > bottom + line_height:
            # Overflowing text is clipped rather than drawn over the next shape.
            break
        x = left
        for word in visual_line.split():
            width = font.getlength(word)
            draw.text((x, y), word, font=font, fill=colour)
            _add_word(
                ocr, word,
                (x, y, x + width, y + ascent + descent),
                block, par, line_index,
            )
            x += width + space
        y += line_height
        line_index += 1

    return line_index


def _pptx_pages(data: bytes, mode: str, report=None) -> tuple:
    """Redraw each slide from its shapes and collect text with boxes."""
    try:
        from pptx import Presentation
        from pptx.util import Pt
    except ImportError as exc:  # pragma: no cover - install-time problem
        raise DocumentFormatError(
            "PowerPoint support needs the python-pptx package: "
            "pip install python-pptx"
        ) from exc

    try:
        presentation = Presentation(BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise DocumentFormatError(
            "This file could not be opened as a .pptx. If it is an older .ppt, "
            f"re-save it as .pptx and try again. ({type(exc).__name__})"
        ) from exc

    slide_width = int(presentation.slide_width or 9144000)
    slide_height = int(presentation.slide_height or 6858000)
    scale = _SLIDE_WIDTH_PX / slide_width  # pixels per EMU
    canvas_height = max(1, round(slide_height * scale))
    default_font_px = round(Pt(18).emu * scale)

    slides = list(presentation.slides)
    shown = slides[:_MAX_PAGES]
    pages = []
    for index, slide in enumerate(shown):
        image = Image.new("RGB", (_SLIDE_WIDTH_PX, canvas_height), _SLIDE_BG)
        draw = ImageDraw.Draw(image)
        ocr = _blank_ocr_data()
        counter = {"block": 0}
        pictures = []

        _render_shapes(
            slide.shapes, draw, image, ocr, counter, pictures,
            scale, default_font_px, offset=(0, 0),
        )

        # Text inside a photograph on a slide — a scanned note, a whiteboard
        # snap — is invisible to python-pptx, so each picture is read separately
        # and its boxes are offset into slide coordinates.
        for picture_box in pictures:
            _read_picture_region(image, ocr, picture_box, mode, counter)

        engine = "digital" if not pictures else f"digital+{mode}"
        pages.append(
            _page_record(index, f"Slide {index + 1}", image, ocr, engine)
        )
        if report:
            report((index + 1) / max(1, len(shown)))

    return pages, len(slides)


def _render_shapes(
    shapes,
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    ocr: dict,
    counter: dict,
    pictures: list,
    scale: float,
    default_font_px: int,
    offset: tuple,
) -> None:
    """Draw one shape tree, recursing into groups."""
    for shape in shapes:
        try:
            box = _shape_box(shape, scale, offset, image)
        except Exception:  # noqa: BLE001 - a shape without geometry is skipped
            continue
        if box is None:
            continue

        shape_type = str(getattr(shape, "shape_type", "") or "")

        if "GROUP" in shape_type.upper():
            try:
                _render_shapes(
                    shape.shapes, draw, image, ocr, counter, pictures,
                    scale, default_font_px, offset,
                )
            except Exception:  # noqa: BLE001
                pass
            continue

        if getattr(shape, "shape_type", None) is not None and hasattr(shape, "image"):
            _paste_picture(shape, image, box, pictures)
            continue

        if getattr(shape, "has_table", False):
            _draw_table(shape, draw, ocr, box, counter, scale, default_font_px)
            continue

        if not getattr(shape, "has_text_frame", False):
            continue

        text_frame = shape.text_frame
        if not str(text_frame.text or "").strip():
            continue

        left, top, right, bottom = box
        counter["block"] += 1
        block = counter["block"]
        # A title is the one thing worth distinguishing visually: it is drawn in
        # the accent colour, which also makes the redraw readable as a slide.
        is_title = _is_title(shape)
        cursor_y = top
        line_index = 0

        for par_index, paragraph in enumerate(text_frame.paragraphs):
            text = str(paragraph.text or "").strip()
            if not text:
                cursor_y += default_font_px * 0.6
                continue
            size_px = _paragraph_font_px(paragraph, scale, default_font_px, is_title)
            font = _slide_font(size_px)
            indent = (getattr(paragraph, "level", 0) or 0) * size_px
            drawn_to = _draw_text_block(
                draw, ocr, text,
                (left + indent, cursor_y, right, bottom),
                font, block, par_index, line_index,
                _SLIDE_ACCENT if is_title else _SLIDE_FG,
            )
            ascent, descent = font.getmetrics()
            cursor_y += (ascent + descent) * 1.25 * max(1, drawn_to - line_index)
            line_index = drawn_to


def _shape_box(shape, scale: float, offset: tuple, image: Image.Image):
    """A shape's pixel box, or None when it has no usable geometry."""
    left, top = shape.left, shape.top
    width, height = shape.width, shape.height
    if left is None or top is None or width is None or height is None:
        return None
    x0 = float(left) * scale + offset[0]
    y0 = float(top) * scale + offset[1]
    x1 = x0 + float(width) * scale
    y1 = y0 + float(height) * scale
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return (
        max(0.0, x0), max(0.0, y0),
        min(float(image.width), x1), min(float(image.height), y1),
    )


def _is_title(shape) -> bool:
    try:
        placeholder = shape.placeholder_format
        return placeholder is not None and int(placeholder.idx) == 0
    except Exception:  # noqa: BLE001 - not a placeholder
        return False


def _paragraph_font_px(paragraph, scale: float, default_px: int, is_title: bool) -> int:
    """The paragraph's own point size in pixels, with a sane default."""
    for run in getattr(paragraph, "runs", []) or []:
        size = getattr(getattr(run, "font", None), "size", None)
        if size is not None:
            return max(9, round(float(size.emu) * scale))
    size = getattr(getattr(paragraph, "font", None), "size", None)
    if size is not None:
        return max(9, round(float(size.emu) * scale))
    return round(default_px * (1.6 if is_title else 1.0))


def _paste_picture(shape, image: Image.Image, box: tuple, pictures: list) -> None:
    """Paste an embedded picture at its real position on the slide."""
    try:
        blob = shape.image.blob
        picture = Image.open(BytesIO(blob)).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - unsupported media, e.g. EMF
        print(f"[PPTX] picture skipped: {type(exc).__name__}: {exc}")
        return
    left, top, right, bottom = (int(round(v)) for v in box)
    width, height = max(1, right - left), max(1, bottom - top)
    try:
        image.paste(picture.resize((width, height), Image.LANCZOS), (left, top))
    except Exception as exc:  # noqa: BLE001
        print(f"[PPTX] picture paste failed: {type(exc).__name__}: {exc}")
        return
    pictures.append((left, top, left + width, top + height))


def _read_picture_region(
    image: Image.Image, ocr: dict, box: tuple, mode: str, counter: dict
) -> None:
    """OCR one pasted picture and merge its boxes into the slide's record."""
    left, top, right, bottom = box
    if right - left < 40 or bottom - top < 24:
        return
    crop = image.crop((left, top, right, bottom))
    region, _engine = _read_page_ink(crop, mode)
    if not region["text"]:
        return
    counter["block"] += 1
    block = counter["block"]
    for index, word in enumerate(region["text"]):
        x0 = region["left"][index] + left
        y0 = region["top"][index] + top
        _add_word(
            ocr, word,
            (x0, y0, x0 + region["width"][index], y0 + region["height"][index]),
            block, region["par_num"][index], region["line_num"][index],
            region["conf"][index],
        )


def _draw_table(
    shape, draw: ImageDraw.ImageDraw, ocr: dict, box: tuple,
    counter: dict, scale: float, default_font_px: int,
) -> None:
    """Draw a slide table cell by cell, recording each cell's text."""
    try:
        table = shape.table
        rows = list(table.rows)
        columns = list(table.columns)
    except Exception:  # noqa: BLE001
        return
    if not rows or not columns:
        return

    left, top, right, bottom = box
    cell_width = (right - left) / len(columns)
    cell_height = (bottom - top) / len(rows)
    font = _slide_font(round(default_font_px * 0.85))
    counter["block"] += 1
    block = counter["block"]

    for row_index, row in enumerate(rows):
        for column_index, cell in enumerate(row.cells):
            text = str(getattr(cell, "text", "") or "").strip()
            cell_x = left + column_index * cell_width
            cell_y = top + row_index * cell_height
            draw.rectangle(
                [cell_x, cell_y, cell_x + cell_width, cell_y + cell_height],
                outline=_SLIDE_FRAME,
            )
            if not text:
                continue
            _draw_text_block(
                draw, ocr, text,
                (cell_x + 4, cell_y + 3, cell_x + cell_width - 4, cell_y + cell_height),
                font, block, row_index, column_index * 100, _SLIDE_FG,
            )


# ── Page assembly ───────────────────────────────────────────────────────────

def _page_record(
    index: int, label: str, image: Image.Image, ocr: dict, engine: str
) -> dict:
    """One page, with its text already grouped into statement-level blocks."""
    lines = _group_ocr_by_line(ocr)
    blocks = _group_into_blocks(lines, image.width) if lines else {}
    return {
        "index": index,
        "label": label,
        "image": image,
        "lines": lines,
        "blocks": blocks,
        "text_source": engine,
    }


def read_document_pages(
    data: bytes, filename: str, mode: str = "fast", report=None
) -> tuple:
    """Ingest an upload into page records. Raises DocumentFormatError.

    `report` is a plain `report(fraction)` callable so this module stays
    independent of the progress plumbing, matching what the other explainers take.

    Returns (pages, total_page_count) — the total can exceed len(pages) when the
    document is longer than _MAX_PAGES, which the response reports so the reader
    knows the tail was not analysed.
    """
    name = str(filename or "").lower().strip()
    mode = "accurate" if str(mode or "").lower() == "accurate" else "fast"

    if name.endswith(".pdf") or data[:5] == b"%PDF-":
        return _pdf_pages(data, mode, report)
    if name.endswith(".pptx"):
        return _pptx_pages(data, mode, report)
    if name.endswith(".ppt"):
        raise DocumentFormatError(
            "Legacy .ppt files use a binary format this reader cannot open. "
            "Open it in PowerPoint and re-save as .pptx, then upload again."
        )
    raise DocumentFormatError(
        "Upload a PDF or a .pptx presentation."
    )


def _document_statements(pages: list) -> list:
    """Every quotable statement in the document, page-tagged, in reading order."""
    statements = []
    for page in pages:
        for key, block in page["blocks"].items():
            statements.append({
                "page": page["index"],
                "label": page["label"],
                "key": key,
                "text": block["text"],
                "heading": _is_document_heading(block, page),
            })
    return statements


def _row_height(page: dict, block: dict) -> float:
    """Mean glyph height of the rows a block wraps across, in pixels."""
    heights = [
        page["lines"][key]["bottom"] - page["lines"][key]["top"]
        for key in block.get("keys", ())
        if key in page["lines"]
    ]
    return sum(heights) / len(heights) if heights else 0.0


def _body_height(page: dict) -> float:
    """The page's median line height — its body-text size."""
    heights = sorted(
        line["bottom"] - line["top"] for line in page["lines"].values()
    )
    if not heights:
        return 0.0
    return float(heights[len(heights) // 2])


def _is_document_heading(block: dict, page: dict) -> bool:
    """Is this block a section title rather than a statement of fact?

    `_is_heading()` keys on ALL CAPS, which is how posters set titles and how the
    image path finds them. A document sets them in Title Case at a larger type
    size instead, so "Limiting Factors" sails past it — and it was being returned
    as the answer to "what limits the rate of photosynthesis?", which asserts
    nothing and is precisely what that guard exists to prevent.

    Type size is the signal to lead with, because it is real typography rather
    than an orthographic guess: a PDF reports it exactly and a redrawn slide is
    measured as it is drawn. Capitalisation is only the tie-breaker for pages
    where every row is the same height, which is what an OCR'd page looks like.
    """
    text = str(block.get("text", "") or "").strip()
    if _is_heading(text):
        return True
    words = text.split()
    if not words or len(words) > 9:
        return False
    # A title does not end in sentence punctuation. A trailing colon is the one
    # exception, since "Limiting Factors:" is still a title.
    if text.endswith((".", "?", "!", ";", ",")):
        return False

    body = _body_height(page)
    if body > 0 and _row_height(page, block) > body * 1.18:
        return True
    # Fallback: nearly every word capitalised, and short enough to be a label.
    if len(words) > 6:
        return False
    capitalised = sum(1 for word in words if word[:1].isupper())
    return capitalised >= max(1, len(words) - 1)



# ── Hosted answer ───────────────────────────────────────────────────────────

_PROMPT = """
You are given {count} page image(s) from a document, each labelled above it.
Answer the question using only what is visible on those pages. Some pages may
contain handwriting; read it as carefully as you can.

Return ONLY a valid JSON object, no markdown, no extra text:
{{
  "answerable": true,
  "answer": "your complete answer here",
  "sources": [
    {{
      "text": "exact text copied from the page",
      "page": "the page label this came from, e.g. Page 2 or Slide 3",
      "contribution": 0.6,
      "reason": "why this part supported the answer"
    }}
  ]
}}

Rules:
- The answer must be complete and readable on its own, not a bare fragment
- If the question asks for several points, give all of them
- Every source must be text that is actually visible on the page you name
- Contribution values must add up to 1.0
- Between 2 and 5 sources when the pages can answer the question
- Never add information that is not in the pages
{not_answerable}

Question: {question}
"""


def _document_scores(pages: list, question: str, answer: str = "") -> dict:
    """Score every statement in the document in one pass. {page: {key: score}}.

    `_score_ocr_lines()` normalises against the best line *in the call*, which is
    right for a single page and wrong for a document. Scoring page by page hands
    every page's own best block a 1.0: measured on a five-page set, a page whose
    real relevance was 0.028 came back as 1.0 and ranked level with the page the
    answer was actually quoted from. Since the scorer treats keys as opaque,
    page-qualifying them lets one call cover the whole document and normalise
    once, globally — which also means the IDF weighting sees the real corpus, so
    a term that appears on every page is properly discounted.
    """
    merged = {}
    for page in pages:
        for key, block in page["blocks"].items():
            merged[(page["index"], key)] = block
    scores = _score_ocr_lines(merged, question, answer) if merged else {}

    per_page = {page["index"]: {} for page in pages}
    for (index, key), value in scores.items():
        per_page[index][key] = value
    return per_page


def _rank_pages(pages: list, question: str) -> list:
    """Order pages by how well their own text matches the question.

    This is what keeps the feature inside a 20-call-a-day budget: the ranking is
    free and local, and only the top few page images are ever uploaded.
    """
    per_page = _document_scores(pages, question)
    scored = []
    for page in pages:
        scores = per_page.get(page["index"], {})
        best = max(scores.values(), default=0.0)
        # A page's rank is its best statement plus a little credit for having
        # several relevant ones, so a page that mentions the topic throughout
        # outranks one with a single coincidental match.
        strong = sum(1 for value in scores.values() if value >= 0.5)
        scored.append((float(best) + 0.05 * strong, page))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [page for _score, page in scored]


def _gemini_answer(pages: list, question: str) -> tuple:
    """Ask for an answer over the chosen pages. (parsed, note) — never raises."""
    if not gemini_available():
        return None, (
            "No API key is configured for the analysis service — this was "
            "answered from the document text instead."
        )

    parts = [
        _PROMPT.format(
            count=len(pages),
            question=question,
            not_answerable=not_answerable_rules("pages"),
        )
    ]
    for page in pages:
        buffer = BytesIO()
        page["image"].save(buffer, format="PNG")
        parts.append(f"--- {page['label']} ---")
        parts.append({"mime_type": "image/png", "data": buffer.getvalue()})

    model = gemini_model()
    last_error = None
    for attempt in range(2):
        try:
            response = model.generate_content(parts)
            return parse_json_block(response.text), None
        except Exception as exc:  # noqa: BLE001 - any failure falls back to local
            last_error = exc
            message = str(exc).lower()
            # A per-minute quota clears on its own; a per-day one does not, and a
            # retry there spends the wait on an allowance that is already gone.
            # The daily marker arrives as a quota_id with no space inside "PerDay",
            # so the shared test squashes punctuation out before looking for it.
            transient = "429" in message or "quota" in message
            if not transient or is_exhausted_for_today(exc) or attempt == 1:
                break
            time.sleep(5)
    print(f"[Document] hosted answer unavailable: "
          f"{type(last_error).__name__}: {last_error}")
    return None, describe_service_error(last_error)


def _page_by_label(pages: list, label: str):
    """Match a page label Gemini reported back to a page, tolerantly."""
    wanted = re.sub(r"[^a-z0-9]+", "", str(label or "").lower())
    if not wanted:
        return None
    for page in pages:
        if re.sub(r"[^a-z0-9]+", "", page["label"].lower()) == wanted:
            return page
    # "3", "p3" and "slide3" all have to land on the same page.
    digits = re.findall(r"\d+", wanted)
    if digits:
        number = int(digits[0])
        for page in pages:
            if page["index"] + 1 == number:
                return page
    return None


# ── Attribution over pages ──────────────────────────────────────────────────

def _heat_for_page(page: dict, weighted_probes: list, answer: str) -> dict:
    """Score one page's blocks against every probe that names it.

    A probe is a quote to look for. Each contributes its own relevance map, and
    the page keeps the strongest claim on each block — np.maximum reasoning, so
    two probes hitting the same statement do not sum into a hotter region than
    either earned.
    """
    blocks = page["blocks"]
    if not blocks:
        return {}
    block_scores: dict = {}
    for probe, weight in weighted_probes:
        scores = _score_ocr_lines(blocks, probe, answer)
        for key in _select_relevant(scores, floor=0.45):
            block_scores[key] = max(
                block_scores.get(key, 0.0), scores.get(key, 0.0) * weight
            )
    return block_scores


def _paint(page: dict, block_scores: dict) -> tuple:
    """Render a page's heatmap. Returns (image or None, peak score).

    None rather than a copy of the page when nothing scored: the browser shows
    the original with a note, which is honest about there being no evidence here
    instead of implying a uniformly cold reading.
    """
    if not block_scores:
        return None, 0.0
    # A block's score covers every row it wraps across, so multi-line evidence
    # is highlighted in full rather than only on its first row.
    heat: dict = {}
    for key, score in block_scores.items():
        for row in page["blocks"].get(key, {}).get("keys", [key]):
            heat[row] = max(heat.get(row, 0.0), score)
    if not heat:
        return None, 0.0
    return (
        _build_heatmap(page["image"], page["lines"], heat),
        max(heat.values(), default=0.0),
    )


def _assemble(pages: list, per_page_scores: dict, answer: str) -> tuple:
    """Build the page payload and the evidence list from scored pages."""
    page_payload = []
    evidence: list = []
    for page in pages:
        block_scores = per_page_scores.get(page["index"], {})
        heatmap, peak = _paint(page, block_scores)
        page_payload.append({
            "index": page["index"],
            "label": page["label"],
            "score": round(float(peak), 4),
            "text_source": page["text_source"],
            "original_b64": _image_to_b64(page["image"]),
            "heatmap_b64": _image_to_b64(heatmap) if heatmap else None,
            "line_count": len(page["blocks"]),
        })
        for key in sorted(
            block_scores, key=lambda k: block_scores.get(k, 0.0), reverse=True
        ):
            block = page["blocks"].get(key)
            if block:
                evidence.append(f"{page['label']} — {block['text']}")
    return page_payload, _dedupe_lines(evidence)


def _chart(sources: list) -> list:
    """Donut data. Labels carry the page so the legend is self-explanatory."""
    return [
        {
            "label": f"{source['page_label']}: {source['text']}"
            if source.get("page_label") else source["text"],
            "full_text": source["text"],
            "contribution": source["contribution"],
            "reason": source["reason"],
            "color": "",
        }
        for source in sources
    ]


def _engine_summary(pages: list) -> str:
    """Which reader actually produced the text, for display."""
    labels = {
        "digital": "embedded text",
        "tesseract": "Tesseract OCR",
        "easyocr": "EasyOCR",
        "trocr": "TrOCR handwriting",
    }
    seen = []
    for page in pages:
        for part in str(page["text_source"]).split("+"):
            name = labels.get(part, part)
            if name not in seen:
                seen.append(name)
    return ", ".join(seen) if seen else "none"


# ── Public entry points ─────────────────────────────────────────────────────

# _compose_answer() is shared with the Image tab, so its two "cannot answer"
# replies name an image. On a PDF or a deck that reads as a mistake, and the
# reworded forms are what this module has to recognise.
def _as_document_wording(text: str) -> str:
    return (
        str(text or "")
        .replace("The image text", "The document text")
        .replace("in the image", "in the document")
    )


_NO_ANSWER_REPLIES = {
    _as_document_wording(reply) for reply in _IMAGE_NO_ANSWER_REPLIES
}


def _document_declined(
    answer: str,
    pages: list,
    statements: list,
    total: int,
    engine: str = "local",
    note: str | None = None,
) -> dict:
    """A document result with no attribution, because there is none to give.

    Sources, chart, evidence and every page heat map go out empty. The extracted
    text stays so the reader can check the refusal, and the page images stay so
    they can see what was read — a null heat map on a page shows the original
    with a note rather than a map of nothing.
    """
    page_payload, _evidence = _assemble(pages, {}, "")
    return {
        "answer": answer,
        "engine": engine,
        "engine_note": note,
        "confidence": 0.0,
        "confidence_label": "Document match",
        "sources": [],
        "chart_data": [],
        "highlighted_lines": [],
        "document_text": [s["text"] for s in statements],
        "pages": page_payload,
        "page_count": total or len(pages),
        "analyzed_pages": len(pages),
        "ocr_engine": _engine_summary(pages),
        "unsupported": True,
    }


def explain_document_local(pages: list, question: str, total: int = 0) -> dict:
    """Answer from the document's own text, with per-page heatmaps.

    This path runs whenever the hosted model is unreachable — including when the
    free daily quota is spent — so it stands on its own, and it reports the same
    shape the hosted path does so the browser renders both identically.
    """
    statements = _document_statements(pages)

    # Not one word of the question appears anywhere in the document, so the
    # extractive reader below can only return the least unrelated statement in
    # it, and every page would be painted with heat that means nothing. Decided
    # first, before any of that work.
    document_text = " ".join(s["text"] for s in statements)
    if question_is_off_topic(question, document_text):
        return _document_declined(
            unsupported_reply("document", question, document_text),
            pages,
            statements,
            total,
        )

    # Titles are page furniture: they name a topic instead of asserting anything,
    # so they cannot be the answer. They stay in the corpus and stay quotable —
    # just not eligible to be read out as the reply.
    quotable = [s for s in statements if not s["heading"]] or statements
    answer, confidence = _compose_answer(
        question, [s["text"] for s in quotable]
    )
    # _compose_answer() is shared with the Image tab and says "image" when it
    # finds nothing. On a PDF or a deck that reads as a mistake.
    answer = _as_document_wording(answer)

    # One relevance pass over the whole document, so a statement on page 5 and a
    # statement on page 2 are scored on the same scale. Selecting globally too —
    # rather than per page — is what keeps an irrelevant page cool instead of
    # painting its own best row bright red.
    per_page = _document_scores(pages, question, answer)
    flat = {
        (page["index"], key): value
        for page in pages
        for key, value in per_page.get(page["index"], {}).items()
    }
    selected = _select_relevant(flat, limit=5)

    # Either the reader found no statement worth returning, or nothing in the
    # document scored above zero against the question and the answer. Either way
    # there is nothing to cite, and citing anyway is what filled the donut.
    if answer in _NO_ANSWER_REPLIES or not selected:
        return _document_declined(
            unsupported_reply("document", question, document_text)
            if document_text
            else answer,
            pages,
            statements,
            total,
        )

    per_page_scores = {page["index"]: {} for page in pages}
    ranked: list = []
    by_index = {page["index"]: page for page in pages}
    for index, key in selected:
        score = flat.get((index, key), 0.0)
        per_page_scores[index][key] = score
        ranked.append((score, by_index[index], key))

    # Contributions are each cited statement's share of the total relevance, so
    # the bars and the donut mean the same thing they do on the hosted path.
    ranked.sort(key=lambda row: row[0], reverse=True)
    total_relevance = sum(max(0.0, score) for score, _p, _k in ranked) or 1.0
    sources = []
    for score, page, key in ranked:
        block = page["blocks"].get(key)
        if not block:
            continue
        sources.append({
            "text": block["text"],
            "page": page["index"],
            "page_label": page["label"],
            "contribution": round(max(0.0, score) / total_relevance * 100, 1),
            "reason": (
                f"Relevance {score:.2f} to the question"
                + (" and to the extracted answer" if answer else "")
                + f", on {page['label'].lower()}"
            ),
        })

    page_payload, evidence = _assemble(pages, per_page_scores, answer)
    return {
        "answer": answer,
        "engine": "local",
        "engine_note": None,
        "confidence": round(float(confidence), 4),
        "confidence_label": "Document match",
        "sources": sources,
        "chart_data": _chart(sources),
        "highlighted_lines": evidence,
        "document_text": [s["text"] for s in statements],
        "pages": page_payload,
        "page_count": total or len(pages),
        "analyzed_pages": len(pages),
        "ocr_engine": _engine_summary(pages),
        "unsupported": False,
    }


def explain_document(
    data: bytes, filename: str, question: str, mode: str = "fast", reporter=None
) -> dict:
    """Answer a question about a PDF or .pptx and report what it rests on."""
    query = str(question or "").strip()
    if not query:
        raise ValueError("Question is required.")

    started = time.perf_counter()
    pages, total = read_document_pages(
        data, filename, mode,
        report=reporter.sub("Reading pages") if reporter else None,
    )
    if not pages:
        raise DocumentFormatError("No pages could be read from this file.")
    ingest_seconds = time.perf_counter() - started
    print(
        f"[Document] {len(pages)} page(s) of {total} read in "
        f"{ingest_seconds:.1f}s via {_engine_summary(pages)}"
    )

    if not any(page["blocks"] for page in pages):
        # Nothing readable anywhere. Say so rather than returning an empty
        # answer with an empty heatmap and letting the reader guess why.
        page_payload, _evidence = _assemble(pages, {}, "")
        return {
            "answer": (
                "No readable text was found in this document. If it is a scan, "
                "try the Accurate reader, or upload a sharper copy."
            ),
            "engine": "local",
            "engine_note": None,
            "confidence": 0.0,
            "confidence_label": "Document match",
            "sources": [],
            "chart_data": [],
            "highlighted_lines": [],
            "document_text": [],
            "pages": page_payload,
            "page_count": total,
            "analyzed_pages": len(pages),
            "ocr_engine": _engine_summary(pages),
            # Nothing was read, so nothing can be attributed — the same empty
            # panels the off-topic case suppresses.
            "unsupported": True,
        }

    # Rank locally, then upload only the strongest pages — one request, whatever
    # the document's length.
    if reporter:
        reporter.stage("Ranking pages")
    ordered = _rank_pages(pages, query)
    chosen = ordered[:max(1, _GEMINI_PAGES)]
    if reporter:
        reporter.stage("Answering question")
    parsed, note = _gemini_answer(chosen, query)

    answer = ""
    raw_sources = []
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
            raw_sources.append({
                "text": quote,
                "label": str(source.get("page", "") or "").strip(),
                "contribution": contribution,
                "reason": str(source.get("reason", "") or "").strip(),
            })

    # A declined answer is a finished answer. It has to return before the fallback
    # below, because it legitimately carries no sources and would otherwise be
    # mistaken for a failed call — sending the reader down the local path only to
    # re-derive the same "nothing here matches" conclusion.
    if answer and reply_is_unsupported(parsed or {}, answer):
        if reporter:
            reporter.stage("Building heatmaps")
        return _document_declined(
            answer, pages, _document_statements(pages), total, engine="gemini"
        )

    if not (answer and raw_sources):
        if reporter:
            reporter.stage("Building heatmaps")
        result = explain_document_local(pages, query, total)
        if note:
            result["engine_note"] = note
            result["error"] = note
        return result

    # Heat is graded against the largest contribution, not against 1.0: five
    # equally-cited sources at 0.2 each would otherwise never rise past the blue
    # end of the ramp and every page would read as uniformly weak.
    strongest = max(
        (source["contribution"] for source in raw_sources), default=0.0
    ) or 1.0

    probes_by_page: dict = {}
    sources = []
    for source in raw_sources:
        page = _page_by_label(pages, source["label"])
        if page is None:
            # An unlabelled or mislabelled quote still has to be attributable,
            # so it is matched against whichever page contains it best.
            page = max(
                pages,
                key=lambda p: max(
                    _score_ocr_lines(p["blocks"], source["text"]).values(),
                    default=0.0,
                ) if p["blocks"] else 0.0,
            )
        weight = 0.45 + 0.55 * min(1.0, source["contribution"] / strongest)
        probes_by_page.setdefault(page["index"], []).append(
            (source["text"], weight)
        )
        sources.append({
            "text": source["text"],
            "page": page["index"],
            "page_label": page["label"],
            "contribution": round(source["contribution"] * 100, 1),
            "reason": source["reason"],
        })

    per_page_scores = {}
    if reporter:
        reporter.stage("Building heatmaps")
    for page in pages:
        probes = probes_by_page.get(page["index"], [])
        scores = _heat_for_page(page, probes, answer) if probes else {}
        if not scores and probes:
            # The quote was paraphrased and matched nothing verbatim. Falling
            # back to question relevance keeps the page from looking unread.
            fallback = _score_ocr_lines(page["blocks"], query, answer)
            scores = {
                key: fallback.get(key, 0.0)
                for key in _select_relevant(fallback)
            }
        per_page_scores[page["index"]] = scores

    page_payload, evidence = _assemble(pages, per_page_scores, answer)
    statements = _document_statements(pages)

    # Grounding stands in for a confidence score the model does not report: the
    # share of each quote that is really in the document, weighted by the
    # contribution claimed for it. A paraphrased "quote" drags it down.
    document_text = " ".join(s["text"] for s in statements)
    weights = [source["contribution"] for source in sources]
    weight_total = sum(weights) or 1.0
    grounding = sum(
        _containment(document_text, source["text"]) * weight
        for source, weight in zip(sources, weights)
    ) / weight_total

    return {
        "answer": answer,
        "engine": "gemini",
        "engine_note": None,
        "confidence": round(min(1.0, grounding), 4),
        "confidence_label": "Source grounding",
        "sources": sources,
        "chart_data": _chart(sources),
        "highlighted_lines": evidence,
        "document_text": [s["text"] for s in statements],
        "pages": page_payload,
        "page_count": total,
        "analyzed_pages": len(pages),
        "pages_sent": [page["label"] for page in chosen],
        "ocr_engine": _engine_summary(pages),
        "unsupported": False,
    }
