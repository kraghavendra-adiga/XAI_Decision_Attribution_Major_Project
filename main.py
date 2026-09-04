# main.py
import os
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

try:
    from .schemas import SummarizeRequest, QARequest
    from .model_utils import summarize_text
    from .image_explainer import (
        IMAGE_STAGES,
        explain_image_gemini,
        explain_image_vqa,
    )
    from .audio_explainer import (
        audio_stage_plan,
        explain_audio,
        is_transcribed,
    )
    from .document_explainer import (
        DOCUMENT_STAGES,
        DocumentFormatError,
        explain_document,
    )
    from .attribution import (
        get_lime_explanation,
        get_shap_explanation,
        get_sentence_attribution,
    )
    from . import explainer  # backwards compat
    from . import progress
    from .progress import Reporter
    from .text_cleaning import clean_educational_text
    from .text_explainer import explain_text_qa, heat_from_sentence_scores
    from .utils import ServiceOverloaded
except ImportError:  # pragma: no cover - fallback for direct script execution
    from schemas import SummarizeRequest, QARequest
    from model_utils import summarize_text
    from image_explainer import (
        IMAGE_STAGES,
        explain_image_gemini,
        explain_image_vqa,
    )
    from audio_explainer import (
        audio_stage_plan,
        explain_audio,
        is_transcribed,
    )
    from document_explainer import (
        DOCUMENT_STAGES,
        DocumentFormatError,
        explain_document,
    )
    from attribution import (
        get_lime_explanation,
        get_shap_explanation,
        get_sentence_attribution,
    )
    import explainer  # backwards compat
    import progress
    from progress import Reporter
    from text_cleaning import clean_educational_text
    from text_explainer import explain_text_qa, heat_from_sentence_scores
    from utils import ServiceOverloaded

app = FastAPI(title="Decision Attribution In LLM's")

# Cross-origin access, for the split deploy: a static host serves index.html and
# the browser calls this container directly. Uploads have to go direct rather
# than through the static host's proxy, which caps request bodies well below the
# size of a video. ALLOWED_ORIGINS is a comma-separated list; unset means any
# origin, which is right for a public demo and wrong for anything with a bill
# attached, since an open endpoint spends the same daily model allowance.
_allowed_origins = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_BASE_DIR = Path(__file__).resolve().parent
static_dir = _BASE_DIR / "static"
templates_dir = _BASE_DIR / "templates"

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

# Stage weights, as shares of one request. Measured, not guessed: on 800–4000
# character passages the summarizer decode took 65–77% of a /api/summarize call
# with LIME at 10–11% and SHAP at 13–24%, while a hosted answer call took 91–95%
# of a /api/qa call. Sentence ranking never reached 0.2% of either.
#
# Only the proportions between stages come from this table. Position *inside* a
# stage is always reported by the work itself — LIME's scored perturbation
# batches, SHAP's evaluations — and a stage that cannot count its own internals
# says so rather than inventing a number.
_SUMMARIZE_STAGES = [
    ("Cleaning text", 1),
    ("Summarizing", 60),
    ("Weighing words (LIME)", 12),
    ("Weighing words (SHAP)", 25),
    ("Ranking sentences", 2),
]
_QA_STAGES = [
    ("Answering question", 85),
    ("Weighing words (LIME)", 5),
    ("Weighing words (SHAP)", 8),
    ("Ranking sentences", 2),
]


@app.middleware("http")
async def no_cache(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/progress/{job_id}")
async def read_progress(job_id: str):
    """Where a running job has actually got to, for the browser's progress bar.

    404 covers both "never started" and "long finished and reaped"; the client
    treats either as nothing left to draw.
    """
    job = progress.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such job.")
    return job


def _summarize_sync(req: SummarizeRequest) -> dict:
    """The blocking half of /api/summarize.

    Kept out of the coroutine on purpose: the analysis holds a worker for tens of
    seconds, and running it on the event loop would block the progress polls that
    are meant to be reporting on it.
    """
    rep = Reporter(req.job_id or "", _SUMMARIZE_STAGES)
    rep.stage("Cleaning text")
    clean_text = clean_educational_text(req.text)
    rep.stage("Summarizing")
    # Long text is summarized chunk by chunk, so the bar can move inside the stage
    # instead of sitting still through the slowest step in the run.
    summary = summarize_text(clean_text, report=rep.sub("Summarizing"))
    # The summary is the reference the three panels explain: they report which
    # parts of the input this particular summary rests on.
    lime_exp = get_lime_explanation(
        clean_text, summary, report=rep.sub("Weighing words (LIME)")
    )
    shap_exp = get_shap_explanation(
        clean_text, summary, report=rep.sub("Weighing words (SHAP)")
    )
    rep.stage("Ranking sentences")
    sentence_attr = get_sentence_attribution(clean_text, summary)
    rep.done()
    return {
        "summary": summary,
        "cleaned_text": clean_text,
        "lime": lime_exp,
        "shap": shap_exp,
        "sentence_attribution": sentence_attr,
        # The same attribution, laid back over the passage in reading order,
        # so the reader sees which parts of *their* text the summary used
        # instead of a ranked list detached from where the words sat.
        "heatmap": heat_from_sentence_scores(clean_text, sentence_attr),
    }


@app.post("/api/summarize")
async def summarize(req: SummarizeRequest):
    progress.start(req.job_id or "", "Cleaning text")
    try:
        return await run_in_threadpool(_summarize_sync, req)
    except HTTPException:
        raise
    except ServiceOverloaded as exc:
        progress.fail(req.job_id or "", str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        progress.fail(req.job_id or "", str(exc))
        raise HTTPException(
            status_code=500, detail=f"{type(exc).__name__}: {exc}"
        ) from exc


def _qa_sync(req: QARequest) -> dict:
    """The blocking half of /api/qa — see _summarize_sync on why it is split out."""
    rep = Reporter(req.job_id or "", _QA_STAGES)
    rep.stage("Answering question")
    # Gemini answers in whole statements and cites the sentences it used;
    # the local retrieval + extractive path takes over when the free tier is
    # spent. Either way `result` carries the sources the heat map paints.
    result = explain_text_qa(req.question, req.context)
    clean_context = result["cleaned_text"]
    answer = result["answer"]

    # Nothing in the passage addresses the question, so there is nothing to
    # attribute: perturbing words that were never used cannot show what the
    # answer rested on, and LIME plus SHAP are the two most expensive stages of
    # the request. The answer says so in words instead.
    if result.get("unsupported"):
        rep.done()
        return {
            "answer": answer,
            "short_answer": "",
            "engine": result["engine"],
            "engine_note": result["engine_note"],
            "confidence": 0.0,
            "confidence_label": result["confidence_label"],
            "cleaned_text": clean_context,
            "sources": [],
            "unsupported": True,
            "heatmap": {"text": clean_context, "spans": [], "sentences": []},
            "lime": [],
            "shap": [],
            "sentence_attribution": [],
        }

    # An extracted answer can be two or three words, which is too sparse to
    # attribute against on its own, so the question is folded into the
    # reference: the panels then show what the input contributed to *this*
    # question being answered *this* way.
    reference = f"{req.question} {answer}".strip()
    lime_exp = get_lime_explanation(
        clean_context, reference, report=rep.sub("Weighing words (LIME)")
    )
    shap_exp = get_shap_explanation(
        clean_context, reference, report=rep.sub("Weighing words (SHAP)")
    )
    rep.stage("Ranking sentences")
    sentence_attr = get_sentence_attribution(clean_context, reference)
    rep.done()
    return {
        "answer": answer,
        "short_answer": result["short_answer"],
        "engine": result["engine"],
        "engine_note": result["engine_note"],
        "confidence": result["confidence"],
        "confidence_label": result["confidence_label"],
        "cleaned_text": clean_context,
        "sources": result["sources"],
        "unsupported": False,
        "heatmap": result["heatmap"],
        "lime": lime_exp,
        "shap": shap_exp,
        "sentence_attribution": sentence_attr,
    }


@app.post("/api/qa")
async def qa(req: QARequest):
    progress.start(req.job_id or "", "Answering question")
    try:
        return await run_in_threadpool(_qa_sync, req)
    except HTTPException:
        raise
    except ServiceOverloaded as exc:
        progress.fail(req.job_id or "", str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        progress.fail(req.job_id or "", str(exc))
        raise HTTPException(
            status_code=500, detail=f"{type(exc).__name__}: {exc}"
        ) from exc


@app.post("/explain-image-vqa")
async def explain_image_vqa_endpoint(
    file: UploadFile = File(...),
    question: str = Form(...),
    job_id: str = Form(""),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    progress.start(job_id, "Reading image")
    try:
        image_bytes = await file.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        reporter = Reporter(job_id, IMAGE_STAGES)
        result = await run_in_threadpool(
            explain_image_gemini, pil_image, question, reporter
        )
        reporter.done()
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except ServiceOverloaded as exc:
        progress.fail(job_id, str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        progress.fail(job_id, str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/explain-document")
async def explain_document_endpoint(
    file: UploadFile = File(...),
    question: str = Form(...),
    mode: str = Form("fast"),
    job_id: str = Form(""),
):
    """Answer a question about a PDF or .pptx, page by page.

    Gated on the extension rather than the content type: browsers report .pptx
    inconsistently (application/octet-stream is common), and a filename is what
    the reader chose anyway.
    """
    name = (file.filename or "").lower()
    if not name.endswith((".pdf", ".pptx", ".ppt")):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be a PDF or a .pptx presentation.",
        )

    progress.start(job_id, "Reading pages")
    try:
        document_bytes = await file.read()
        reporter = Reporter(job_id, DOCUMENT_STAGES)
        result = await run_in_threadpool(
            explain_document, document_bytes, file.filename, question, mode, reporter
        )
        reporter.done()
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except DocumentFormatError as exc:
        # Already phrased for a reader — a legacy .ppt, a locked PDF, a missing
        # package — so it goes out as-is rather than as a 500.
        progress.fail(job_id, str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ServiceOverloaded as exc:
        progress.fail(job_id, str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        progress.fail(job_id, str(exc))
        raise HTTPException(
            status_code=500, detail=f"{type(exc).__name__}: {exc}"
        ) from exc


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/explain-audio")
async def explain_audio_endpoint(
    file: UploadFile = File(...),
    question: str = Form(...),
    job_id: str = Form(""),
):
    progress.start(job_id, "Reading file")
    try:
        audio_bytes = await file.read()
        # Whether this recording has been transcribed before decides the shape of
        # the bar, not just its speed: a second question about the same file skips
        # four fifths of the work, and a plan that still reserved room for it
        # would leave the bar parked at 2% and then jump.
        reporter = Reporter(
            job_id,
            audio_stage_plan(file.filename, transcribed=is_transcribed(audio_bytes)),
        )
        result = await run_in_threadpool(
            explain_audio, audio_bytes, file.filename, question, reporter
        )
        reporter.done()
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except ServiceOverloaded as exc:
        progress.fail(job_id, str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        progress.fail(job_id, "Audio decoding failed: ffmpeg was not found.")
        raise HTTPException(
            status_code=500,
            detail=(
                "Audio decoding failed: ffmpeg was not found on PATH. "
                "Whisper needs the ffmpeg binary to read audio files. "
                f"({exc})"
            ),
        ) from exc
    except Exception as exc:
        progress.fail(job_id, str(exc))
        raise HTTPException(
            status_code=500, detail=f"{type(exc).__name__}: {exc}"
        ) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
