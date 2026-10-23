---
title: XAi Decision Attribution
emoji: 🔍
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: agpl-3.0
short_description: Answers that show which sentences, regions and seconds they came from
---

# XAi — Decision Attribution

**Every AI tool hands you an answer and expects you to trust it. This one hands you the
answer *and shows you where it came from*.**

A FastAPI web app that summarises and answers questions about text, images, PDFs, slide
decks, audio and video — and beside every output, shows the exact sentences, words,
image regions or spoken lines it was built from, scored strongest-first.

A sentence scoring 76% is one the answer leaned on. A sentence scoring 2% was ignored.
So you verify an answer by glancing at three lines instead of re-reading the page — and
if the top-scoring sentence isn't the one you'd have picked, that's your warning.

---

## Table of contents

- [Why this exists](#why-this-exists)
- [What it does](#what-it-does)
- [How the attribution works](#how-the-attribution-works)
- [Setup](#setup)
- [Running it](#running-it)
- [Deploying](#deploying)
- [API reference](#api-reference)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Design decisions worth knowing](#design-decisions-worth-knowing)
- [Gotchas](#gotchas)
- [Contact](#contact)

---

## Why this exists

Ask any AI to summarise your notes and you get a clean paragraph with no way to tell
where it came from. If it quietly skipped the one line that mattered, or made something
up, nothing on the screen tells you. The only way to be sure is to read the whole passage
again — which is the exact work you used the tool to avoid.

**Where it's actually useful**

- **Studying** — know whether a summary is safe to revise from before you rely on it.
- **Teaching** — see which part of a handout the model latched onto, and which it walked past.
- **Research** — trace a claim back to the paragraph it came from instead of hunting for it.
- **Anything with a consequence** — medicine, law, finance, where "the model said so" is not a reason.

**Why it isn't just an API wrapper.** Producing the summary is the easy part — one API
call does that. The project is everything around it: cleaning pasted text so citation
markers, view counts and headings glued onto sentences don't hijack the result; scoring
every sentence against the output that was *actually produced*; and running two
independent explanation methods so that when they disagree, you can see the disagreement
rather than being handed one confident number.

**What it does not claim.** It shows what an output rests on, not whether that output is
true. An answer built on the wrong sentence is still wrong — the difference is that now
you can see the wrong sentence. Scores are also not rescaled so the best one always reads
100%: a weak match is shown as a weak match.

---

## What it does

Five tabs, one question in every format.

| Tab | Input | What you get back |
|---|---|---|
| **✦ Summarize** | Pasted text | Abstractive summary + LIME weights + SHAP tokens + per-sentence attribution + a source heatmap laid over your original passage. Ask a question about the same passage in the row below and it swaps to an extractive answer with quoted sources. |
| **⬡ Image** | PNG / JPG | Answer, the OCR'd lines it used, a graded heatmap over the picture, per-source contribution chart. |
| **▤ Document** | PDF / .pptx | Answer, page-by-page heatmaps, quoted evidence tagged with its page ("Slide 3"), contribution chart. Handwriting supported via two offline OCR engines. |
| **♪ Audio** | mp3 / wav / m4a … | Whisper transcript, then the answer with each part backed by the spoken line it came from — quoted, with a timestamp. |
| **⏵ Video** | mp4 / mov / mkv … | Same as audio, plus an extracted keyframe at each cited timestamp. |

Long requests report real progress. The browser generates a `job_id`, polls
`GET /api/progress/{job_id}`, and the bar advances on measured stage weights — LIME's
scored perturbation batches and SHAP's evaluation count report their own position, and a
stage that can't count its internals says so rather than inventing a number.

---

## How the attribution works

Three explainers, all answering the same question: *which parts of the input does the
output you're looking at actually rest on?*

- **LIME** (`lime.lime_text`) — perturbs the input and fits a local linear model over which words mattered.
- **SHAP** (partition explainer, 500 evals) — token-level contributions from a different theoretical basis, so agreement between the two is meaningful and disagreement is visible.
- **Sentence attribution** — relevance scores from the selected text evidence model, ranked.

The explained quantity is **the alignment between a candidate input and the output that
was actually produced**, measured by the selected evidence scorer.

> **An honest caveat, stated because it matters.** That alignment score is a *surrogate*.
> Perturbing the real summariser would need one distilbart decode per LIME sample —
> minutes of CPU per request. The earlier version instead explained a sentiment
> classifier (distilbert SST-2), which has nothing to do with summarisation: removing
> words barely moved its POSITIVE probability, every weight came back around ±0.005, and
> the ranking was noise. The alignment score responds to exactly the words the summary was
> built from, which the classifier never did — but it is a stand-in for perturbing the
> generator, not the real thing.

### Models

| Role | Model | Runs |
|---|---|---|
| Summarisation | `sshleifer/distilbart-cnn-6-6` | locally, CPU |
| Extractive QA fallback | `distilbert-base-cased-distilled-squad` | locally, CPU |
| Transcription | `openai-whisper` (`base`) | locally, CPU |
| Print OCR | Tesseract via `pytesseract` | locally |
| Handwriting OCR | EasyOCR (Fast) / `microsoft/trocr-small-handwritten` (Accurate) | locally, optional |
| Grounded answers + vision | `gemini-3.6-flash` via `google-genai` | hosted |

**Fallbacks depend on the feature.** If the key is missing, the daily allowance is
spent, or the service refuses, text / image / document paths can fall back to local
engines and label themselves `engine: "local"` with `confidence_label: "Passage match"`
instead of `"Grounding"`. Audio and video use Gemini to judge questions against the
transcript by meaning and context, not exact keyword overlap; without the service, they
report an error rather than claim an unsupported answer.

---

## Setup

Requires **Python 3.11** (3.11.9 is what this was built and tested on).

### 1. Virtual environment and dependencies

```bash
python -m venv .venv311
```

```bash
./.venv311/Scripts/python.exe -m pip install -r Decision_Attribution/requirements.txt
```

**The handwriting extras must be installed in this order, with `--no-deps`.** A plain
`pip install easyocr` resolves its own torch and will replace the working `torch 2.11.0+cpu`
that Whisper depends on:

```bash
./.venv311/Scripts/python.exe -m pip install --no-deps "torchvision==0.26.*" easyocr && ./.venv311/Scripts/python.exe -m pip install shapely pyclipper "python-bidi<0.6"
```

Both handwriting engines are genuinely optional — if neither imports, pages still analyse
on Tesseract and the response says so.

### 2. Tesseract

Not a pip package. Install the binary and either put it on `PATH` or leave it at the
default Windows location (`C:\Program Files\Tesseract-OCR\tesseract.exe`), which
`image_explainer.py` probes for automatically.

### 3. ffmpeg

Whisper needs it. `imageio-ffmpeg` ships a static binary and the app wires it up on
startup — you should see `[explainer] Using bundled ffmpeg from …` in the log, and no
system-wide install is required.

### 4. API key

Create `xai/.env`:

```dotenv
GEMINI_API_KEY=your-key-here
```

Get one from [Google AI Studio](https://aistudio.google.com/apikey). The app runs without
it — every path falls back to its local engine — but grounded answers, image analysis and
document answering need it.

> The free tier allows **20 `generate_content` requests per day** on `gemini-3.6-flash`,
> resetting on a 24-hour cycle. A video run costs 5 calls (1 answer + 4 keyframe
> descriptions); an image run costs 1.

---

## Running it

```bash
venv/bin/python -m uvicorn main:app --reload --host 127.0.0.1 --port 8020
```

Then open <http://localhost:8020>.

Run that command from the `xai/` project directory. `utils.py` loads the key from its
adjacent `.env` file at import, so a running process keeps the key it started with — see
[Gotchas](#gotchas).

### Evidence-model fine-tuning experiment

The LIME, SHAP, sentence ranking and source heat map in the text summarize/Q&A workflows
can switch between the original DistilBERT SST-2 checkpoint and the fine-tuned DistilBERT
evidence scorer. Fine-tuning changes attribution only; the generated summary and answer
are unchanged. Image, document, audio and video keep their existing behavior. Pretrained
SST-2 is selected by default.

Q&A answers are generated by the hosted analysis service when available; if that
service is unavailable or its quota is exhausted, the app uses local extractive
DistilBERT Q&A over overlapping passage windows. The SST-2 selector changes
explanations only, not answer generation.

The training run adapts the SST-2 checkpoint to score whether candidate source text is
annotated evidence for a scientific claim. It uses the public
[SciFact dataset](https://github.com/allenai/scifact) and compares pretrained SST-2 with
fine-tuned SST-2 on the labeled development split. The `/finetuning` dashboard shows the
measured results; it does not assume fine-tuning wins.

From the folder containing `xai/`:

```bash
cd xai
venv/bin/python finetune_evidence.py
```

The first run downloads the SciFact archive and the pretrained checkpoint. Training
weights and `results.json` are saved under the ignored `finetuning_artifacts/` directory.
Restart or refresh the app after training. The fine-tuned model then appears in the text
explanation model selector; open <http://localhost:8020/finetuning> for the comparison
dashboard.
If the app is running on a different port, use that port instead. The benchmark is
scientific-domain only; a win on SciFact does not establish improved explanations on
every subject. The text tab labels which scorer produced the attribution so you can run
the same text through multiple choices and compare their explanations.

---

## Deploying

Live as two pieces: **Vercel** serves the page, a **Docker container** serves the models.
Full walkthrough in [deploy/DEPLOY.md](deploy/DEPLOY.md).

```
 your-app.vercel.app  ──►  <space>.hf.space
 static index.html         torch · distilbart · Whisper · Tesseract · LIME/SHAP
```

The analysis side cannot run on Vercel, and not by a small margin — its Python functions
cap at **250 MB unzipped** and `torch` alone is ~800 MB; Tesseract and ffmpeg are system
binaries with no `apt-get` available; the model weights are ~950 MB against a 512 MB
ephemeral `/tmp`; and a document run measured ~2 minutes against a 60s (Hobby) / 300s (Pro)
ceiling. The blocker no paid tier fixes is statefulness: [progress.py](progress.py) keeps
its job registry and [audio_explainer.py](audio_explainer.py) its transcript cache in
module globals, so on stateless serverless a `/api/progress/{job_id}` poll lands on a
different instance than the work and 404s forever — the progress bar would never move.

So the container keeps everything, and it is a complete app on its own: the Space URL
serves the UI too. Hugging Face Spaces' free tier (2 vCPU / **16 GB RAM**) is the only
free option big enough to hold torch, distilbart and Whisper at once — Render's free 512 MB
cannot.

The split costs one indirection. `templates/index.html` reads

```js
const API_BASE = (window.XAI_API_BASE || "").replace(/\/+$/, "");
```

which is `""` unless something injects that global, so a local run is byte-for-byte
unchanged. [deploy/vercel-build.sh](deploy/vercel-build.sh) injects it at build time and
copies the template into `public/` — one source of truth, no second HTML file to drift.
The browser posts **directly** to the container rather than through Vercel, because a
proxied multipart upload would meet the host's ~4.5 MB body cap and a video is exactly the
input that exceeds it.

Relevant files: [`Dockerfile`](Dockerfile), [`vercel.json`](vercel.json),
[`.dockerignore`](.dockerignore), [`.vercelignore`](.vercelignore).

Two environment variables beyond `GEMINI_API_KEY`:

| Where | Variable | Value |
|---|---|---|
| Vercel | `XAI_API_BASE` | `https://<your-space>.hf.space` — baked in at build, so changing it needs a redeploy |
| Container | `ALLOWED_ORIGINS` | `https://<your-app>.vercel.app`. Unset means any origin, which lets any site spend your daily model allowance |

---

## API reference

All responses are JSON. `job_id` is optional everywhere; a request without one simply
reports no progress.

### `GET /`
The single-page UI.

### `GET /health`
`{"status": "ok"}`

### `GET /api/progress/{job_id}`
Where a running job has got to. `404` covers both "never started" and "long finished and
reaped" — the client treats either as nothing left to draw.

```json
{ "done": false, "error": null, "pct": 64, "precise": false, "stage": "Answering question" }
```

### `POST /api/summarize`
```json
{ "text": "…", "job_id": "optional" }
```
Returns `summary`, `cleaned_text`, `lime`, `shap`, `sentence_attribution`, `heatmap`.

### `POST /api/qa`
```json
{ "question": "…", "context": "…", "job_id": "optional" }
```
Returns `answer`, `short_answer`, `engine` (`"gemini"` \| `"local"`), `engine_note`,
`confidence`, `confidence_label`, `cleaned_text`, `sources[]`, `unsupported`, `heatmap`.

### `POST /explain-image-vqa`
`multipart/form-data` — `file` (must be `image/*`), `question`, `job_id`.

Returns `answer`, `sources[]` (each with `text`, `contribution`, `reason`,
`matched_line_keys`), `chart_data[]`, `highlighted_lines[]`, `ocr_text[]`, `unsupported`.

### `POST /explain-document`
`multipart/form-data` — `file` (`.pdf` \| `.pptx` \| `.ppt`), `question`,
`mode` (`fast` \| `accurate`), `job_id`.

Gated on the file extension rather than content type, because browsers report `.pptx`
inconsistently (`application/octet-stream` is common) and a filename is what the reader
chose anyway. A legacy `.ppt`, a locked PDF or a missing package returns **400** with a
message already phrased for a reader, not a 500.

### `POST /explain-audio`
`multipart/form-data` — `file`, `question`, `job_id`. **Handles video too** — the
extension decides whether keyframes are extracted.

Returns `answer`, `transcript`, `segments[]`, `timestamp_evidence[]` (each with `text`,
`matched_transcript`, `start`, `end`, `timestamp_label`, `contribution`, `reason`),
`chart_data[]`, `unsupported`.

### Status codes

| Code | Meaning |
|---|---|
| `400` | Wrong file type, or a document format problem already phrased for the reader |
| `503` | `ServiceOverloaded` — hosted model refused; carries a display-ready message |
| `500` | Anything else, as `TypeName: message` |

---

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `GEMINI_API_KEY` | — | Enables every hosted path. Absent ⇒ local engines only. |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS allowlist. Only needed for the split deploy; `*` is right for a local run and wrong for a public one. |
| `DOC_MAX_PAGES` | `25` | Page cap per document. Each page costs a render, sometimes an OCR pass, and ships two PNGs to the browser. |
| `DOC_GEMINI_PAGES` | `3` | How many page images are sent for the hosted answer. Pages are ranked locally first. |
| `DOC_PDF_DPI` | `150` | Render DPI. 150 is where Tesseract stops losing 9pt body text; higher mostly costs render time. |
| `DOC_DISPLAY_WIDTH` | `1000` | Max width of page images returned to the browser. |
| `DOC_TROCR_MODEL` | `microsoft/trocr-small-handwritten` | Accurate-mode handwriting model. |
| `DOC_TROCR_BATCH` | `8` | Line crops pushed through TrOCR together. |

---

## Architecture

The live application is these fifteen modules. Everything else ending in `.py` in this
folder is dead codegen — see [Gotchas](#gotchas).

```
Decision_Attribution/
├── main.py                  FastAPI app, 8 routes, stage-weight tables
├── __init__.py
├── schemas.py               SummarizeRequest / QARequest
├── progress.py              Job registry + Reporter for the real progress bar
│
├── model_utils.py           distilbart summariser + distilbert extractive QA (lazy-loaded)
├── text_cleaning.py         Strips citation markers, view counts, glued-on headings
├── attribution.py           LIME + SHAP + model-based sentence attribution
├── utils.py                 Gemini adapter, error classification, off-topic detection
│
├── text_explainer.py        /api/qa — grounded answer, local fallback, source heatmap
├── image_explainer.py       /explain-image-vqa — OCR, line scoring, heatmap painter
├── document_explainer.py    /explain-document — PDF/pptx → boxed pages → image pipeline
├── audio_explainer.py       /explain-audio — Whisper, timestamp alignment, keyframes
├── explainer.py             Backwards-compat shim (re-exports the four above)
│
├── templates/index.html     The entire front end — one file, no build step
├── static/logo.png
├── requirements.txt
│
├── Dockerfile               Backend image — see Deploying
├── vercel.json              Static front end config
└── deploy/
    ├── vercel-build.sh      templates/index.html → public/, injects XAI_API_BASE
    └── DEPLOY.md            Step-by-step walkthrough
```

**The document tab reuses the image tab wholesale.** None of the image machinery is
actually specific to photographs — it needs text that carries a bounding box and nothing
more. So `document_explainer.py` converts a PDF or `.pptx` into a list of *pages*, each
carrying boxed lines in exactly the shape `_group_ocr_by_line()` produces, then hands
those to the same scorer and the same heatmap painter. What a document needs on top:
page identity (so a quote can say "Slide 3"), quota discipline, and handwriting support.

---

## Design decisions worth knowing

**Off-topic questions get a refusal, not invented evidence.** Ask "what is a dog" about a
mitochondria lecture and every path returns `unsupported: true` with a plain statement
that the source doesn't contain it — no sources, no chart, no heatmap. Attribution
machinery exists to point at evidence; when there is none, running it anyway is how a
question about dogs ended up lighting up whichever lines happened to score least badly,
with a 50/50 donut and two invented "reasons".

**Transcripts are cached in-process.** A second question about the same recording skips
Whisper entirely — measured 19.8s → 0.0s. The progress bar knows this and reshapes itself:
the transcription stage is dropped from the plan rather than left as a dead span, because
weights are relative and a bar that reserved room for skipped work would park at 2% and
then jump.

**Long documents don't become 25 API calls.** Pages are ranked locally first and only the
best three are sent, in a single request.

**Progress weights are measured, not guessed.** On 800–4000 character passages the
summariser decode took 65–77% of a `/api/summarize` call, with LIME at 10–11% and SHAP at
13–24%; a hosted answer took 91–95% of a `/api/qa` call. Sentence ranking never reached
0.2% of either.

**Provider errors are translated, once.** Raw payloads are unreadable
(`429 You exceeded your current quota … [violations { quota_metric: … }]`) so
`describe_service_error()` maps each class to one sentence saying what happened and what
to do. The daily allowance gets its own test — `is_exhausted_for_today()` — because it's
the one refusal a retry cannot ride out. The marker arrives as a `quotaId`
(`GenerateRequestsPerDayPerProjectPerModel-FreeTier`) with no space inside "PerDay", so
punctuation is squashed out before matching; looking for `"per day"` silently misses every
real payload.

**The UI never names the provider.** Every user-facing string says "the hosted analysis
service". `confidence_label` distinguishes the engines neutrally — `"Grounding"` for
hosted, `"Passage match"` for local.

**Blocking work runs in a threadpool.** The analysis holds a worker for tens of seconds;
on the event loop it would block the very progress polls meant to be reporting on it.

---

## Gotchas

**A rotated API key needs a server restart.** `utils.py` calls `load_dotenv()` at import,
so a running uvicorn process keeps the key it started with. Editing `.env` underneath it
changes nothing and the 429s continue — which reads exactly like "the new key is also
exhausted". Compare the `.env` mtime against the server's start time, and test the key in
a *fresh* process, before concluding anything about the key itself.

**Legacy one-off scripts — do not run these.** `build_index.py`, `build_index_safe.py`,
`add_styles.py`, `update_styles.py`, `fix_*.py`, `apply_*.py`, `restore_js.py` and
`original_index.html` are historical codegen that rewrote `templates/index.html` and the
old monolithic `explainer.py`. They are stale:

- `build_index.py` contains a non-UTF-8 byte at line 583 and does not even compile.
- `build_index_safe.py` and `original_index.html` still describe a *separate* Ask tab and the old "Explainable Insights" tagline — running either would undo the Ask-into-Summarize merge.
- `apply_json_fixes.py` and friends patch `explainer.py`, which is now only a compatibility shim.

They're kept for history. Edit `templates/index.html` directly.

**First run downloads models.** distilbart, distilbert, Whisper `base` and (in Accurate
mode) TrOCR weights (~240 MB) fetch on first use and then cache.

**CPU only.** Nothing here assumes or requires a GPU; EasyOCR is constructed with
`gpu=False`.

---

## Contact

**Prakhyath Jain**

- Phone — [7349526753](tel:7349526753)
- Email — [prakhyathjain20@gmail.com](mailto:prakhyathjain20@gmail.com)
- LinkedIn — [linkedin.com/in/prakhyath-jain-a15031306](https://www.linkedin.com/in/prakhyath-jain-a15031306/)
