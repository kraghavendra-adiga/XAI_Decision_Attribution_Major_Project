# Deploying

The app goes live as **two pieces**: Vercel serves the page, a container serves
the models. You get a `vercel.app` URL, and everything the app does locally keeps
working — including the progress bar, the transcript cache, Whisper timestamps
and Tesseract's word boxes.

```
  visitor
     │
     ▼
 your-app.vercel.app          static index.html + logo, edge-cached, free
     │
     │  fetch(`${API_BASE}/api/summarize`)  ← direct, not proxied
     ▼
 <name>.hf.space              Docker, free CPU (2 vCPU / 16 GB RAM)
  ├ distilbart-cnn-6-6        summaries
  ├ distilbert-squad          extractive fallback
  ├ Whisper base              transcripts + timestamps
  ├ Tesseract / EasyOCR       OCR + the boxes heatmaps need
  ├ LIME / SHAP / TF-IDF      attribution
  └ gemini-3.6-flash          grounded answers (hosted)
```

## Why not Vercel alone

Vercel cannot host the analysis side, and the gap is not marginal:

| | Vercel allows | This app needs |
|---|---|---|
| Function bundle | 250 MB unzipped | `torch` alone ≈ 800 MB |
| System binaries | none (no `apt-get`) | Tesseract, ffmpeg |
| Writable disk | 512 MB ephemeral `/tmp` | ≈ 950 MB of model weights |
| `maxDuration` | 60s Hobby / 300s Pro | document run measured ≈ 2 min |
| Process memory | per-invocation, not shared | [progress.py:23](../progress.py#L23) `_JOBS` and [audio_explainer.py:156](../audio_explainer.py#L156) `_TRANSCRIPT_CACHE` are module globals |

That last row is the one no amount of paid tier fixes. The browser polls
`GET /api/progress/{job_id}` while the work runs; on stateless serverless that
poll lands on a different instance than the one doing the work, finds no such
job, and 404s forever. The progress bar would never move.

Hugging Face Spaces' free tier is the only free option with enough RAM to hold
torch, distilbart and Whisper at once. Render's free tier is 512 MB and cannot.

---

## 1. Backend — Hugging Face Space

**Create the Space.** At <https://huggingface.co/new-space>: any name, **SDK →
Docker → Blank**, hardware **CPU basic (free)**. Public or private both work.

**Add the API key.** In the Space, *Settings → Variables and secrets → New
secret*:

| | |
|---|---|
| Name | `GEMINI_API_KEY` |
| Value | your key |

A secret, not a variable — variables are visible to anyone who can see the Space.
The app still builds and runs without it; every path falls back to its local
engine, so grounded answers and image analysis go away but nothing crashes.

**Push this repo to it.** From the repo root:

```bash
git remote add space https://huggingface.co/spaces/<your-username>/<your-space-name>
```

```bash
git push space main
```

Use a [write token](https://huggingface.co/settings/tokens) as the password if
prompted. `Dockerfile` and the `sdk: docker` / `app_port: 7860` frontmatter in
`README.md` are already in place — HF reads both from the repo root, which is why
the README carries that block.

**Watch the build.** ~10–15 minutes the first time: it installs CPU torch and
bakes in ~700 MB of weights so the first visitor isn't the one who downloads
them. When the Space goes green, open it — the FastAPI server serves the full UI
itself at that URL, so **this alone is already a working live app**. Confirm:

```bash
curl https://<your-space>.hf.space/health
```

Expect `{"status":"ok"}`. Copy that origin; it's `XAI_API_BASE` below.

> Free Spaces sleep after ~48h idle and cold-start on the next request. For a
> demo you're about to show someone, open it once first.

## 2. Front end — Vercel

**Import the repo** at <https://vercel.com/new>. Vercel reads `vercel.json`, so
leave framework preset as **Other** and don't touch the build settings.

**Set the environment variable** — *Settings → Environment Variables*, all
environments:

| | |
|---|---|
| Name | `XAI_API_BASE` |
| Value | `https://<your-space>.hf.space` |

No trailing slash, and it must be `https` — a browser on an https page refuses to
call an http backend. The build **fails loudly** if this is missing rather than
shipping a page that 404s every request against its own static host.

**Deploy.** The build copies `templates/index.html` → `public/index.html`,
injecting one line before `</head>`:

```html
<script>window.XAI_API_BASE="https://<your-space>.hf.space";</script>
```

## 3. Lock down CORS

Until you do this, any site can spend your daily model allowance. Back in the
Space, *Settings → Variables and secrets*, add a **variable** (not a secret):

| | |
|---|---|
| Name | `ALLOWED_ORIGINS` |
| Value | `https://<your-app>.vercel.app` |

Comma-separate to allow more than one. Unset means any origin — fine while
testing, wrong once it's public. The Space restarts on save.

---

## Verifying it's actually live

Open the Vercel URL and check, in order:

1. **Page loads, logo renders.** If the logo 404s, `public/static/` didn't build.
2. **DevTools → Network.** A Summarize request should go to `<your-space>.hf.space`, not the Vercel domain. If it hits Vercel, `XAI_API_BASE` wasn't set at build time — set it and **redeploy**, since it's baked in at build, not read at runtime.
3. **The progress bar moves.** This is the real integration test: it means `/api/progress/{job_id}` polls are reaching the same process that's doing the work.
4. **Ask a second question about the same audio file.** It should skip transcription entirely — proof the in-process cache survived the move.

## When something's wrong

**Every request fails with a CORS error.** `ALLOWED_ORIGINS` doesn't match your
Vercel origin exactly — scheme included, no trailing slash. Preview deployments
get their own subdomains, so add them or leave the variable unset while testing.

**Space build fails on `pip install`.** Almost always the torch line. It installs
from `download.pytorch.org/whl/cpu` on purpose: PyPI's default Linux torch is the
CUDA build and pulls ~2.5 GB of `nvidia-*` wheels for a GPU the free tier hasn't
got.

**Everything answers, but only from the passage — never grounded.** The key isn't
reaching the container, or the day's 20 hosted calls are spent. Check the Space
logs; `utils.py` calls `load_dotenv()` at import, so a key added after boot needs
a restart — *Settings → Factory rebuild* is the reliable one.

**Uploads over ~4.5 MB fail.** They shouldn't — the front end posts directly to
the Space precisely so multipart bodies never cross Vercel's proxy limit. If you
see this, `API_BASE` is empty and the page is calling itself; see (2) above.

**Document analysis times out.** It doesn't on the Space — there's no request
ceiling there. A ~2 minute run is normal for a 25-page PDF; `DOC_MAX_PAGES` and
`DOC_GEMINI_PAGES` are the knobs.

## Running the container locally

Worth doing once before pushing — it catches a broken build in 2 minutes instead
of 15:

```bash
docker build -t xai-attribution . && docker run --rm -p 7860:7860 -e GEMINI_API_KEY="$GEMINI_API_KEY" xai-attribution
```

Then <http://localhost:7860> — the container serves the full UI, no Vercel
needed. To test the split front end against it:

```bash
XAI_API_BASE=http://localhost:7860 bash deploy/vercel-build.sh && python -m http.server 3000 --directory public
```

## What this does not change

Nothing about a normal local run. `API_BASE` is `""` unless something injects
`window.XAI_API_BASE`, and FastAPI serving `templates/index.html` injects
nothing — so same-origin, exactly as before:

```bash
./.venv311/Scripts/python.exe -m uvicorn Decision_Attribution.main:app --port 8020
```
