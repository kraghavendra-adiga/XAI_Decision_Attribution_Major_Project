# Backend image for the split deploy: a static host serves templates/index.html,
# this container serves every /api and /explain-* route with the real models.
#
# Vercel cannot host the analysis side at all — its Python functions cap at
# 250 MB unzipped and torch alone is ~800 MB, Tesseract is a system binary with
# no apt-get available, and a document run measured ~2 minutes against a 60s
# (Hobby) / 300s (Pro) ceiling. Two features are also architecturally
# incompatible with stateless serverless: progress.py keeps its job registry in
# a module global and audio_explainer.py caches transcripts the same way, so a
# progress poll would land on a different instance and 404 forever. A long-lived
# container has none of those problems.
#
# Targets Hugging Face Spaces (Docker SDK, free CPU tier: 2 vCPU / 16 GB RAM),
# which is the only free tier large enough to hold torch, distilbart and Whisper
# at once. Nothing here is Spaces-specific beyond the port, so it runs the same
# on any container host.

# Pinned to bookworm rather than plain :3.11-slim on purpose. Debian's time64
# transition renamed several libraries in trixie (libglib2.0-0 became
# libglib2.0-0t64), so an unpinned base would one day fail at apt-get with a
# package-not-found that has nothing to do with anything in this repo.
FROM python:3.11-slim-bookworm

# System packages the app shells out to or links against. None are pip-installable:
#   tesseract-ocr      printed-text OCR, and the per-word boxes every heatmap needs
#   tesseract-ocr-eng  the English model — the base package ships no languages
#   ffmpeg             Whisper decodes audio through it
#   libgl1, libglib2.0-0, libsm6, libxext6  opencv-python's shared libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Spaces runs containers as a non-root uid 1000, and the model caches below have
# to be writable by whoever the process actually is.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR $HOME/app

# torch first, and from the CPU index deliberately. PyPI's default torch wheel
# for Linux is the CUDA build, which drags in ~2.5 GB of nvidia-* wheels for a
# GPU the free tier does not have. 2.11.0 / 0.26.0 is the upstream support
# matrix pair; installing them here also means the requirements.txt pass below
# cannot quietly resolve a different torch, which is the exact failure that
# file's comments warn about.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.11.0 torchvision==0.26.0

COPY --chown=user:user requirements.txt ./Decision_Attribution/requirements.txt
RUN pip install --no-cache-dir -r ./Decision_Attribution/requirements.txt

# Pull the model weights at build time instead of on the first request. They are
# ~700 MB and the download is otherwise paid for by whoever visits first, who
# would sit through several minutes with no way to tell the app apart from a
# broken one. Baking them in makes a cold start a load from disk.
# TrOCR (~240 MB, Accurate-mode handwriting) and EasyOCR's CRAFT weights are
# left to download lazily — both are optional at runtime and the app degrades to
# Tesseract without them.
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering; \
AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-6-6'); \
AutoModelForSeq2SeqLM.from_pretrained('sshleifer/distilbart-cnn-6-6'); \
AutoTokenizer.from_pretrained('distilbert-base-cased-distilled-squad'); \
AutoModelForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad'); \
import whisper; whisper.load_model('base'); \
print('models cached')"

# Copied last so a source edit does not invalidate the dependency and model
# layers above. The directory name matters: main.py prefers relative imports
# (from .schemas import ...) and only falls back to flat ones, so keeping the
# package name lets the container run the same import path as a local dev run.
COPY --chown=user:user . ./Decision_Attribution

# 7860 is what Spaces expects; app_port in README.md's frontmatter must agree.
EXPOSE 7860

# One worker on purpose. The progress registry and the transcript cache are both
# process-local, so a second worker would serve progress polls that know nothing
# about the job running in the first — the same break that ruled out serverless.
CMD ["uvicorn", "Decision_Attribution.main:app", \
     "--host", "0.0.0.0", "--port", "7860", "--workers", "1", \
     "--timeout-keep-alive", "75"]
