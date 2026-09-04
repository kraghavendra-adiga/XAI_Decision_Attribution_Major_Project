"""Audio and video lecture analysis with Whisper transcription and timestamp evidence."""

import base64
import difflib
import hashlib
import importlib
import json
import os
import re
import shutil as _shutil
import subprocess
import tempfile
import threading
from collections import OrderedDict
from contextlib import contextmanager

import cv2
import whisper
from PIL import Image

try:
    from .utils import (
        ServiceOverloaded,
        _extract_keywords,
        _image_to_b64,
        describe_service_error,
        gemini_model,
        not_answerable_rules,
        question_is_off_topic,
        reply_is_unsupported,
        unsupported_reply,
    )
except ImportError:
    from utils import (
        ServiceOverloaded,
        _extract_keywords,
        _image_to_b64,
        describe_service_error,
        gemini_model,
        not_answerable_rules,
        question_is_off_topic,
        reply_is_unsupported,
        unsupported_reply,
    )

# Whisper decodes audio by shelling out to `ffmpeg` by name, so the binary must
# be on PATH. imageio-ffmpeg ships a static build under a versioned filename;
# expose a copy named ffmpeg.exe so Whisper can find it without a system install.
if not _shutil.which("ffmpeg"):
    try:
        import imageio_ffmpeg

        _ffmpeg_src = imageio_ffmpeg.get_ffmpeg_exe()
        _ffmpeg_dir = os.path.join(tempfile.gettempdir(), "whisper_ffmpeg")
        os.makedirs(_ffmpeg_dir, exist_ok=True)
        _ffmpeg_dst = os.path.join(
            _ffmpeg_dir, "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
        )
        if not os.path.isfile(_ffmpeg_dst):
            _shutil.copy2(_ffmpeg_src, _ffmpeg_dst)
        os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        print(f"[explainer] Using bundled ffmpeg from {_ffmpeg_dir}")
    except Exception as _ffmpeg_err:
        print(f"[explainer] ffmpeg unavailable, audio will fail: {_ffmpeg_err}")

_whisper_model = None

# whisper.transcribe drives a tqdm bar over decoded mel frames and calls
# pbar.update() after each window; there is no callback parameter, so reading
# that bar is the only way to get true transcription progress. Swapping the name
# in whisper's own namespace is module-global, hence the lock.
_TQDM_LOCK = threading.Lock()


@contextmanager
def _whisper_progress(report):
    """Forward Whisper's real decode position to `report(0.0 .. 1.0)`.

    Only the `tqdm` name inside whisper.transcribe is replaced, not the tqdm
    package itself, so nothing else in the process sees the substitution. If the
    installed whisper is shaped differently the block still runs — it just
    reports no sub-stage progress rather than failing the transcription.
    """
    if report is None:
        yield
        return
    module = importlib.import_module("whisper.transcribe")
    original = getattr(module, "tqdm", None)
    base = getattr(original, "tqdm", None)
    if base is None:
        yield
        return

    class _ReportingBar(base):
        def update(self, n=1):
            result = super().update(n)
            total = self.total or 0
            if total > 0:
                report(min(1.0, self.n / total))
            return result

    class _Namespace:
        tqdm = _ReportingBar

    with _TQDM_LOCK:
        module.tqdm = _Namespace
        try:
            yield
        finally:
            module.tqdm = original

__all__ = [
    "VIDEO_EXTENSIONS",
    "audio_stage_plan",
    "is_transcribed",
    "load_whisper",
    "extract_audio_from_video",
    "extract_frame_at_timestamp",
    "analyze_frame_with_gemini",
    "explain_audio",
    "_format_timestamp",
    "_format_range",
    "_normalize_token",
    "_normalized_tokens",
    "_join_words",
    "_align_phrase",
    "_best_segment_span",
]


def load_whisper():
    global _whisper_model
    if _whisper_model is None:
        print("[explainer] Loading Whisper model...")
        _whisper_model = whisper.load_model("base")
    return _whisper_model


VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".avi",
    ".mkv", ".webm", ".flv", ".wmv",
}


# ── Transcript cache ────────────────────────────────────────────────────────
#
# Transcription is 70-82% of an audio request and minutes of wall clock on a
# long recording, and it produces the same text every time — so asking a second
# question about the same file used to mean waiting through the whole of Whisper
# again for a transcript already in hand. Keyed on the bytes rather than the
# filename: a renamed file is the same recording, and two different recordings
# that happen to share a name are not.
#
# Bounded by count rather than by age. What makes an entry worth keeping is that
# the reader is still asking about that file, and eight recordings is more than a
# session works through; the store dies with the process either way.
_TRANSCRIPT_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_TRANSCRIPT_CACHE_MAX = 8
_TRANSCRIPT_LOCK = threading.Lock()


def _cache_key(audio_bytes: bytes) -> str:
    return hashlib.sha256(audio_bytes or b"").hexdigest()


def _cached_transcript(key: str) -> dict | None:
    """The stored transcription for these bytes, as a copy, or None.

    Copied on the way out so a caller that annotates segments or word timings
    cannot write through to the next question's cache hit.
    """
    with _TRANSCRIPT_LOCK:
        entry = _TRANSCRIPT_CACHE.get(key)
        if entry is None:
            return None
        _TRANSCRIPT_CACHE.move_to_end(key)
        return {
            "transcript": entry["transcript"],
            "segments": [dict(seg) for seg in entry["segments"]],
            "word_timestamps": [dict(w) for w in entry["word_timestamps"]],
        }


def _store_transcript(
    key: str, transcript: str, segments: list, word_timestamps: list
) -> None:
    with _TRANSCRIPT_LOCK:
        _TRANSCRIPT_CACHE[key] = {
            "transcript": transcript,
            "segments": segments,
            "word_timestamps": word_timestamps,
        }
        _TRANSCRIPT_CACHE.move_to_end(key)
        while len(_TRANSCRIPT_CACHE) > _TRANSCRIPT_CACHE_MAX:
            _TRANSCRIPT_CACHE.popitem(last=False)


def is_transcribed(audio_bytes: bytes) -> bool:
    """Has this exact recording already been transcribed in this process?

    Public so the endpoint can pick its stage plan before the work starts — a
    request that skips transcription must not reserve four fifths of the bar for
    it. Only a hint: if the entry is evicted between this call and the work, the
    transcription simply runs and reports nothing, which the Reporter allows.
    """
    with _TRANSCRIPT_LOCK:
        return _cache_key(audio_bytes) in _TRANSCRIPT_CACHE


def audio_stage_plan(
    filename: str, transcribed: bool = False
) -> list[tuple[str, float]]:
    """Stage weights for `explain_audio`, for the caller to build a Reporter with.

    The labels live here, next to the code that reports them, so the two cannot
    drift apart. Weights are shares of wall clock measured on a few-minute clip:
    transcription dominates everything else, and reading keyframes only happens
    for video, so audio-only omits that stage rather than leaving a dead span the
    bar would have to skip over.

    `transcribed` says the recording is already in the cache. Its stage is then
    left out entirely — the weights are relative, so what is left simply spreads
    across the bar, and the reader sees a request that is mostly answering.
    """
    is_video = os.path.splitext(filename or "")[-1].lower() in VIDEO_EXTENSIONS
    stages = [("Reading file", 2)]
    if not transcribed:
        stages.append(("Transcribing audio", 70 if is_video else 82))
    stages += [("Answering question", 12), ("Locating timestamps", 3)]
    if is_video:
        stages.append(("Reading keyframes", 12))
    stages.append(("Finishing up", 1))
    return stages


def extract_audio_from_video(video_path: str) -> str:
    """Strip a 16kHz mono WAV audio track out of a video for Whisper.

    Returns the path to the extracted .wav, which the caller owns and must
    delete. Raises RuntimeError with a readable message if ffmpeg is missing
    or the extraction fails.
    """
    ffmpeg_exe = _shutil.which("ffmpeg")
    if not ffmpeg_exe:
        raise RuntimeError(
            "ffmpeg was not found on PATH, so audio cannot be extracted from "
            "video. Install ffmpeg, or `pip install imageio-ffmpeg` so the "
            "bundled binary is used."
        )

    fd, output_wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    command = [
        ffmpeg_exe, "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        "-y", output_wav_path,
    ]
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        if os.path.exists(output_wav_path):
            os.unlink(output_wav_path)
        stderr_tail = (process.stderr or "").strip()[-500:]
        raise RuntimeError(f"ffmpeg failed to extract audio: {stderr_tail}")

    return output_wav_path


def extract_frame_at_timestamp(
    video_path: str,
    timestamp_seconds: float
) -> str | None:
    """Grab the video frame shown at a timestamp, as a base64 PNG.

    Returns None if the frame cannot be read, so a missing keyframe degrades
    the response instead of failing the whole request.
    """
    capture = None
    try:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"cv2 could not open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        # Some containers report 0 or NaN; fall back to a sane default.
        if not fps or fps <= 0 or fps != fps:
            fps = 25.0

        frame_number = int(timestamp_seconds * fps)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(
                f"no frame at {timestamp_seconds}s (frame {frame_number})"
            )

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)

        if image.width > 640:
            scale = 640 / image.width
            image = image.resize(
                (640, max(1, int(image.height * scale))),
                Image.LANCZOS,
            )

        return _image_to_b64(image)
    except Exception as e:
        print(f"[Frame extraction error] {type(e).__name__}: {e}")
        return None
    finally:
        if capture is not None:
            capture.release()


def analyze_frame_with_gemini(frame_b64: str, context: str) -> str:
    """Describe what is on screen in a keyframe, given the topic being spoken.

    Returns "" on any failure so a missing description never breaks the
    surrounding attribution response.
    """
    try:
        image_bytes = base64.b64decode(frame_b64)
        prompt_text = (
            "You are analyzing a frame from an educational video lecture at "
            "the moment the following topic was being discussed: "
            f"{context}\n\n"
            "Describe what is visible on screen — slides, text, diagrams, "
            "equations, whiteboard content, or what the lecturer is doing. "
            "Be specific and concise. Max 2 sentences."
        )
        model = gemini_model()
        response = model.generate_content([
            {"mime_type": "image/png", "data": image_bytes},
            prompt_text,
        ])
        return response.text.strip()
    except Exception as e:
        print(f"[Frame analysis error] {type(e).__name__}: {e}")
        return ""


# ── Locating a quoted phrase in the transcript ──────────────────────────────

def _normalize_token(token: str) -> str:
    """Strip a token down to comparable characters ("ATP," -> "atp")."""
    return re.sub(r"[^a-z0-9]", "", str(token or "").lower())


def _normalized_tokens(text: str) -> list[str]:
    return [
        norm for norm in
        (_normalize_token(word) for word in re.findall(r"[\w']+", str(text or "")))
        if norm
    ]


# Tokens that attach to the previous word rather than starting a new one.
_BINDS_LEFT = tuple("-'’,.!?;:%)]}")
_BINDS_RIGHT = ("(", "[", "{", "-", "'", "’")


def _join_words(words: list[dict]) -> str:
    """Re-join word tokens without inserting spaces that were never spoken.

    Whisper emits hyphenated and contracted words as separate tokens ("re",
    "-enter"), so a plain join renders "re -enter".
    """
    parts: list[str] = []
    for word in words:
        token = str(word.get("word", "")).strip()
        if not token:
            continue
        if parts and (token.startswith(_BINDS_LEFT) or parts[-1].endswith(_BINDS_RIGHT)):
            parts[-1] += token
        else:
            parts.append(token)
    return " ".join(parts).strip()


def _align_phrase(phrase: str, words: list[dict]) -> dict | None:
    """Pin a quoted phrase to the words Whisper actually timed.

    Whisper reports per-word start/end times, so a phrase can be located to the
    moment it was spoken rather than to whichever segments happen to share a
    keyword. Returns None when nothing plausible lines up, letting the caller
    fall back to segment scoring.
    """
    target = _normalized_tokens(phrase)
    if not target or not words:
        return None

    stream = [word["norm"] for word in words]
    matcher = difflib.SequenceMatcher(a=stream, b=target, autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size]
    if not blocks:
        return None

    lo = min(block.a for block in blocks)
    hi = max(block.a + block.size for block in blocks)

    # One word repeated across the recording ("ATP") can otherwise stretch the
    # span over the whole lecture; fall back to the longest single run.
    max_span = max(len(target) * 2, len(target) + 10)
    if hi - lo > max_span:
        longest = max(blocks, key=lambda block: block.size)
        lo, hi = longest.a, longest.a + longest.size

    matched = sum(block.size for block in blocks if lo <= block.a < hi)
    return {
        "start": float(words[lo]["start"]),
        "end": float(words[hi - 1]["end"]),
        "text": _join_words(words[lo:hi]),
        "ratio": matched / len(target),
    }


def _best_segment_span(phrase: str, segments: list[dict]) -> dict | None:
    """Segment-level fallback for when word timings are missing.

    Keeps the best-scoring contiguous run of segments. Unioning every segment
    that shares a keyword fuses unrelated parts of the recording into one
    quote and drags the start time away from the phrase.
    """
    keywords = _extract_keywords(phrase)
    if not keywords or not segments:
        return None

    scores = []
    for segment in segments:
        segment_keywords = _extract_keywords(segment["text"])
        scores.append(
            len(keywords & segment_keywords) / len(keywords)
            if segment_keywords else 0.0
        )

    anchor = max(range(len(scores)), key=scores.__getitem__)
    if scores[anchor] < 0.2:
        return None

    # Grow outward only while neighbours are comparably strong.
    threshold = max(0.2, scores[anchor] * 0.5)
    lo = hi = anchor
    while lo > 0 and scores[lo - 1] >= threshold:
        lo -= 1
    while hi + 1 < len(segments) and scores[hi + 1] >= threshold:
        hi += 1

    return {
        "start": float(segments[lo]["start"]),
        "end": float(segments[hi]["end"]),
        "text": " ".join(
            segment["text"].strip() for segment in segments[lo:hi + 1]
        ).strip(),
        "ratio": scores[anchor],
    }


def _audio_declined(answer: str, transcript: str, segments: list) -> dict:
    """The audio path's result when the recording cannot answer the question.

    Evidence and chart go out empty: there is no phrase to pin to a timestamp and
    no keyframe worth pulling, which is what the 50/50 donut was made of. The
    transcript stays — it is how the reader checks the refusal is honest.
    """
    return {
        "answer": answer,
        "transcript": transcript,
        "segments": segments,
        "timestamp_evidence": [],
        "chart_data": [],
        "unsupported": True,
    }


def explain_audio(
    audio_bytes: bytes,
    audio_filename: str,
    question: str,
    reporter=None,
):
    """Answer `question` from a recording and pin each source to its timestamp.

    `reporter` is optional and only needs `.stage()`, `.within()` and `.sub()`
    (see progress.Reporter); its labels come from `audio_stage_plan`. Passing
    None disables reporting without changing any of the work below.
    """
    tmp_path = None
    audio_path = None
    video_tmp_path = None

    def stage(label):
        if reporter:
            reporter.stage(label)

    try:
        stage("Reading file")
        suffix = os.path.splitext(audio_filename)[-1] or ".mp3"
        file_ext = suffix.lower()
        is_video = file_ext in VIDEO_EXTENSIONS

        cache_key = _cache_key(audio_bytes)
        cached = _cached_transcript(cache_key)

        # A cached audio-only request never touches the disk: the file was only
        # ever written out so ffmpeg and Whisper could read it, and neither runs.
        # Video still needs it, for the keyframes further down.
        if cached is None or is_video:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            # Keep the video around after transcription so keyframes can be read
            # from it further down; it is deleted once frames are extracted.
            video_tmp_path = tmp_path if is_video else None

        if cached is not None:
            # Same bytes, same transcript. Asking a second question about a
            # recording used to mean sitting through the whole of Whisper again
            # for text that was already in hand.
            full_transcript = cached["transcript"]
            segments = cached["segments"]
            word_timestamps = cached["word_timestamps"]
            print(
                "[explainer] Reusing cached transcript "
                f"({len(full_transcript)} chars, {len(segments)} segments)"
            )
        else:
            if is_video:
                print(f"[explainer] Video detected ({file_ext}), extracting audio...")
                audio_path = extract_audio_from_video(tmp_path)
            else:
                audio_path = tmp_path

            stage("Transcribing audio")
            model = load_whisper()
            # The bar advances on Whisper's own decoded-frame count, so it stops
            # moving exactly when decoding stalls.
            with _whisper_progress(
                reporter.sub("Transcribing audio") if reporter else None
            ):
                result = model.transcribe(
                    audio_path,
                    word_timestamps=True,
                    verbose=False
                )

            # Drop the extracted wav now; the video itself is still needed.
            if is_video:
                if audio_path != tmp_path and os.path.exists(audio_path):
                    os.unlink(audio_path)
                    audio_path = None
            elif os.path.exists(tmp_path):
                os.unlink(tmp_path)
                tmp_path = None
                audio_path = None

            word_timestamps = []
            for segment in result["segments"]:
                for word_data in segment.get("words", []):
                    word_timestamps.append({
                        "word": word_data["word"].strip(),
                        "start": round(float(word_data["start"]), 2),
                        "end": round(float(word_data["end"]), 2),
                    })

            full_transcript = result["text"].strip()

            # Also build segment-level timestamps for evidence
            segments = [
                {
                    "text": seg["text"].strip(),
                    "start": round(float(seg["start"]), 2),
                    "end": round(float(seg["end"]), 2),
                }
                for seg in result["segments"]
            ]

            _store_transcript(
                cache_key, full_transcript, segments, word_timestamps
            )

        # The recording never mentions what the question is about, so the answer
        # is already known and no model is needed to write it. Settled here rather
        # than after the call for two reasons: this path is the only one with no
        # local answerer, so when the hosted allowance is spent asking anyway
        # turns an answerable "that is not in this recording" into an error; and
        # the timestamps and keyframes below would be pinned to phrases nothing
        # cited.
        if question_is_off_topic(question, full_transcript):
            stage("Answering question")
            print("[explainer] Question is off-topic for this transcript")
            stage("Finishing up")
            return _audio_declined(
                unsupported_reply("transcript", question, full_transcript),
                full_transcript,
                segments,
            )

        prompt = (
            "You are analyzing an educational audio transcript.\n"
            "Respond with ONLY a JSON object. "
            "No preamble, no explanation, no markdown.\n"
            "Start your response with { and end with }.\n\n"
            "Format:\n"
            "{\n"
            '  "answerable": true,\n'
            '  "answer": "complete answer here",\n'
            '  "sources": [\n'
            '    {\n'
            '      "text": "exact phrase from transcript",\n'
            '      "contribution": 0.6,\n'
            '      "reason": "why this contributed"\n'
            '    }\n'
            '  ]\n'
            "}\n\n"
            "Rules:\n"
            "- answer must be complete and generalized\n"
            "- if multiple points exist list all of them\n"
            "- sources must be exact phrases from transcript\n"
            "- contribution values must add up to 1.0\n"
            "- between 2 and 5 sources when the transcript can answer\n"
            "- never add information not in the transcript\n"
            f"{not_answerable_rules('transcript')}\n\n"
            f"TRANSCRIPT:\n{full_transcript}\n\n"
            f"QUESTION: {question}"
        )
        
        genai_model = gemini_model()
        stage("Answering question")
        try:
            response = genai_model.generate_content(prompt)
            raw = response.text.strip()
        except Exception as exc:
            # The transcript is already done at this point, but without an answer
            # there is nothing to attribute, so this has to surface as a failure —
            # just one the user can read and act on.
            raise ServiceOverloaded(describe_service_error(exc)) from exc
        
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
            print(f"[Audio JSON error] {e}")
            print(f"[Raw Gemini output] {raw[:500]}")
            raise ServiceOverloaded(describe_service_error(e)) from e

        answer = parsed.get("answer", "").strip()
        sources = parsed.get("sources", [])

        # The model looked at the transcript and said the same thing the test
        # above does — worth keeping both, because the local test only fires when
        # not one word of the question appears, and the model catches the rest.
        # Temp files are left to the `finally` block, which unlinks them anyway.
        if answer and reply_is_unsupported(parsed, answer):
            stage("Finishing up")
            return _audio_declined(answer, full_transcript, segments)

        stage("Locating timestamps")

        # Whisper's per-word timings let each quoted phrase be pinned to the
        # moment it was spoken. Scoring segments by keyword overlap and taking
        # the union instead fused non-adjacent parts of the recording into one
        # garbled quote, and pulled the start time — and so the keyframe — well
        # away from the phrase being cited.
        word_stream = []
        for word in word_timestamps:
            norm = _normalize_token(word["word"])
            if norm:
                word_stream.append({**word, "norm": norm})

        timestamp_evidence = []

        for source in sources:
            source_text = source.get("text", "")
            contribution = float(source.get("contribution", 0.0))
            reason = source.get("reason", "")

            located = _align_phrase(source_text, word_stream)
            if not located or located["ratio"] < 0.5:
                located = _best_segment_span(source_text, segments) or located

            if located:
                start_ts = round(located["start"], 2)
                end_ts = round(located["end"], 2)
                matched_text = located["text"]
            else:
                start_ts = 0.0
                end_ts = 0.0
                matched_text = source_text

            timestamp_evidence.append({
                "text": source_text,
                "matched_transcript": matched_text,
                "start": start_ts,
                "end": end_ts,
                "timestamp_label": _format_timestamp(start_ts),
                "end_label": _format_timestamp(end_ts),
                "timestamp_range": _format_range(start_ts, end_ts),
                "contribution": round(contribution * 100, 1),
                "reason": reason,
            })

        # For video, attach what was on screen at each evidence timestamp:
        # the keyframe itself, then a short description of what it shows.
        # This must run before the video temp file is deleted below.
        if is_video and timestamp_evidence:
            stage("Reading keyframes")
        for index, evidence in enumerate(timestamp_evidence):
            if is_video and video_tmp_path:
                frame_b64 = extract_frame_at_timestamp(
                    video_tmp_path, evidence["start"]
                )
                evidence["keyframe"] = frame_b64
                if frame_b64:
                    evidence["frame_description"] = analyze_frame_with_gemini(
                        frame_b64, evidence["text"]
                    )
                else:
                    evidence["frame_description"] = ""
                if reporter:
                    reporter.within((index + 1) / len(timestamp_evidence))
            else:
                evidence["keyframe"] = None
                evidence["frame_description"] = ""

        stage("Finishing up")
        # Frames are extracted, so the video temp file can go now.
        if video_tmp_path and os.path.exists(video_tmp_path):
            os.unlink(video_tmp_path)
        if tmp_path == video_tmp_path:
            tmp_path = None
        video_tmp_path = None

        # Full phrase, not a 40-char prefix: the legend needs to be
        # readable and to match the evidence card word for word.
        chart_data = [
            {
                "label": e["text"],
                "full_text": e["text"],
                "contribution": e["contribution"],
                "reason": e["reason"],
                "timestamp_label": e["timestamp_label"],
                "timestamp_range": e["timestamp_range"],
            }
            for e in timestamp_evidence
        ]

        return {
            "answer": answer,
            "transcript": full_transcript,
            "segments": segments,
            "timestamp_evidence": timestamp_evidence,
            "chart_data": chart_data,
            "unsupported": False,
        }

    except Exception as e:
        print(f"[Audio error] {type(e).__name__}: {e}")
        raise

    finally:
        # Safety net: an early failure can leave temp files behind, and the
        # video path is deliberately kept alive past transcription.
        for leftover in {tmp_path, audio_path, video_tmp_path}:
            if leftover and os.path.exists(leftover):
                try:
                    os.unlink(leftover)
                except OSError as cleanup_err:
                    print(f"[explainer] Could not remove {leftover}: {cleanup_err}")

def _format_timestamp(seconds: float) -> str:
    # Round to the nearest tenth instead of truncating each field separately.
    # (7.1 % 1) * 10 is 0.9999999999999964 in binary floating point, so int()
    # cut it to 0 and 7.1s displayed as "00:07.0" — and 58.9s as "00:58.8".
    tenths_total = int(round(max(0.0, float(seconds)) * 10))
    mins, rem = divmod(tenths_total, 600)
    secs, tenths = divmod(rem, 10)
    return f"{mins:02d}:{secs:02d}.{tenths}"


def _format_range(start: float, end: float) -> str:
    """The span a quote occupies, as MM:SS.t-MM:SS.t.

    Falls back to the start alone when there is no span to show: an unlocatable
    phrase leaves both ends at 0.0, and a single short word can round to the
    same tenth at both ends, where "00:03.4-00:03.4" reads as a bug rather than
    as a duration.
    """
    start_label = _format_timestamp(start)
    end_label = _format_timestamp(end)
    if end <= start or end_label == start_label:
        return start_label
    return f"{start_label}–{end_label}"
