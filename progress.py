"""Server-side progress reporting for long requests.

The analysis endpoints take tens of seconds — transcription alone can run for
minutes — and a spinner cannot say whether a request is a third of the way in or
about to time out. So each stage of the work writes its real position here, and
the browser polls GET /api/progress/{job_id} to draw a determinate bar.

Progress is deliberately not derived from elapsed time. Whisper reports the audio
frames it has actually decoded, LIME reports the perturbation batches it has
actually scored, and SHAP reports evaluations against its own budget, so the bar
stalls when the work stalls instead of gliding to 99% and waiting there.

The store is in-memory and process-local, which is the right scope: a job only
matters to the one request that created it and the one browser tab polling it.
"""

import threading
import time

__all__ = ["Reporter", "start", "get", "fail", "finish", "reap"]

# job_id -> {stage, pct, done, error, updated}
_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()

# A tab that navigates away mid-request leaves its job behind, so entries are
# dropped once nothing has polled or written them for this long.
_TTL_SECONDS = 900


def _now() -> float:
    return time.monotonic()


def reap() -> None:
    """Drop jobs nothing has touched for a while."""
    cutoff = _now() - _TTL_SECONDS
    with _LOCK:
        for job_id in [k for k, v in _JOBS.items() if v["updated"] < cutoff]:
            del _JOBS[job_id]


def start(job_id: str, stage: str = "Starting") -> None:
    if not job_id:
        return
    reap()
    with _LOCK:
        _JOBS[job_id] = {
            "stage": stage,
            "pct": 0.0,
            # False means "inside a stage that cannot count its own internals" —
            # a single hosted-model call, for instance. The client sweeps the track
            # instead of implying a number it does not have.
            "precise": False,
            "done": False,
            "error": None,
            "updated": _now(),
        }


def _write(job_id: str, **fields) -> None:
    if not job_id:
        return
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job.update(fields)
        job["updated"] = _now()


def get(job_id: str) -> dict | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        job["updated"] = _now()
        return {k: v for k, v in job.items() if k != "updated"}


def finish(job_id: str, stage: str = "Done") -> None:
    _write(job_id, stage=stage, pct=100.0, precise=True, done=True)


def fail(job_id: str, message: str) -> None:
    _write(job_id, done=True, error=str(message))


class Reporter:
    """Writes one job's progress, one weighted stage at a time.

    A stage is declared with the slice of the overall bar it owns, so a stage
    that reports 0..1 internally lands in the right place globally:

        rep = Reporter(job_id, [("Transcribing", 60), ("Attributing", 40)])
        rep.stage("Transcribing")
        rep.within(0.5)          # -> 30% overall
        rep.stage("Attributing") # -> 60% overall

    An unknown stage name is a no-op rather than an error: progress reporting
    must never be the reason a request fails.
    """

    def __init__(self, job_id: str, stages: list[tuple[str, float]]):
        self.job_id = job_id or ""
        total = sum(max(0.0, float(w)) for _, w in stages) or 1.0
        self._spans: dict[str, tuple[float, float]] = {}
        cursor = 0.0
        for label, weight in stages:
            width = max(0.0, float(weight)) / total * 100.0
            self._spans[label] = (cursor, width)
            cursor += width
        self._current = None

    def stage(self, label: str) -> None:
        """Enter a stage, jumping the bar to its starting point.

        `precise` drops to False here: until the stage reports from inside itself,
        the only truthful claim is "this much of the run is behind us", not a
        position within the step now running.
        """
        if label not in self._spans:
            return
        self._current = label
        base, _ = self._spans[label]
        _write(self.job_id, stage=label, pct=round(base, 1), precise=False)

    def within(self, fraction: float) -> None:
        """Report position inside the current stage, as 0.0 .. 1.0."""
        if self._current is None:
            return
        base, width = self._spans[self._current]
        frac = min(1.0, max(0.0, float(fraction)))
        _write(
            self.job_id,
            pct=round(base + width * frac, 1),
            precise=True,
        )

    def sub(self, label: str):
        """A callback that reports 0..1 inside `label`, for passing downwards.

        The explainers take a plain callable so they stay independent of this
        module — they receive `report(fraction)` and know nothing about jobs.
        """
        def report(fraction: float) -> None:
            if self._current != label:
                self.stage(label)
            self.within(fraction)
        return report

    def done(self, stage: str = "Done") -> None:
        finish(self.job_id, stage)
