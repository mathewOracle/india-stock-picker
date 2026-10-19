"""Run the (slow) data refresh in a background thread and expose its progress.

The web request that clicks "Refresh" returns instantly; the browser polls
`snapshot()` until the job is done or has failed -- no more silent failures.
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class JobStatus:
    state: str  # idle | running | done | failed
    message: str = ""
    done: int = 0
    total: int = 0
    finished_at: float | None = None


class RefreshJob:
    def __init__(self, runner: Callable):
        self._runner = runner
        self._lock = threading.Lock()
        self._status = JobStatus("idle")

    def start(self) -> bool:
        """Kick off a refresh. Returns False if one is already running."""
        with self._lock:
            if self._status.state == "running":
                return False
            self._status = JobStatus("running", "Starting...")
        threading.Thread(target=self._run, daemon=True).start()
        return True

    def snapshot(self) -> JobStatus:
        return self._status

    def _progress(self, done: int, total: int, symbol: str) -> None:
        self._status = JobStatus("running", f"Fetched {symbol}", done, total)

    def _run(self) -> None:
        try:
            result = self._runner(progress=self._progress)
            self._status = JobStatus("done", result.message, finished_at=time.time())
        except Exception as exc:  # surface *any* failure to the UI instead of swallowing it
            self._status = JobStatus("failed", str(exc) or type(exc).__name__, finished_at=time.time())
