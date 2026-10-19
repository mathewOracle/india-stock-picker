import time

from app.data import RefreshResult
from app.jobs import RefreshJob


def _wait(job: RefreshJob, timeout: float = 2.0):
    deadline = time.time() + timeout
    while job.snapshot().state == "running" and time.time() < deadline:
        time.sleep(0.01)
    return job.snapshot()


def test_successful_refresh_reports_message():
    def runner(progress):
        progress(1, 1, "X.NS")
        return RefreshResult(ok=1, failed=[], seconds=0.1)

    job = RefreshJob(runner)
    assert job.start()
    status = _wait(job)
    assert status.state == "done" and "Updated 1" in status.message


def test_failure_is_surfaced_not_swallowed():
    def runner(progress):
        raise RuntimeError("proxy said no")

    job = RefreshJob(runner)
    job.start()
    status = _wait(job)
    assert status.state == "failed" and "proxy said no" in status.message


def test_cannot_start_twice_concurrently():
    def slow(progress):
        time.sleep(0.2)
        return RefreshResult(ok=0, failed=[], seconds=0.2)

    job = RefreshJob(slow)
    assert job.start() is True
    assert job.start() is False
    _wait(job)
