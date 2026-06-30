from datetime import datetime, timedelta

from gridcast.scheduler import Job, run_pending


def test_jobs_run_on_first_pass_then_wait_for_interval():
    calls = []
    jobs = [Job("hourly", timedelta(hours=1), lambda: calls.append("h"))]
    start = datetime(2026, 1, 1, 8, 0)
    assert run_pending(jobs, start) == ["hourly"]
    assert run_pending(jobs, start + timedelta(minutes=30)) == []
    assert run_pending(jobs, start + timedelta(hours=1)) == ["hourly"]
    assert calls == ["h", "h"]


def test_failing_job_is_rescheduled_and_counted():
    def broken():
        raise RuntimeError("boom")

    job = Job("flaky", timedelta(minutes=10), broken)
    now = datetime(2026, 1, 1)
    run_pending([job], now)
    assert job.failures == 1
    assert job.next_run == now + timedelta(minutes=10)
