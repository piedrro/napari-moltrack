"""Cancellation at the QRunnable boundary cannot publish a completed result."""

import threading
from concurrent.futures import CancelledError

from qtpy.QtCore import QThreadPool

from moltrack.funcs.compute_utils import Worker


def test_running_worker_stop_suppresses_result_and_finishes_once(qtbot):
    entered, release = threading.Event(), threading.Event()

    def compute(progress_callback=None):
        entered.set()
        assert release.wait(5)
        return "pending scientific result"

    worker = Worker(compute)
    results, finished, errors = [], [], []
    worker.signals.result.connect(results.append)
    worker.signals.finished.connect(lambda: finished.append(True))
    worker.signals.error.connect(errors.append)
    pool = QThreadPool()
    pool.start(worker)
    try:
        qtbot.waitUntil(entered.is_set, timeout=5000)
        worker.stop()
        release.set()
        assert pool.waitForDone(5000)
        qtbot.waitUntil(lambda: bool(finished), timeout=5000)
        qtbot.wait(20)
        assert results == []
        assert finished == [True]
        assert errors == []
    finally:
        release.set()
        pool.waitForDone(5000)


def test_cooperative_worker_cancellation_is_not_an_error(qtbot):
    def compute(progress_callback=None):
        raise CancelledError()

    worker = Worker(compute)
    results, errors, finished = [], [], []
    worker.signals.result.connect(results.append)
    worker.signals.error.connect(errors.append)
    worker.signals.finished.connect(lambda: finished.append(True))
    worker.run()
    assert results == []
    assert errors == []
    assert finished == [True]
