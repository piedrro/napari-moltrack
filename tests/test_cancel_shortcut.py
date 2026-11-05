import threading

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication


def test_pool_cancels_running_and_queued_workers(qtbot):
    from moltrack.funcs.compute_utils import CancellableThreadPool, Worker

    pool = CancellableThreadPool()
    pool.setMaxThreadCount(1)
    entered, release = threading.Event(), threading.Event()
    results, finished = [], []

    def compute(progress_callback=None):
        entered.set()
        assert release.wait(5)
        return "discard me"

    workers = [
        Worker(compute),
        Worker(lambda progress_callback=None: "queued"),
    ]
    for worker in workers:
        worker.signals.result.connect(results.append)
        worker.signals.finished.connect(lambda: finished.append(True))
        pool.start(worker)
    try:
        qtbot.waitUntil(entered.is_set)
        assert pool.cancel_all() == 2
        release.set()
        assert pool.waitForDone(5000)
        qtbot.waitUntil(lambda: len(finished) == 2)
        qtbot.waitUntil(lambda: not pool.has_workers())
        assert results == []
    finally:
        release.set()
        pool.waitForDone(5000)


@pytest.mark.parametrize("target", ["Tracking", "viewer"])
def test_ctrl_period_cancels_from_focused_control_and_waits_for_worker(
    qtbot, monkeypatch, target
):
    import napari

    from moltrack._widget import MolTrack
    from moltrack.funcs.compute_utils import Worker

    monkeypatch.setattr(MolTrack, "import_dev_data", lambda self: None)
    viewer = napari.Viewer(show=False)
    entered, release = threading.Event(), threading.Event()
    second_entered, second_release = threading.Event(), threading.Event()
    try:
        widget = MolTrack(viewer)
        viewer.window.add_dock_widget(widget)
        viewer.show()
        segmentation = widget.open_tool_window("Segmentation")
        segmentation.tool_selector.setCurrentIndex(2)
        # An idle shortcut must not poison the next operation.
        widget.cancel_active_operation()
        assert not widget.stop_event.is_set()
        widget.update_ui(init=True)
        results = []

        def compute(progress_callback=None):
            entered.set()
            assert release.wait(10)
            return "cancelled result"

        worker = Worker(compute)
        worker.signals.result.connect(results.append)
        worker.signals.finished.connect(widget.update_ui)
        widget.threadpool.start(worker)
        first_finished = []
        worker.signals.finished.connect(lambda: first_finished.append(True))

        def second_compute(progress_callback=None):
            second_entered.set()
            assert second_release.wait(10)
            return "another cancelled result"

        second = Worker(second_compute)
        second.signals.result.connect(results.append)
        second.signals.finished.connect(widget.update_ui)
        widget.threadpool.start(second)
        qtbot.waitUntil(entered.is_set)
        qtbot.waitUntil(second_entered.is_set)
        segmentation.close()
        if target == "viewer":
            control = widget.gui.moltrack_dataset_selector
            control.window().activateWindow()
        else:
            widget.open_tool_window(target)
            control = widget.gui.min_track_length
        control.setFocus()
        qtbot.waitUntil(
            lambda: QApplication.activeWindow() is control.window()
        )
        qtbot.keyClick(control, Qt.Key_Period, modifier=Qt.ControlModifier)
        qtbot.waitUntil(widget.stop_event.is_set)
        assert not widget.gui.fit_segmentations.isEnabled()
        release.set()
        qtbot.waitUntil(lambda: bool(first_finished))
        assert not widget.gui.fit_segmentations.isEnabled()
        second_release.set()
        qtbot.waitUntil(
            lambda: widget.gui.fit_segmentations.isEnabled(), timeout=10000
        )
        assert results == []
        assert widget.stop_event.is_set()
        widget.update_ui(init=True)
        assert not widget.stop_event.is_set()
        widget.update_ui()
    finally:
        release.set()
        second_release.set()
        if "widget" in locals():
            widget.threadpool.waitForDone(10000)
        viewer.close()
