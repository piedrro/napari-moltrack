from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QFileDialog, QPushButton, QScrollArea


@pytest.fixture
def launcher(qtbot, monkeypatch):
    import napari

    from moltrack._widget import MolTrack

    monkeypatch.setattr(MolTrack, "import_dev_data", lambda self: None)
    viewer = napari.Viewer(show=False)
    try:
        dock, widget = viewer.window.add_plugin_dock_widget("napari-moltrack")
        viewer.show()
        qtbot.waitUntil(widget.isVisible)
        yield viewer, widget, dock
    finally:
        viewer.close()


def test_all_section_tools_are_reachable(launcher):
    _, widget, _ = launcher
    expected = {
        "Import & Project": 4,
        "Segmentation": 3,
        "Localisation": 6,
        "Tracking": 8,
        "Heatmaps": 3,
        "Export": 5,
        "Utilities": 3,
    }
    buttons = widget.findChildren(
        QPushButton, options=Qt.FindDirectChildrenOnly
    )
    assert [button.text().replace("&&", "&") for button in buttons] == list(
        expected
    )
    assert all(
        not window.isVisible() for window in widget.tool_windows.values()
    )
    for section, count in expected.items():
        window = widget.open_tool_window(section)
        assert window.isWindow() and window.windowModality() == Qt.NonModal
        assert all(
            not button.autoDefault()
            for button in window.findChildren(QPushButton)
        )
        assert window.tool_selector.count() == window.pages.count() == count
        for index in range(count):
            window.tool_selector.setCurrentIndex(index)
            assert window.pages.currentIndex() == index
            assert window.pages.currentWidget().isVisible()
    assert (
        sum(window.isVisible() for window in widget.tool_windows.values()) == 7
    )


def test_reopening_tool_restores_state_and_launcher_hides_windows(
    launcher, qtbot
):
    _, widget, dock = launcher
    button = next(
        button
        for button in widget.findChildren(QPushButton)
        if button.text() == "Segmentation"
    )
    qtbot.mouseClick(button, Qt.LeftButton)
    segmentation = widget.tool_windows["Segmentation"]
    segmentation.tool_selector.setCurrentIndex(2)
    widget.gui.fit_min_radius.setValue(5)
    tracking = widget.open_tool_window("Tracking")
    assert tracking.isVisible() and segmentation.isVisible()

    segmentation.close()
    assert not segmentation.isVisible() and tracking.isVisible()
    assert widget.open_tool_window("Segmentation") is segmentation
    qtbot.waitUntil(lambda: QApplication.activeWindow() is segmentation)
    assert segmentation.tool_selector.currentIndex() == 2
    assert widget.gui.fit_min_radius.value() == 5

    qtbot.mouseClick(dock.titleBarWidget().close_button, Qt.LeftButton)
    qtbot.waitUntil(
        lambda: not segmentation.isVisible() and not tracking.isVisible()
    )
    assert not dock.isVisible()
    dock.show()
    assert not segmentation.isVisible() and not tracking.isVisible()
    assert widget.open_tool_window("Segmentation") is segmentation


def test_viewer_close_disposes_tool_windows(launcher, qtbot):
    viewer, widget, _ = launcher
    window = widget.open_tool_window("Tracking")
    with qtbot.waitSignal(window.destroyed):
        viewer.close()


def test_tracking_finishes_after_closing_its_window(
    launcher, qtbot, tmp_path, monkeypatch
):
    from moltrack.funcs import import_utils

    viewer, widget, _ = launcher
    image_path = tmp_path / "sample.tif"
    tifffile.imwrite(
        image_path,
        np.arange(600, dtype=np.uint16).reshape(6, 10, 10),
        photometric="minisblack",
    )
    locs_path = tmp_path / "localisations.csv"
    locs_path.write_text(
        "frame,x,y\n" + "".join(f"{frame},4,5\n" for frame in range(6))
    )
    monkeypatch.setattr(
        QFileDialog, "getOpenFileNames", lambda *args: ([str(image_path)], "")
    )
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *args: (str(locs_path), "")
    )
    # Single-file import uses a thread. Its progress mapping can stay in-process.
    monkeypatch.setattr(
        import_utils,
        "Manager",
        lambda: nullcontext(SimpleNamespace(dict=dict)),
    )

    importer = widget.open_tool_window("Import & Project")
    widget.gui.import_mode.setCurrentText("Data (Single Channel)")
    widget.gui.import_channel_name.setText("image")
    widget.gui.import_images.click()
    dataset = image_path.name
    qtbot.waitUntil(lambda: widget.active_dataset == dataset, timeout=10000)
    assert widget.gui.moltrack_channel_selector.currentText() == "image"

    importer.tool_selector.setCurrentIndex(1)
    widget.gui.locs_import_mode.setCurrentText("CSV")
    widget.gui.import_localisations.click()
    assert len(widget.session.query_localizations(dataset, "image")) == 6

    tracking = widget.open_tool_window("Tracking")
    widget.gui.tracking_dataset.setCurrentText(dataset)
    widget.gui.tracking_channel.setCurrentText("image")
    widget.gui.link_localisations.click()
    assert widget.threadpool.has_workers()
    tracking.close()
    qtbot.waitUntil(lambda: not widget.threadpool.has_workers(), timeout=45000)
    tracks = widget.session.query_tracks(dataset, "image")
    assert len(tracks) == 6
    assert tracks.track_id.nunique() == 1
    assert not tracking.isVisible()
    with qtbot.waitSignals([tracking.destroyed, importer.destroyed]):
        viewer.close()


def test_tool_window_scrolls_when_short(launcher, qtbot):
    _, widget, _ = launcher
    window = widget.open_tool_window("Segmentation")
    window.resize(300, 180)
    scroll = window.findChild(QScrollArea)
    qtbot.waitUntil(lambda: scroll.verticalScrollBar().maximum() > 0)
    scroll.ensureWidgetVisible(widget.gui.segment_active)
    assert scroll.verticalScrollBar().value() > 0
    assert window.tool_selector.isVisible()
