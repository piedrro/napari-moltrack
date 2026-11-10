from types import SimpleNamespace

import pandas as pd
from qtpy.QtWidgets import QLabel, QSlider


def test_cellpose_sliders_update_labels_and_processing_settings(qtbot):
    import napari

    from moltrack._widget import MolTrack

    viewer = napari.Viewer(show=False)
    try:
        host = MolTrack(viewer, load_dev_data=False)
        qtbot.addWidget(host)
        for name, value, expected in (
            ("flowthresh", 34, "0.34"),
            ("maskthresh", 67, "0.67"),
            ("minsize", 123, "123"),
            ("diameter", 42, "42"),
        ):
            getattr(host.gui, f"cellpose_{name}").setValue(value)
            assert getattr(host.gui, f"cellpose_{name}_label").text() == expected

        # Labels are a display projection, not the processing settings source.
        host.gui.cellpose_flowthresh_label.setText("0.9")
        settings = host._capture_cellpose_settings()
        assert settings.flow_threshold == 0.34
        assert settings.cellprob_threshold == 0.67
        assert settings.min_size == 123
        assert settings.diameter == 42
        host.gui.cellpose_diameter.setValue(0)
        assert host.gui.cellpose_diameter_label.text() == "0"
        assert host._capture_cellpose_settings().diameter is None
    finally:
        viewer.close()


def test_track_slider_refreshes_display_after_range_clamp_and_empty_data(qtbot):
    from moltrack.funcs.trackplot_utils import _trackplot_utils

    host = _trackplot_utils()
    slider, label = QSlider(), QLabel("stale")
    qtbot.addWidget(slider)
    qtbot.addWidget(label)
    slider.setRange(0, 5)
    slider.setValue(3)
    host.gui = SimpleNamespace(trackplot_slider=slider, trackplot_slider_label=label)
    host.trackplot_tracks = pd.DataFrame({"track_id": [10], "particle": [42]})
    host.init_trackplot_tracks = lambda **kwargs: host.trackplot_tracks
    host.initialise_trackplot_slider()
    assert slider.value() == 0
    assert label.text() == "42"

    host.trackplot_tracks = host.trackplot_tracks.iloc[:0]
    host.initialise_trackplot_slider()
    assert slider.minimum() == slider.maximum() == slider.value() == 0
    assert label.text() == "0"
