"""Exercise the unchanged widgets against current napari and Qt6."""

from pathlib import Path


def test_manifest_resolves_moltrack_widget():
    from npe2 import PluginManifest
    from npe2._command_registry import CommandRegistry

    import moltrack

    manifest = PluginManifest.from_file(
        str(Path(moltrack.__file__).with_name("napari.yaml"))
    )
    registry = CommandRegistry()
    registry.register_manifest(manifest)
    from moltrack._widget import MolTrack

    assert registry.get(manifest.contributions.widgets[0].command) is MolTrack


def test_widget_initializes_with_current_napari(qtbot, monkeypatch):
    import napari

    from moltrack._widget import MolTrack

    monkeypatch.setattr(MolTrack, "import_dev_data", lambda self: None)
    viewer = napari.Viewer(show=False)
    try:
        widget = MolTrack(viewer)
        qtbot.addWidget(widget)
        for alias in ("dataset_dict", "localisation_dict", "tracking_dict"):
            assert not hasattr(widget, alias)
        assert widget.gui.picasso_detect.text()
        assert widget.gui.smlm_fit_mode.itemText(0) == "Picasso"
        assert widget.gui.undrift_dataset is not None
        assert [
            widget.gui.cellpose_model.itemText(i)
            for i in range(widget.gui.cellpose_model.count())
        ] == ["cyto3", "cyto2", "cyto", "nuclei", "custom"]
    finally:
        viewer.close()


def test_preview_populates_partition_metadata(monkeypatch):
    import pandas as pd
    from test_model_processing import make_session

    from moltrack import model_processing as proc
    from moltrack.model_views import gui_query

    session = make_session()
    monkeypatch.setattr(proc, "_detect_partition", lambda *a: pd.DataFrame(
        {"frame": [1], "x": [9.], "y": [8.]}
    ))
    request = proc.ProcessingRequest("detect", (("a", "image"),),
                                     proc.DetectionSettings(frame_index=1))
    proc.publish(session, proc.compute(proc.prepare(session, request)))
    rows = gui_query(session, "a", "image")
    assert rows.dataset.tolist() == ["a", "a"]
    assert rows.channel.tolist() == ["image", "image"]
    assert rows.frame.tolist() == [0, 1]


def test_full_empty_result_replaces_old_localizations(monkeypatch):
    import pandas as pd
    from test_model_processing import make_session

    from moltrack import model_processing as proc

    session = make_session()
    monkeypatch.setattr(proc, "_detect_partition", lambda *args:
                        pd.DataFrame(columns=["frame", "x", "y"]))
    request = proc.ProcessingRequest("detect", (("a", "image"),), proc.DetectionSettings())
    proc.publish(session, proc.compute(proc.prepare(session, request)))
    assert session.query_localizations("a", "image").empty


def test_tracking_short_segmentation_does_not_discard_valid_group():
    import pandas as pd
    from test_model_processing import make_session

    from moltrack import model_processing as proc

    session = make_session()
    session.replace_localizations("a", "image", pd.DataFrame({
        "frame": [0, 0, 1], "x": [3., 10., 10.], "y": [3., 10., 10.]
    }))
    polygons = (
        ((1., 1.), (5., 1.), (5., 5.), (1., 5.)),
        ((8., 8.), (12., 8.), (12., 12.), (8., 12.)),
    )
    request = proc.ProcessingRequest("tracking", (("a", "image"),),
        proc.TrackingSettings(search_range=1, min_length=2, polygons=polygons,
                              segmentation_column="segmentation_index"))
    proc.publish(session, proc.compute(proc.prepare(session, request)))
    assert len(session.query_tracks("a", "image")) == 2
    assert len(session.query_localizations("a", "image")) == 3


def test_cancelled_tracking_batch_does_not_return_partial_results(monkeypatch):
    import threading
    from concurrent.futures import CancelledError

    import pytest
    from test_model_processing import make_session

    from moltrack import model_processing as proc

    session = make_session()
    session.bind("a", "second", "camera", "image")
    event = threading.Event()
    original = proc._track_partition

    def track(*args, **kwargs):
        result = original(*args, **kwargs)
        event.set()
        return result

    monkeypatch.setattr(proc, "_track_partition", track)
    request = proc.ProcessingRequest("tracking", (("a", "image"), ("a", "second")),
                                     proc.TrackingSettings())
    with pytest.raises(CancelledError):
        proc.compute(proc.prepare(session, request), cancel_event=event)
    assert session.query_tracks("a", "image").empty
    assert len(session.query_localizations("a", "image")) == 2


def test_empty_tracking_result_clears_previous_partition():
    from test_model_processing import make_session

    from moltrack import model_processing as proc

    session = make_session()
    for minimum in (1, 3):
        request = proc.ProcessingRequest("tracking", (("a", "image"),),
            proc.TrackingSettings(search_range=2., min_length=minimum))
        proc.publish(session, proc.compute(proc.prepare(session, request)))
        assert len(session.query_tracks("a", "image")) == (2 if minimum == 1 else 0)
    assert len(session.query_localizations("a", "image")) == 2
