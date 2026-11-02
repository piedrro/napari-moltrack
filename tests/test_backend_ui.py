"""Exercise the unchanged widgets against current napari and Qt6."""

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np


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
        assert widget.gui.picasso_detect.text()
        assert widget.gui.smlm_fit_mode.itemText(0) == "Picasso"
        assert widget.gui.undrift_dataset is not None
        assert [
            widget.gui.cellpose_model.itemText(i)
            for i in range(widget.gui.cellpose_model.count())
        ] == ["cpsam_v2", "cpsam", "cpdino", "cpdino-vitb", "custom"]
    finally:
        viewer.close()


def _localizer():
    from moltrack.funcs.picasso_detect_utils import _picasso_detect_utils

    host = _picasso_detect_utils()
    host.dataset_dict = {"a": {"images": {"image": np.zeros((2, 15, 15))}}}
    old = np.rec.fromarrays(
        [[0, 1], [5.0, 6.0], [5.0, 6.0]], names="frame,x,y"
    )
    host.localisation_dict = {
        "a": {"image": {"localisations": old, "fitted": False}}
    }
    host.stop_event = threading.Event()
    host.generate_roi = dict

    def value(number):
        return SimpleNamespace(value=lambda: number)

    host.gui = SimpleNamespace(
        picasso_box_size=value(7),
        smlm_threshold=value(100),
        picasso_remove_overlapping=SimpleNamespace(isChecked=lambda: False),
        moltrack_kernel_size=SimpleNamespace(text=lambda: "3"),
    )
    return host


def test_preview_empty_result_only_clears_selected_frame(monkeypatch):
    import moltrack.funcs.picasso_detect_utils as module

    host = _localizer()
    empty = np.recarray(0, dtype=[("frame", "i8"), ("x", "f8"), ("y", "f8")])
    monkeypatch.setattr(module, "localize_movie", lambda *a, **k: empty)
    host._picasso_wrapper(None, True, False, ["a"], ["image"], frame_index=1)
    result = host.localisation_dict["a"]["image"]["localisations"]
    assert result.frame.tolist() == [0]


def test_full_empty_result_replaces_old_localizations(monkeypatch):
    import moltrack.funcs.picasso_detect_utils as module

    host = _localizer()
    empty = np.recarray(0, dtype=[("frame", "i8"), ("x", "f8"), ("y", "f8")])
    monkeypatch.setattr(module, "localize_movie", lambda *a, **k: empty)
    host._picasso_wrapper(None, True, False, ["a"], ["image"])
    assert len(host.localisation_dict["a"]["image"]["localisations"]) == 0


def test_failed_partition_does_not_commit_previous_partitions(monkeypatch):
    import moltrack.funcs.picasso_detect_utils as module

    host = _localizer()
    original = host.localisation_dict["a"]["image"]["localisations"].copy()
    host.dataset_dict["a"]["images"]["other"] = np.zeros((2, 15, 15))
    calls = []

    def compute(*args, **kwargs):
        if calls:
            raise ValueError("failed partition")
        calls.append(1)
        return original[:1]

    monkeypatch.setattr(module, "localize_movie", compute)
    import pytest

    with pytest.raises(ValueError, match="failed partition"):
        host._picasso_wrapper(None, True, False, ["a"], ["image", "other"])
    np.testing.assert_array_equal(
        host.localisation_dict["a"]["image"]["localisations"], original
    )


def test_cellpose_gui_returns_molcore_masks(monkeypatch):
    from moltrack.funcs.segmentation_utils import _segmentation_utils
    import moltrack.backend as backend

    host = _segmentation_utils()
    host.stop_event = threading.Event()
    text = lambda value: SimpleNamespace(text=lambda: str(value))
    host.gui = SimpleNamespace(
        cellpose_model=SimpleNamespace(currentText=lambda: 'cpsam_v2'),
        cellpose_diameter_label=text(15), cellpose_flowthresh_label=text(.4),
        cellpose_maskthresh_label=text(0), cellpose_minsize_label=text(15),
        cellpose_invert_images=SimpleNamespace(isChecked=lambda: False))
    mask = np.ones((5, 6), dtype=np.uint16)
    monkeypatch.setattr(backend, 'segment_images', lambda *a, **k: [mask])
    mode, masks = host.pixseq_segment([np.zeros((5, 6))])
    assert mode == 'active'
    np.testing.assert_array_equal(masks[0], mask)


def test_tracking_short_segmentation_does_not_discard_valid_group():
    from moltrack.funcs.tracking_utils import _tracking_utils
    import pandas as pd

    host = _tracking_utils()
    host.gui = SimpleNamespace(
        tracking_segmentations=SimpleNamespace(currentText=lambda: 'Segmentations'),
        trackpy_search_range=SimpleNamespace(value=lambda: 1.),
        trackpy_memory=SimpleNamespace(value=lambda: 0),
        min_track_length=SimpleNamespace(value=lambda: 3))
    host.dataset_dict = {'a': {'pixel_size': 100., 'exposure_time': 50.}}
    rows = pd.DataFrame({'frame': [0, 0, 1, 2], 'x': [1., 5., 5., 5.],
        'y': [1., 5., 5., 5.], 'segmentation_index': [1, 2, 2, 2],
        'dataset': ['a'] * 4, 'channel': ['image'] * 4})
    host.detect_seglocs = lambda *a: (rows.to_records(index=False), 'segmentation_index')

    def link(data, **kwargs):
        result = data.copy() if len(data) >= kwargs.get('min_length', 3) else data.iloc[:0].copy()
        result['particle'] = np.ones(len(result), dtype=int)
        return result

    host.link_locs = link
    result = host.detect_tracks([{'dataset': 'a', 'channel': 'image',
                                  'localisations': rows.to_records(index=False)}])
    assert len(result) == 3


def test_preview_populates_partition_metadata(monkeypatch):
    import moltrack.funcs.picasso_detect_utils as module
    import pandas as pd
    host = _localizer()
    old = pd.DataFrame(host.localisation_dict['a']['image']['localisations'])
    old['dataset'], old['channel'] = 'a', 'image'
    host.localisation_dict['a']['image']['localisations'] = old.to_records(index=False)
    new = np.rec.fromarrays([[1], [9.], [8.]], names='frame,x,y')
    monkeypatch.setattr(module, 'localize_movie', lambda *a, **k: new)
    host._picasso_wrapper(None, True, False, ['a'], ['image'], frame_index=1)
    result = host.localisation_dict['a']['image']['localisations']
    assert list(result.dataset) == ['a', 'a']
    assert list(result.channel) == ['image', 'image']


def test_cancelled_tracking_batch_does_not_return_partial_results():
    from concurrent.futures import CancelledError
    from moltrack.funcs.tracking_utils import _tracking_utils
    import pandas as pd
    import pytest

    host = _tracking_utils()
    host.stop_event = threading.Event()
    host.gui = SimpleNamespace(
        tracking_segmentations=SimpleNamespace(currentText=lambda: ''),
        trackpy_search_range=SimpleNamespace(value=lambda: 1.),
        trackpy_memory=SimpleNamespace(value=lambda: 0),
        min_track_length=SimpleNamespace(value=lambda: 1))
    host.dataset_dict = {'a': {'pixel_size': 100., 'exposure_time': 50.}}
    rows = pd.DataFrame({'frame': [0], 'x': [5.], 'y': [5.]})

    def link(data, **kwargs):
        host.stop_event.set()
        return data.assign(particle=1)

    host.link_locs = link
    jobs = [{'dataset': 'a', 'channel': channel, 'localisations': rows}
            for channel in ['first', 'second']]
    with pytest.raises(CancelledError):
        host.detect_tracks(jobs)


def test_empty_tracking_result_clears_previous_partition():
    from moltrack.funcs.tracking_utils import _tracking_utils
    import pandas as pd

    host = _tracking_utils()
    host.stop_event = threading.Event()
    host._tracking_partitions = [('a', 'image')]
    host.tracking_dict = {'a': {'image': {'tracks': [1, 2, 3]}}}
    host.gui = SimpleNamespace(remove_unlinked=SimpleNamespace(isChecked=lambda: False))
    host.process_tracking_results(pd.DataFrame(columns=['dataset', 'channel', 'particle', 'frame', 'x', 'y']))
    assert len(host.tracking_dict['a']['image']['tracks']) == 0


def test_cancelled_drift_commit_is_discarded():
    from moltrack.funcs.undrift_utils import MoltrackUndrifting
    event = threading.Event()
    event.set()
    parent = SimpleNamespace(stop_event=event, dataset_dict={})
    tools = MoltrackUndrifting(None, parent)
    tools._apply([('removed', {}, {}, {}, {}, None)])
    assert parent.dataset_dict == {}
