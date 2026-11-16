from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from test_model_processing import make_session


@pytest.mark.parametrize("selected", ["Segmentations", "Cells"])
def test_layer_changes_preserve_tracking_segmentation(qtbot, selected):
    from qtpy.QtWidgets import QComboBox
    from moltrack.funcs.events_utils import _events_utils

    combos = {}
    for name in ("shapes_export_data", "remove_seglocs_segmentation",
                 "segtracks_seg", "tracking_segmentations"):
        combo = QComboBox()
        qtbot.addWidget(combo)
        combos[name] = combo
    host = _events_utils()
    host.gui = SimpleNamespace(**combos)
    host.viewer = SimpleNamespace(layers=[
        SimpleNamespace(name="Segmentations"), SimpleNamespace(name="Cells")])
    host.update_segmentation_combos()
    host.gui.tracking_segmentations.setCurrentText(selected)
    host.viewer.layers.append(SimpleNamespace(name="Tracks"))
    host.update_segmentation_combos()
    assert host.gui.tracking_segmentations.currentText() == selected


def test_segmented_rerun_clears_previous_outside_tracks():
    from moltrack.model import processing as proc

    session = make_session()
    ids = session.replace_localizations("a", "image", pd.DataFrame({
        "frame": [0, 1, 0, 1, 0, 1],
        "x": [3., 3., 12., 12., 1., 1.],
        "y": [3., 3., 12., 12., 3., 3.],
    }))
    def run(settings):
        req = proc.ProcessingRequest("tracking", (("a", "image"),), settings)
        proc.publish(session, proc.compute(proc.prepare(session, req)))

    run(proc.TrackingSettings(search_range=1, min_length=2))
    assert len(session.query_tracks("a", "image")) == 6
    run(proc.TrackingSettings(search_range=1, min_length=2,
        segmentation_column="segmentation_index",
        polygons=(((1., 1.), (5., 1.), (5., 5.), (1., 5.)),)))
    assert set(session.query_tracks("a", "image").id) == set(ids[:2])
    np.testing.assert_array_equal(session.query_localizations("a", "image").id, ids)


@pytest.mark.parametrize(
    ("allowance", "expected_labels"),
    [(0, {2: 1}), (1.5, {0: 1, 1: 0, 2: 1, 3: 0, 4: 1})],
)
def test_tracking_boundary_allowance(allowance, expected_labels):
    from moltrack.model import processing as proc

    session = make_session()
    # Nearest to B, tied, inside B, on A, exactly at the limit, past the limit.
    x = [5.75, 5.5, 6.25, 1.0, 12.5, 12.51]
    ids = session.replace_localizations(
        "a", "image",
        pd.DataFrame({
            "frame": np.tile([0, 1], len(x)),
            "x": np.repeat(x, 2),
            "y": np.repeat(np.arange(2, 2 + len(x)), 2),
        }),
    )
    polygons = (
        ((1, 1), (5, 1), (5, 8), (1, 8)),
        ((6, 1), (11, 1), (11, 8), (6, 8)),
    )
    settings = proc.TrackingSettings(
        search_range=0.5,
        min_length=2,
        segmentation_column="cell_index",
        polygons=polygons,
        boundary_allowance=allowance,
    )
    request = proc.ProcessingRequest("tracking", (("a", "image"),), settings)
    proc.publish(session, proc.compute(proc.prepare(session, request)))

    tracks = session.query_tracks("a", "image")
    assert dict(zip(tracks.id, tracks.cell_index, strict=True)) == {
        lid: label
        for pair, label in expected_labels.items()
        for lid in ids[2 * pair : 2 * pair + 2]
    }
    assert sorted(tracks.groupby("track_id").size()) == [2] * len(expected_labels)


def test_tracking_boundary_allowance_control_defaults_to_zero(qtbot):
    from qtpy.QtWidgets import QWidget

    from moltrack.GUI.widget_ui import Ui_Frame

    frame = QWidget()
    qtbot.addWidget(frame)
    gui = Ui_Frame()
    gui.setupUi(frame)

    assert gui.tracking_boundary_allowance.value() == 0
    gui.tracking_boundary_allowance.setValue(1.5)
    assert gui.tracking_boundary_allowance.value() == 1.5
