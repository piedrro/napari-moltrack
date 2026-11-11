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
    from moltrack import model_processing as proc

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
