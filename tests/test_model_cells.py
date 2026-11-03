from types import SimpleNamespace as NS

import numpy as np
import pytest
from molcore import StaleResultError
from test_model_mutation_gui import control
from test_session_mutations import make_session


def test_cell_worker_result_rejects_shape_edits():
    from moltrack.model_cells import (
        finish_cells,
        prepare_cells,
        validate_cells,
    )

    s = make_session()
    prepared = prepare_cells(s, {"geometry": [1, 2, 3]})
    pending = finish_cells(prepared, ["detached cell"])
    s.set_annotations("cells", {"data": [], "properties": {}})
    with pytest.raises(StaleResultError):
        validate_cells(s, pending)


def test_bactfit_launcher_captures_geometry_and_settings(monkeypatch):
    import moltrack.funcs.bactfit_utils as module

    calls = []

    class Worker:
        def __init__(self, fn, *args):
            self.fn, self.args = fn, args
            self.signals = NS(
                **{
                    name: NS(connect=lambda *args: None)
                    for name in ("progress", "result", "finished", "error")
                }
            )
            calls.append(self)

    monkeypatch.setattr(module, "Worker", Worker)
    h = module._bactfit_utils()
    h.session = make_session()
    from threading import Event

    h.stop_event = Event()
    polygon = np.array([[1.0, 1.0], [1.0, 5.0], [5.0, 5.0], [5.0, 1.0]])
    h.session.set_annotations("segmentations", {"data": [polygon]})
    h.segLayer = NS(data=[polygon])
    h.gui = NS(
        fit_min_radius=control(1),
        fit_max_radius=control(4),
        fit_max_error=control(2),
        bactfit_progressbar=None,
    )
    h.update_ui = lambda **kw: None
    h.moltrack_progress = lambda **kw: None
    h.threadpool = NS(start=lambda worker: None)
    h.initialise_bactfit()
    assert calls[0].fn is module.fit_cells_worker
    assert calls[0].args[0].cancel_event is h.stop_event
    expected = polygon.copy()
    h.segLayer.data[0][:] = 99
    np.testing.assert_array_equal(
        calls[0].args[0].payload["segmentations"][0], expected
    )
    assert calls[0].args[0].payload["settings"] == {
        "min_radius": 1.0,
        "max_radius": 4.0,
        "max_error": 2.0,
    }


def test_heatmap_stale_cache_clears_projected_data():
    from molcore.staging import state_token

    from moltrack.funcs.cell_heatmap_utils import _cell_heatmap_utils

    h = _cell_heatmap_utils()
    h.session = make_session()
    h.celllist = object()
    h.heatmap_image = np.ones((2, 2))
    h.heatmap_locs = np.ones(3)
    h._heatmap_token = state_token(h.session.model)
    h.heatmap_canvas = NS(clear=lambda: None)
    h.session.update_metadata("data", pixel_size=200)
    assert h._heatmap_is_current() is False
    assert h.heatmap_image is None
    assert h.heatmap_locs is None
