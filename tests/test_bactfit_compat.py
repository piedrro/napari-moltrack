from threading import Event

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon


def rod():
    points = np.asarray(LineString([(10, 10), (30, 10)]).buffer(3).exterior.coords).copy()
    points += np.random.default_rng(0).normal(0, 0.1, points.shape)
    points[-1] = points[0]
    return points


def test_numpy2_fit_uses_real_algorithm_without_mutating_numpy():
    from moltrack.bactfit_compat import _cells_from_shapes, _fit_cell

    had_rank_warning = hasattr(np, "RankWarning")
    cell = _cells_from_shapes([rod()])[0]
    fitted = _fit_cell(cell, {"max_error": 5})
    assert fitted is not None
    assert np.isfinite(fitted.fit_error)
    assert fitted.cell_midline.length > 10
    assert hasattr(np, "RankWarning") == had_rank_warning


def test_framed_polygon_keeps_all_vertices_and_frame():
    from moltrack.bactfit_compat import _cells_from_shapes

    points = rod()
    framed = np.column_stack((np.full(len(points), 7), points))
    cell = _cells_from_shapes([framed])[0]
    assert cell.frame_index == 7
    assert cell.cell_polygon.equals_exact(Polygon(points), 0)
    assert cell.cell_polygon.area > 100


def test_fitting_in_spawned_processes_preserves_frames():
    from moltrack.bactfit_compat import fit_segmentations

    points = rod()
    framed = np.column_stack((np.full(len(points), 7), points))
    progress = []
    from types import SimpleNamespace

    result = fit_segmentations(
        [points, framed], {"max_error": 5}, max_workers=2,
        progress_callback=SimpleNamespace(emit=progress.append),
    )
    assert [cell.frame_index for cell in result.data] == [-1, 7]
    assert [cell.cell_index for cell in result.data] == [0, 1]
    assert all(np.isfinite(cell.fit_error) for cell in result.data)
    assert progress[-1] == 100


def test_all_failed_fits_report_failure_instead_of_returning_unfitted_cells():
    from moltrack.bactfit_compat import fit_segmentations

    with pytest.raises(ValueError, match="No cells could be fitted"):
        fit_segmentations([rod()], {"max_error": 0.000001}, max_workers=1)


def test_empty_shapes_and_cancellation():
    from concurrent.futures import CancelledError

    from moltrack.bactfit_compat import fit_segmentations

    assert fit_segmentations([], {}) is None
    cancelled = Event()
    cancelled.set()
    with pytest.raises(CancelledError):
        fit_segmentations([rod()], {}, cancel_event=cancelled)
