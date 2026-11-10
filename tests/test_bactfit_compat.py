from threading import Event

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon


def rod():
    points = np.asarray(LineString([(10, 10), (30, 10)]).buffer(3).exterior.coords).copy()
    points += np.random.default_rng(0).normal(0, 0.1, points.shape)
    points[-1] = points[0]
    return points


@pytest.fixture
def cell_render_host():
    from types import SimpleNamespace as NS

    import pandas as pd
    from bactfit.cell import CellList

    from moltrack.bactfit_compat import _cells_from_shapes
    from moltrack.funcs.cell_heatmap_utils import _cell_heatmap_utils

    cell = _cells_from_shapes([rod()])[0]
    cell.transformed = True
    cell.locs = pd.DataFrame({
        "frame": [0, 1], "x": [12.0, 15.0], "y": [10.0, 10.0],
        "lpx": [0.2, 0.3], "lpy": [0.3, 0.2],
    }).to_records(index=False)
    celllist = CellList([cell])
    celllist.filter_by_length = lambda *args: celllist
    original_render = CellList.plot_render.__globals__["render"]
    host = _cell_heatmap_utils()
    host.celllist = celllist
    host._heatmap_is_current = lambda: True
    host.update_ui = lambda **kwargs: None
    shown = []
    host.show_heatmap = shown.append
    host.heatmap_canvas = NS(clear=lambda: None)

    def control(value):
        return NS(currentText=lambda: value, value=lambda: value, isChecked=lambda: value)

    host.gui = NS(**{name: control(value) for name, value in {
        "heatmap_dataset": "All Datasets", "heatmap_channel": "All Channels",
        "heatmap_mode": "Render", "heatmap_colourmap": "jet",
        "render_draw_outline": True, "heatmap_min_length": 0,
        "heatmap_max_length": 100, "heatmap_min_msd": 0,
        "heatmap_max_msd": 100, "render_symmetry": False,
    }.items()})
    host.heatmap_binning = control(10)
    host.heatmap_blur_method = control("One-Pixel-Blur")
    host.heatmap_min_blur_width = control(0.2)
    host.heatmap_oversampling = control(2)
    return host, shown, original_render


@pytest.mark.parametrize("blur_method", [
    "One-Pixel-Blur", "Global Localisation Precision",
    "Individual Localisation Precision, iso", "Individual Localisation Precision",
])
def test_cell_heatmap_render_uses_current_picasso_api(cell_render_host, blur_method):
    from bactfit.cell import CellList

    host, shown, original_render = cell_render_host
    host.heatmap_blur_method.currentText = lambda: blur_method

    host.plot_heatmap()

    assert len(shown) == 1
    assert shown[0] is not None
    assert shown[0].ndim == 3
    assert np.isfinite(shown[0]).all()
    assert CellList.plot_render.__globals__["render"] is original_render


def test_failed_cell_render_is_not_displayed_or_exported(cell_render_host, monkeypatch):
    import moltrack.bactfit_compat as compat
    import moltrack.funcs.cell_heatmap_utils as heatmap

    host, shown, _ = cell_render_host
    notices, updates = [], []
    host.update_ui = lambda **kwargs: updates.append(kwargs)
    host.heatmap_image = np.ones((2, 2))

    def fail(*args, **kwargs):
        raise ValueError("Rendering failed")

    monkeypatch.setattr(compat, "_render_cell_locs", fail)
    monkeypatch.setattr(heatmap, "show_info", notices.append)
    host.plot_heatmap()

    assert shown == []
    assert host.heatmap_image is None
    assert updates[-1] == {}
    assert "did not produce an image" in notices[-1]


@pytest.mark.parametrize("oversampling", [1, 2.5])
def test_cell_render_adapter_preserves_viewport_and_sampling(oversampling):
    import pandas as pd

    from moltrack.bactfit_compat import _render_cell_locs

    locs = pd.DataFrame({"frame": [0], "x": [2.0], "y": [3.0]}).to_records(index=False)
    count, image = _render_cell_locs(
        locs, viewport=((0, 0), (8, 12)), oversampling=oversampling,
        blur_method=None, min_blur_width=0.2,
    )
    assert count == 1
    assert image.shape == (int(8 * oversampling), int(12 * oversampling))
    assert image.sum() == pytest.approx(1)


def test_global_cell_blur_uses_finite_precision_without_dropping_positions():
    import pandas as pd
    from picasso.render import render

    from moltrack.bactfit_compat import _render_cell_locs

    locs = pd.DataFrame({
        "frame": [0, 1, 2], "x": [2., 4., 6.], "y": [3., 4., 5.],
        "lpx": [0.2, np.nan, 0.4], "lpy": [0.3, 0.5, np.inf],
    }).to_records(index=False)
    original = locs.copy()
    count, image = _render_cell_locs(
        locs, viewport=((0, 0), (8, 12)), oversampling=2,
        blur_method="convolve", min_blur_width=0.2,
    )
    expected_rows = pd.DataFrame(locs).assign(lpx=0.3, lpy=0.4)
    _, expected = render(
        expected_rows, {"Pixelsize": 1.0}, disp_px_size=0.5,
        viewport=((0, 0), (8, 12)), blur_method="convolve", min_blur_width=0.2,
    )
    assert count == 3
    assert np.isfinite(image).all()
    np.testing.assert_allclose(image, expected, atol=1e-7)
    pd.testing.assert_frame_equal(pd.DataFrame(locs), pd.DataFrame(original))


@pytest.mark.parametrize("method", ["convolve", "gaussian", "gaussian_iso"])
@pytest.mark.parametrize("precision", [None, np.nan, np.inf, 0., -1.])
def test_cell_blur_falls_back_with_notice_when_precision_unavailable(
    method, precision, monkeypatch,
):
    import pandas as pd
    from napari.utils import notifications
    from picasso.render import render

    from moltrack.bactfit_compat import _render_cell_locs

    rows = pd.DataFrame({"frame": [0, 1], "x": [2., 4.], "y": [3., 4.]})
    if precision is not None:
        rows["lpx"], rows["lpy"] = precision, precision
    locs = rows.to_records(index=False)
    notices = []
    monkeypatch.setattr(notifications, "show_info", notices.append)
    count, image = _render_cell_locs(
        locs, viewport=((0, 0), (8, 12)), oversampling=2,
        blur_method=method, min_blur_width=0.2,
    )
    _, expected = render(
        rows, {"Pixelsize": 1.0}, disp_px_size=0.5,
        viewport=((0, 0), (8, 12)), blur_method="smooth", min_blur_width=0.2,
    )
    assert count == 2
    assert np.isfinite(image).all()
    np.testing.assert_allclose(image, expected)
    assert notices and "one-pixel blur" in notices[-1].lower()


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
