"""BactFit 0.1.6 fitting with isolated NumPy 2 and framed-shape adapters.

The upstream batch API counts unfitted cells as successes and crashes while
summarising an empty fit-error list. Use its per-cell algorithm and validate
results before constructing a CellList instead.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import FunctionType, SimpleNamespace

import numpy as np
from bactfit.cell import CellList
from bactfit.fit import bactfit
from bactfit.preprocess import data_to_cells
from bactfit.utils import fit_poly, manual_fit as _manual_fit

from .model_processing import _cancel
from .project_geometry import outline_fingerprint


def _render_cell_locs(locs, *, viewport, oversampling, blur_method,
                      min_blur_width, ang=0):
    """Translate BactFit's legacy render call to Picasso 0.11's pixel-size API."""
    import pandas as pd
    from napari.utils.notifications import show_info
    from picasso.render import render

    rows = pd.DataFrame(locs).copy(deep=True)
    if blur_method in ("convolve", "gaussian", "gaussian_iso"):
        (y0, x0), (y1, x1) = viewport
        in_view = (rows.x > x0) & (rows.x < x1) & (rows.y > y0) & (rows.y < y1)
        precision = rows.reindex(columns=["lpx", "lpy"])
        valid = np.isfinite(precision) & (precision > 0)
        available = precision[in_view].where(valid[in_view]).median()
        if blur_method == "convolve" and available.notna().all():
            # A global kernel needs one width per axis. Missing measurements
            # must not poison its median or discard otherwise valid positions.
            rows["lpx"], rows["lpy"] = available["lpx"], available["lpy"]
            if not valid[in_view].to_numpy().all():
                show_info("Global blur estimated from available valid localisation precisions.")
        elif not {"lpx", "lpy"} <= set(rows) or not valid[in_view].to_numpy().all():
            # Individual uncertainty cannot be recovered from other detections.
            # Retain every position with an explicit, fixed-width render instead.
            blur_method = "smooth"
            show_info("Localisation precision is missing or invalid; using one-pixel blur for this cell render.")

    return render(
        rows, {"Pixelsize": 1.0},
        disp_px_size=1.0 / oversampling, viewport=viewport,
        blur_method=blur_method, min_blur_width=min_blur_width,
        ang=None if np.isscalar(ang) and ang == 0 else ang,
    )


def plot_cell_render(celllist, **settings):
    """Use BactFit's plot layout without changing its module's render function."""
    original = CellList.plot_render
    plot = FunctionType(
        original.__code__, {**original.__globals__, "render": _render_cell_locs},
        argdefs=original.__defaults__, closure=original.__closure__,
    )
    image = plot(celllist, **settings)
    if image is None or np.ndim(image) < 2:
        raise ValueError("Cell rendering did not produce an image")
    return image


def _numpy2_fit_poly():
    # BactFit's polynomial helper references np.RankWarning, removed in NumPy 2.
    # Copy only its function globals; never patch NumPy or BactFit globally.
    polynomial = fit_poly
    if not hasattr(np, "RankWarning"):
        numpy_api = SimpleNamespace(**{**vars(np), "RankWarning": np.exceptions.RankWarning})
        polynomial = FunctionType(
            fit_poly.__code__, {**fit_poly.__globals__, "np": numpy_api},
            argdefs=fit_poly.__defaults__, closure=fit_poly.__closure__,
        )
    return polynomial


def manual_fit(cell_coords, midline_coords, radius=None):
    """Fit a manual cell with the same isolated NumPy adapter as batch fits."""
    original = _manual_fit
    fit = FunctionType(
        original.__code__, {**original.__globals__, "fit_poly": _numpy2_fit_poly()},
        argdefs=original.__defaults__, closure=original.__closure__,
    )
    return fit(cell_coords, midline_coords, radius)


def _numpy2_fit_cell():
    original = bactfit.fit_cell
    return FunctionType(
        original.__code__, {**original.__globals__, "fit_poly": _numpy2_fit_poly()},
        argdefs=original.__defaults__, closure=original.__closure__,
    )


def _cells_from_shapes(shapes):
    cells = []
    for shape in shapes:
        shape = np.asarray(shape)
        source_outline = outline_fingerprint(shape)
        if shape.ndim != 2 or shape.shape[1] not in (2, 3):
            raise ValueError("Cell outlines must contain 2D coordinates with an optional frame column")
        if len(shape) < 3:
            continue
        frame = -1
        if shape.shape[1] == 3:
            frame = shape[0, 0]
            if not np.isfinite(frame) or frame != int(frame) or not np.all(shape[:, 0] == frame):
                raise ValueError("A cell outline must belong to a single integer frame")
            shape = shape[:, 1:]
        # Upstream's 3-column importer slices rows instead of the frame column.
        # Feed it spatial coordinates, then restore frame provenance explicitly.
        imported = data_to_cells([shape])
        for cell in getattr(imported, "data", []):
            cell.frame_index = int(frame)
            cell._moltrack_source_outline = source_outline
            cells.append(cell)
    return cells


def _fit_cell(cell, settings):
    fitted = _numpy2_fit_cell()(cell, **settings)
    error = getattr(fitted, "fit_error", None)
    if error is None or not np.isfinite(error):
        return None
    return fitted


def fit_segmentations(shapes, settings, *, progress_callback=None,
                      cancel_event=None, max_workers=None):
    _cancel(cancel_event)
    cells = _cells_from_shapes(shapes)
    if not cells:
        return None
    workers = min(len(cells), max_workers or os.cpu_count() or 1)
    results = {}

    def record(index, cell):
        _cancel(cancel_event)
        results[index] = cell
        if progress_callback is not None:
            progress_callback.emit(int(100 * len(results) / len(cells)))

    if workers == 1:
        for index, cell in enumerate(cells):
            _cancel(cancel_event)
            record(index, _fit_cell(cell, settings))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_fit_cell, cell, settings): index
                       for index, cell in enumerate(cells)}
            try:
                for future in as_completed(futures):
                    record(futures[future], future.result())
            finally:
                for future in futures:
                    future.cancel()
    _cancel(cancel_event)
    fitted = [results[index] for index in range(len(cells)) if results[index] is not None]
    if not fitted:
        raise ValueError("No cells could be fitted. Check the outlines, radius limits and maximum fit error.")
    result = CellList(fitted)
    result.assign_cell_indices(reindex=True)
    return result
