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
from bactfit.utils import fit_poly

from .model_processing import _cancel


def _numpy2_fit_cell():
    # BactFit's polynomial helper references np.RankWarning, removed in NumPy 2.
    # Copy only its function globals; never patch NumPy or BactFit globally.
    polynomial = fit_poly
    if not hasattr(np, "RankWarning"):
        numpy_api = SimpleNamespace(**{**vars(np), "RankWarning": np.exceptions.RankWarning})
        polynomial = FunctionType(
            fit_poly.__code__, {**fit_poly.__globals__, "np": numpy_api},
            argdefs=fit_poly.__defaults__, closure=fit_poly.__closure__,
        )
    original = bactfit.fit_cell
    return FunctionType(
        original.__code__, {**original.__globals__, "fit_poly": polynomial},
        argdefs=original.__defaults__, closure=original.__closure__,
    )


def _cells_from_shapes(shapes):
    cells = []
    for shape in shapes:
        shape = np.asarray(shape)
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
