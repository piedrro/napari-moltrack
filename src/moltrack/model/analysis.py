"""Persistent-model trajectory analysis and detached worker publication."""

from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from weakref import ref

import numpy as np
import pandas as pd
from molcore import (
    CoordinateSpace,
    CpuExecutor,
    SerialExecutor,
    SpatialUnit,
    StaleResultError,
)
from molcore.staging import state_token, transact, validate_expected
from molcore.trajectory import trajectory_statistics
from shapely import distance, points
from shapely.errors import GEOSException
from shapely.geometry import LineString, Point

from . import config
from .backend import _model, progress_adapter
from .processing import _cancel

_ANALYSIS_ROWS_PER_WORKER = 256

STAT_COLUMNS = (
    "step_size",
    "speed",
    "time",
    "msd",
    "D",
    "D*",
    "rD",
    "rD*",
    "angle",
    "centroid_distance",
    "membrane_distance",
    "midline_distance",
    "cell_pole_distance",
)


@dataclass(frozen=True)
class PreparedAnalysis:
    session: object
    expected: object
    owner: object
    bindings: object
    fit_lags: int
    rolling_window: int
    shape_data: object
    cancel_event: object = None


@dataclass(frozen=True)
class PendingAnalysis:
    prepared: PreparedAnalysis
    rows: object


@dataclass(frozen=True)
class _TrackInput:
    coordinates: np.ndarray
    times: np.ndarray
    sigma: np.ndarray | None
    scale: float
    interval: float


def prepare_analysis(
    session,
    *,
    fit_lags=4,
    rolling_window=4,
    shape_data=None,
    cancel_event=None,
):
    return PreparedAnalysis(
        session.snapshot(),
        state_token(session.model),
        ref(session.model),
        deepcopy(session._partitions),
        int(fit_lags),
        int(rolling_window),
        deepcopy(shape_data),
        cancel_event,
    )


def compute_analysis(prepared, progress_callback=None):
    _cancel(prepared.cancel_event)
    pending = PendingAnalysis(
        prepared,
        prepared.session.analyze_tracks(
            fit_lags=prepared.fit_lags,
            rolling_window=prepared.rolling_window,
            shape_data=prepared.shape_data,
            progress_callback=progress_callback,
            cancel_event=prepared.cancel_event,
        ),
    )
    _cancel(prepared.cancel_event)
    return pending


def publish_analysis(session, pending):
    prepared = pending.prepared
    _cancel(prepared.cancel_event)
    if (
        session.model is not prepared.owner()
        or session._partitions != prepared.bindings
    ):
        raise StaleResultError("Session bindings changed")
    validate_expected(session.model, prepared.expected)
    rows = pending.rows
    columns = {
        name: rows[name].to_numpy() for name in STAT_COLUMNS if name in rows
    }
    if len(rows) and columns:

        def commit(model):
            _cancel(prepared.cancel_event)
            model.localizations.update_measurements(
                rows.id.to_numpy(), **columns
            )
            _cancel(prepared.cancel_event)

        transact(session.model, commit, expected=prepared.expected)
    _cancel(prepared.cancel_event)
    return rows


def _calculate_track_angles(coords):
    # Calculate the difference between each consecutive point
    vectors = np.diff(coords, axis=0)

    # Calculate the angles of these vectors relative to the x-axis
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])

    # Convert angles from radians to degrees
    angles = np.degrees(angles)

    # Initialize the relative angles array with zeros
    relative_angles = np.zeros(len(coords))

    # The first angle is zero
    relative_angles[0] = 0

    # Calculate the difference in angles between consecutive vectors
    for i in range(1, len(angles) + 1):
        relative_angles[i] = (angles[i - 1] - angles[i - 2] + 360) % 360 if i > 1 else angles[i - 1]

    return relative_angles


def _track_shape_stats(coordinates, shape_data, scale):
    if not shape_data or len(coordinates) < 2:
        return {}
    track_line = LineString(coordinates)
    match = next(
        (
            (index, polygon)
            for index, polygon in enumerate(shape_data.get("polygons", ()))
            if track_line.intersects(polygon)
        ),
        None,
    )
    if match is None:
        return {}

    index, polygon = match
    locations = points(coordinates)
    metrics = {"centroid_distance": distance(locations, polygon.centroid) * scale}
    if hasattr(polygon, "exterior"):
        metrics["membrane_distance"] = distance(locations, polygon.exterior) * scale
    midlines = shape_data.get("midlines", ())
    if index < len(midlines) and midlines[index] is not None:
        midline = midlines[index]
        metrics["midline_distance"] = distance(locations, midline) * scale
        if isinstance(midline, LineString) and not midline.is_empty:
            first = distance(locations, Point(midline.coords[0]))
            last = distance(locations, Point(midline.coords[-1]))
            metrics["cell_pole_distance"] = np.minimum(first, last) * scale
    return metrics


def _trajectory(snapshot, track_id, ids, interval, fit_lags):
    options = {
        "ids": ids,
        "space": CoordinateSpace.RAW,
        "unit": SpatialUnit.PIXELS,
        "frame_interval": interval,
    }
    result = trajectory_statistics(snapshot, track_id, **options)
    lags = result.msd["lag"][:fit_lags]
    if len(lags) >= 2:
        result = trajectory_statistics(
            snapshot,
            track_id,
            fit_lag_range=(lags[0], lags[-1]),
            **options,
        )
    return result


def _analyze_track(track, fit_lags, rolling_window, shape_data, reporter):
    coordinates, scale, interval = track.coordinates, track.scale, track.interval
    # Corrected positions and their absolute times are already resolved by the caller.
    # The local model uses compact frame indices and carries no movie data.
    rows = pd.DataFrame({
        "frame": np.arange(len(coordinates)),
        "x": coordinates[:, 0],
        "y": coordinates[:, 1],
    })
    model, ids = _model(rows=rows, timestamps=track.times)
    track_id = model.tracks.add(ids)
    snapshot = model.snapshot()
    result = _trajectory(snapshot, track_id, ids, interval, fit_lags)
    diffusion = (
        np.nan
        if result.diffusion is None
        else result.diffusion.coefficient * scale**2
    )
    sigma = track.sigma * scale if track.sigma is not None else None
    corrected = (
        diffusion - np.mean(sigma**2) / interval
        if sigma is not None
        else np.nan
    )
    rolling, rolling_corrected = [], []
    for offset in range(len(coordinates)):
        reporter.check_cancelled()
        first = max(0, offset - rolling_window // 2)
        last = min(len(coordinates), offset + rolling_window // 2 + 1)
        current = (
            _trajectory(snapshot, track_id, ids[first:last], interval, fit_lags)
            if last - first >= 3
            else None
        )
        value = (
            np.nan
            if current is None or current.diffusion is None
            else current.diffusion.coefficient * scale**2
        )
        rolling.append(value)
        rolling_corrected.append(
            value - np.mean(sigma[first:last] ** 2) / interval
            if sigma is not None
            else np.nan
        )
    msd = np.full(len(coordinates), np.nan)
    msd[0] = 0.0
    count = min(len(coordinates) - 1, len(result.msd["msd"]))
    msd[1 : count + 1] = result.msd["msd"][:count] * scale**2
    metrics = {
        "step_size": np.r_[0.0, result.steps["distance"]] * scale,
        "speed": np.r_[0.0, result.steps["speed"]] * scale,
        "D": diffusion,
        "D*": corrected,
        "rD": rolling,
        "rD*": rolling_corrected,
        "msd": msd,
        "time": track.times - track.times[0],
        "angle": _calculate_track_angles(coordinates),
    }
    # Keep trajectory statistics when optional shape geometry is invalid.
    with suppress(GEOSException):
        metrics.update(_track_shape_stats(coordinates, shape_data, scale))
    return metrics


def _analyze_batch(payload, reporter):
    tracks, fit_lags, rolling_window, shape_data = payload
    results = []
    for index, track in enumerate(tracks, 1):
        reporter.check_cancelled()
        results.append(_analyze_track(track, fit_lags, rolling_window, shape_data, reporter))
        reporter.report(index, len(tracks), "Computing track statistics")
    return results


def _track_input(session, dataset, channel, rows):
    view = session._view(dataset, channel)
    source = view.acquisition
    interval = source.metadata.get("exposure_s", 0)
    interval = interval if interval > 0 else 1.0
    frames = view.source_frames[rows.frame.to_numpy(dtype=int)]
    times = frames * interval if source.timestamps is None else source.timestamps[frames]
    return _TrackInput(
        rows[["x", "y"]].to_numpy(dtype=float),
        times,
        rows.sigma.to_numpy(dtype=float, na_value=np.nan) if "sigma" in rows else None,
        source.pixel_size * 0.001,
        interval,
    )


def analyze_tracks(
    session,
    *,
    fit_lags=4,
    rolling_window=4,
    shape_data=None,
    progress_callback=None,
    cancel_event=None,
):
    from .views import gui_query

    _cancel(cancel_event)
    data = pd.DataFrame(
        gui_query(session, "All Datasets", "All Channels", tracks=True)
    )
    if data.empty:
        return data
    groups = [
        (dataset, channel, rows.sort_values("frame", kind="stable"))
        for (dataset, channel, _), rows in data.groupby(
            ["dataset", "channel", "track_id"], sort=False
        )
    ]
    # Amortize process startup and limit worker counts on high-core machines.
    workers = 1
    if len(data) >= 4 * _ANALYSIS_ROWS_PER_WORKER:
        workers = min(config.NUM_CPUS, len(groups), len(data) // _ANALYSIS_ROWS_PER_WORKER)
    batch_size = max(1, min(16, len(groups) // (4 * workers)))
    starts = range(0, len(groups), batch_size)

    def batches():
        for first in starts:
            _cancel(cancel_event)
            tracks = [
                _track_input(session, dataset, channel, rows)
                for dataset, channel, rows in groups[first : first + batch_size]
            ]
            yield tracks, fit_lags, rolling_window, shape_data

    executor = CpuExecutor(num_cpus=workers) if workers > 1 else SerialExecutor()
    with executor:
        batches_results = executor.imap(
            _analyze_batch,
            batches(),
            total=len(starts),
            progress=progress_adapter(progress_callback),
            cancel_event=cancel_event,
        )
        metrics = (track_metrics for batch in batches_results for track_metrics in batch)
        results = [
            rows.assign(**track_metrics)
            for (_, _, rows), track_metrics in zip(groups, metrics, strict=True)
        ]
    _cancel(cancel_event)
    return pd.concat(results, ignore_index=True)
