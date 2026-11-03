"""Persistent-model trajectory analysis and detached worker publication."""

from copy import deepcopy
from dataclasses import dataclass
from weakref import ref

import numpy as np
import pandas as pd
from molcore import CoordinateSpace, SpatialUnit, StaleResultError
from molcore.staging import state_token, transact, validate_expected
from molcore.trajectory import trajectory_statistics

from .model_processing import _cancel

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


def _trajectory(model, track_id, ids, interval, fit_lags):
    options = {
        "ids": ids,
        "space": CoordinateSpace.CORRECTED,
        "unit": SpatialUnit.PIXELS,
        "frame_interval": interval,
    }
    result = trajectory_statistics(model, track_id, **options)
    lags = result.msd["lag"][:fit_lags]
    if len(lags) >= 2:
        result = trajectory_statistics(
            model, track_id, fit_lag_range=(lags[0], lags[-1]), **options
        )
    return result


def analyze_tracks(
    session,
    *,
    fit_lags=4,
    rolling_window=4,
    shape_data=None,
    progress_callback=None,
    cancel_event=None,
):
    _cancel(cancel_event)
    # GUI-only geometry extensions share no ownership with the model tables.
    from moltrack.funcs.trackstats_utils import _trackstats_utils
    from moltrack.model_views import gui_query

    data = pd.DataFrame(
        gui_query(session, "All Datasets", "All Channels", tracks=True)
    )
    if data.empty:
        return data
    groups = list(data.groupby(["dataset", "channel", "track_id"], sort=False))
    results = []
    for index, ((dataset, channel, track_id), rows) in enumerate(groups):
        _cancel(cancel_event)
        rows = (
            rows.sort_values("frame", kind="stable")
            .reset_index(drop=True)
            .copy()
        )
        acquisition, _ = session.partition(dataset, channel)
        source = session.model.acquisitions[acquisition]
        scale = source.pixel_size * 1e-3
        interval = source.metadata.get("exposure_s", 0)
        # A missing legacy exposure retains the historical 1-second fallback.
        interval = interval if interval > 0 else 1.0
        ids = rows.id.to_numpy()
        result = _trajectory(
            session.model, int(track_id), ids, interval, fit_lags
        )
        rows["step_size"] = np.r_[0.0, result.steps["distance"]] * scale
        rows["speed"] = np.r_[0.0, result.steps["speed"]] * scale
        diffusion = (
            np.nan
            if result.diffusion is None
            else result.diffusion.coefficient * scale**2
        )
        sigma = rows.sigma.to_numpy() * scale if "sigma" in rows else None
        rows["D"] = diffusion
        rows["D*"] = (
            diffusion - np.mean(sigma**2) / interval
            if sigma is not None
            else np.nan
        )
        rolling, corrected = [], []
        for offset in range(len(rows)):
            _cancel(cancel_event)
            first = max(0, offset - rolling_window // 2)
            last = min(len(rows), offset + rolling_window // 2 + 1)
            current = (
                _trajectory(
                    session.model,
                    int(track_id),
                    ids[first:last],
                    interval,
                    fit_lags,
                )
                if last - first >= 3
                else None
            )
            value = (
                np.nan
                if current is None or current.diffusion is None
                else current.diffusion.coefficient * scale**2
            )
            rolling.append(value)
            corrected.append(
                value - np.mean(sigma[first:last] ** 2) / interval
                if sigma is not None
                else np.nan
            )
        rows["rD"], rows["rD*"] = rolling, corrected
        msd = np.full(len(rows), np.nan)
        msd[0] = 0.0
        count = min(len(rows) - 1, len(result.msd["msd"]))
        msd[1 : count + 1] = result.msd["msd"][:count] * scale**2
        rows["msd"] = msd
        camera_frames = session._view(dataset, channel).source_frames[
            rows.frame.to_numpy(dtype=int)
        ]
        times = (
            camera_frames * interval
            if source.timestamps is None
            else source.timestamps[camera_frames]
        )
        rows["time"] = times - times[0]
        rows["angle"] = _trackstats_utils.calculate_track_angles(
            rows[["x", "y"]].to_numpy()
        )
        if len(rows) >= 2:
            for key, value in _trackstats_utils.get_track_shape_stats(
                rows, shape_data, scale
            ).items():
                rows[key] = value
        results.append(rows)
        if progress_callback is not None:
            progress_callback.emit(int(100 * (index + 1) / len(groups)))
    _cancel(cancel_event)
    return pd.concat(results, ignore_index=True)
