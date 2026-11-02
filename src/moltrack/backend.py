"""MolCore adapters for MolTrack's existing array and record-array interfaces.

Each operation owns a detached model. No GUI objects or application dictionaries
are sent to worker processes, and source pixels/records are never mutated.
"""

from concurrent.futures import CancelledError
from dataclasses import replace

import numpy as np
import pandas as pd
from molcore import (
    Acquisition,
    ApertureShape,
    ArraySource,
    Calibration,
    CameraSettings,
    ChannelSpec,
    CoordinateSpace,
    Dataset,
    DriftSettings,
    LocalisationSettings,
    PhotometrySettings,
    RenderSettings,
    TrackingSettings,
    localisation,
    photometry,
    rendering,
    tracking,
    undrifting,
)
from molcore.transforms import project_points

ACQUISITION = "movie"
CHANNEL = "image"


def progress_adapter(callback):
    if callback is None:
        return None
    emit = getattr(callback, "emit", callback)
    return lambda update: emit(int(np.clip(update.percent, 0, 100)))


def _check_cancelled(event):
    if event is not None and event.is_set():
        raise CancelledError()


def _model(movie=None, rows=None, shape=None):
    if movie is None:
        frame_count = (
            max(1, int(np.max(rows["frame"])) + 1) if len(rows) else 1
        )
        # Broadcast geometry without allocating a synthetic movie.
        height, width = shape or (1, 1)
        movie = np.broadcast_to(
            np.zeros((1, 1, 1), dtype=np.uint8), (frame_count, height, width)
        )
    data = np.asarray(movie)
    if data.ndim == 2:
        data = data[None]
    dataset = Dataset()
    dataset.add_acquisition(
        Acquisition(
            ArraySource(data),
            id=ACQUISITION,
            channel_specs={CHANNEL: ChannelSpec()},
        )
    )
    ids = np.array([], dtype=np.uint64)
    if rows is not None and len(rows):
        frame = pd.DataFrame(rows)
        measurements = {
            k: frame[k].to_numpy()
            for k in frame.columns
            if k
            not in {
                "id",
                "frame",
                "x",
                "y",
                "channel",
                "acquisition_id",
                "dataset",
            }
            and frame[k].dtype.kind in "biuf"
        }
        ids = dataset.localizations.add(
            acquisition_id=ACQUISITION,
            channel=CHANNEL,
            frame=frame["frame"].to_numpy(dtype=np.int64),
            x=frame["x"].to_numpy(dtype=float),
            y=frame["y"].to_numpy(dtype=float),
            **measurements,
        )
    return dataset, ids


def transform_locs(locs, transform_matrix, progress_callback=None):
    matrix = _matrix(transform_matrix)
    result = np.array(locs, copy=True).view(np.recarray)
    points = project_points(matrix, np.column_stack([locs["x"], locs["y"]]))
    result.x, result.y = points[:, 0], points[:, 1]
    if progress_callback is not None:
        getattr(progress_callback, "emit", progress_callback)(100)
    return result


def _matrix(value):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, [0, 0, 1]])
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite (2, 3) or (3, 3) transformation")
    return matrix


def transform_image(img, transform_matrix, progress_callback=None):
    data = np.asarray(img)
    dataset, _ = _model(data)
    dataset.transforms.add_calibration(
        Calibration("registration", _matrix(transform_matrix))
    )
    dataset.transforms.select(ACQUISITION, CHANNEL, "registration")
    result = np.empty_like(data)
    emit = getattr(progress_callback, "emit", progress_callback)
    for index in range(len(data)):
        result[index] = dataset.transforms.image(ACQUISITION, CHANNEL, index)
        if emit is not None:
            emit(int(100 * (index + 1) / len(data)))
    return result


def link_locs(
    rows,
    search_range,
    *,
    memory=0,
    min_length=1,
    progress_callback=None,
    cancel_event=None,
):
    frame = pd.DataFrame(rows).copy()
    _check_cancelled(cancel_event)
    dataset, ids = _model(rows=frame)
    result = tracking.track(
        dataset,
        ACQUISITION,
        CHANNEL,
        settings=TrackingSettings(
            search_range=search_range,
            memory=memory,
            min_length=min_length,
            space=CoordinateSpace.RAW,
        ),
        apply=False,
        num_cpus=1,
        progress=progress_adapter(progress_callback),
        cancel_event=cancel_event,
    )
    lookup = {int(lid): index for index, lid in enumerate(ids)}
    indices, particles = [], []
    for particle, members in enumerate(result.memberships, 1):
        indices.extend(lookup[int(lid)] for lid in members)
        particles.extend([particle] * len(members))
    output = frame.iloc[indices].copy()
    output["particle"] = np.asarray(particles, dtype=np.int64)
    return output.sort_values(["particle", "frame"]).reset_index(drop=True)


def render_locs(
    locs,
    image_shape,
    *,
    blur_method=None,
    min_blur_width=1,
    oversampling=10,
    progress_callback=None,
):
    dataset, _ = _model(rows=locs, shape=image_shape[-2:])
    height, width = image_shape[-2:]
    result = rendering.render_localizations(
        dataset,
        ACQUISITION,
        CHANNEL,
        space="raw",
        settings=RenderSettings(
            oversampling=oversampling,
            blur_method=blur_method or "none",
            min_blur_width=min_blur_width,
        ),
        viewport=((0.0, 0.0), (float(height), float(width))),
        progress=progress_adapter(progress_callback),
    )
    return result.count, result.image


def _filter_candidates(columns, roi, remove_overlapping, box_size):
    from scipy.spatial import cKDTree

    frame = pd.DataFrame(columns)
    keep = np.ones(len(frame), dtype=bool)
    if roi is not None:
        (y0, x0), (y1, x1) = roi
        keep &= (
            (frame.x >= x0) & (frame.x < x1) & (frame.y >= y0) & (frame.y < y1)
        )
    if remove_overlapping:
        for _, group in frame.groupby("frame"):
            pairs = cKDTree(group[["x", "y"]]).query_pairs(
                box_size, output_type="ndarray"
            )
            if len(pairs):
                keep[group.index.to_numpy()[np.unique(pairs)]] = False
    return {k: np.asarray(v)[keep].copy() for k, v in columns.items()}


def localize_movie(
    movie,
    *,
    threshold,
    box_size,
    fit=True,
    candidates=None,
    frame_index=None,
    roi=None,
    remove_overlapping=False,
    detect_mode="Picasso",
    kernel_size=3,
    use_gpu=False,
    progress_callback=None,
    cancel_event=None,
):
    """Detect/fit through MolCore, retaining the application's frame numbering."""
    _check_cancelled(cancel_event)
    dataset, _ = _model(movie)
    settings = LocalisationSettings(
        min_net_gradient=threshold, box_size=box_size
    )
    options = {
        "num_cpus": 1,
        "progress": progress_adapter(progress_callback),
        "cancel_event": cancel_event,
    }
    frame_range = (
        None if frame_index is None else (frame_index, frame_index + 1)
    )
    if candidates is None:
        custom = (
            MoltrackDetection(kernel_size)
            if detect_mode.lower() != "picasso"
            else None
        )
        detected = localisation.detect(
            dataset,
            ACQUISITION,
            CHANNEL,
            settings=settings,
            frame_range=frame_range,
            backend=custom,
            **options,
        )
    else:
        # Obtain the public source token without processing any frames.
        detected = localisation.detect(
            dataset,
            ACQUISITION,
            CHANNEL,
            settings=settings,
            frame_range=(0, 0),
            **options,
        )
        data = pd.DataFrame(candidates).reset_index(drop=True)
        if frame_index is not None:
            data = data[data.frame == frame_index]
        columns = {k: data[k].to_numpy(copy=True) for k in ("frame", "x", "y")}
        columns["net_gradient"] = (
            data.net_gradient.to_numpy(copy=True)
            if "net_gradient" in data
            else np.zeros(len(data))
        )
        # Picasso preserves its public cross-channel row identifier through fits.
        # Keep arbitrary application columns in the caller, keyed by that ID.
        columns['n_id'] = data.index.to_numpy(dtype=np.uint32)
        detected = replace(detected, columns=columns, frame_range=frame_range)
    filtered = _filter_candidates(
        detected.columns, roi, remove_overlapping, box_size
    )
    detected = replace(detected, columns=filtered)
    if fit:
        result = localisation.fit(
            dataset,
            detected,
            camera=CameraSettings(100.0, 1.0, 1.0),
            apply=False,
            backend=PicassoGpu() if use_gpu else None,
            **options,
        )
        if result.failed_fits and not len(result.columns["frame"]):
            raise ValueError(
                "No fits converged; previous localisations were preserved"
            )
        columns = result.columns
    else:
        columns = detected.columns
    if candidates is not None:
        identity = columns.pop('n_id', np.array([], dtype=np.uint32))
        if len(identity) != len(columns['frame']):
            raise ValueError('Fitter did not preserve candidate row identities')
        original = pd.DataFrame(candidates).reset_index(drop=True)
        for name in original:
            if name not in columns or name == 'n_id':
                columns[name] = original[name].to_numpy()[identity]
    _check_cancelled(cancel_event)
    return pd.DataFrame(columns).to_records(index=False)


class MoltrackDetection:
    """Retained custom detector, plugged into MolCore's chunk executor."""

    def __init__(self, kernel_size):
        self.kernel_size = max(1, round(kernel_size))

    version = "moltrack-bandpass-1"

    def compute(self, payload, reporter):
        import cv2
        from skimage.feature import peak_local_max

        images, frames, settings, _, _, _ = payload
        size = self.kernel_size * 2 + 1
        gaussian = cv2.getGaussianKernel(5, 1)
        gaussian = gaussian @ gaussian.T
        gaussian = cv2.filter2D(gaussian, -1, gaussian)
        box = np.ones((size, size)) / size**2
        box = cv2.filter2D(box, -1, box)
        # Filtering separately also supports unequal kernel dimensions.
        rows = []
        border = max(5, self.kernel_size, settings.box_size // 2)
        for index, (image, frame) in enumerate(
            zip(images, frames, strict=False)
        ):
            reporter.check_cancelled()
            source = image.astype(np.float32)
            filtered = cv2.filter2D(source, -1, gaussian) - cv2.filter2D(
                source, -1, box
            )
            filtered[:border] = filtered[-border:] = 0
            filtered[:, :border] = filtered[:, -border:] = 0
            for y, x in peak_local_max(
                filtered,
                min_distance=1,
                threshold_abs=settings.min_net_gradient,
            ):
                rows.append((frame, x, y, float(filtered[y, x])))
            reporter.report(index + 1, len(frames), "MolTrack detection")
        data = pd.DataFrame(rows, columns=["frame", "x", "y", "net_gradient"])
        return {
            k: data[k].to_numpy(dtype=np.int64 if k == "frame" else float)
            for k in data
        }, 0


class PicassoGpu:
    """Use Picasso's unified Gaussian CUDA fitter as a MolCore extension."""

    @property
    def version(self):
        from importlib.metadata import version

        return version("picassosr")

    def compute(self, payload, reporter):
        from picasso import localize

        images, frames, settings, camera, candidates, _ = payload
        rows, failed = [], 0
        for index, (image, frame) in enumerate(
            zip(images, frames, strict=False)
        ):
            reporter.check_cancelled()
            found = pd.DataFrame(
                {
                    k: v[candidates["frame"] == frame]
                    for k, v in candidates.items()
                }
            )
            found["frame"] = np.zeros(len(found), dtype=np.uint32)
            if len(found):
                spots = localize.get_spots(
                    image[None],
                    found,
                    settings.box_size,
                    {
                        "Baseline": camera.baseline,
                        "Sensitivity": camera.sensitivity,
                        "Gain": camera.gain,
                    },
                )
                theta, likelihood, iterations, chi_square = (
                    localize.fit_spots_gauss(
                        spots,
                        use_gpu=True,
                        mle=settings.fit_method == "mle",
                        return_stats=True,
                        tolerance=settings.tolerance,
                        max_iterations=settings.max_iterations,
                    )
                )
                found = localize.locs_from_fits_gauss(
                    found,
                    theta,
                    settings.box_size,
                    em=camera.em,
                    mle=settings.fit_method == "mle",
                    iterations=iterations,
                    chi_square=chi_square,
                    log_likelihood=likelihood,
                )
                valid = np.isfinite(
                    found[["x", "y", "photons", "sx", "sy"]]
                ).all(axis=1)
                valid &= (found.photons >= 0) & (found.sx > 0) & (found.sy > 0)
                failed += int((~valid).sum())
                found = found.loc[valid].copy()
                found["frame"] = frame
                rows.append(found)
            reporter.report(index + 1, len(frames), "Picasso GPU fitting")
        data = (
            pd.concat(rows, ignore_index=True)
            if rows
            else pd.DataFrame(
                {
                    k: np.array([], dtype=float)
                    for k in (
                        "frame",
                        "x",
                        "y",
                        "net_gradient",
                        "photons",
                        "bg",
                        "sx",
                        "sy",
                        "lpx",
                        "lpy",
                    )
                }
            )
        )
        return {k: data[k].to_numpy(copy=True) for k in data}, failed


def measure_locs(
    movie,
    rows,
    *,
    spot_size,
    spot_shape,
    background_buffer,
    background_width,
    progress_callback=None,
    cancel_event=None,
):
    data = pd.DataFrame(rows).copy()
    dataset, ids = _model(movie, data)
    result = photometry.measure_localizations(
        dataset,
        ids,
        apply=False,
        num_cpus=1,
        settings=PhotometrySettings(
            spot_size=spot_size,
            shape=ApertureShape(spot_shape.lower()),
            buffer=background_buffer,
            background_width=background_width,
            exclude_neighbours=False,
        ),
        progress=progress_adapter(progress_callback),
        cancel_event=cancel_event,
    )
    measurements = (
        pd.DataFrame(result.columns).set_index("localization_id").reindex(ids)
    )
    for metric in ("mean", "median", "sum", "max", "std"):
        data[f"pixel_{metric}"] = measurements[f"signal_{metric}"].to_numpy()
        background = measurements[f"local_background_{metric}"].to_numpy()
        if metric == "sum":
            background = (
                measurements.local_background_mean * measurements.signal_count
            ).to_numpy()
        data[f"pixel_{metric}_bg"] = background
    return data


def estimate_drift(
    movie,
    locs,
    *,
    method,
    segment_frames,
    intersect_distance=0.15,
    search_radius=0.5,
    progress_callback=None,
    cancel_event=None,
):
    dataset, _ = _model(movie, locs)
    return undrifting.estimate_drift(
        dataset,
        ACQUISITION,
        CHANNEL,
        settings=DriftSettings(
            method=method.lower(),
            segment_frames=segment_frames,
            intersect_distance=intersect_distance,
            search_radius=search_radius,
        ),
        num_cpus=1,
        progress=progress_adapter(progress_callback),
        cancel_event=cancel_event,
    )


def correct_drift(movie, locs, result, *, images=True, cancel_event=None,
                  source_frames=None, target_frames=None):
    """Correct measured coverage; leave uncovered frames unchanged."""
    _check_cancelled(cancel_event)
    dataset, _ = _model(movie, locs)
    displacement = np.zeros((len(movie), 2), dtype=float)
    reference = result.frames if source_frames is None else np.asarray(source_frames)[result.frames]
    target = np.arange(len(movie)) if target_frames is None else np.asarray(target_frames)
    if len(reference):
        covered = (target >= reference[0]) & (target <= reference[-1])
        for axis in (0, 1):
            displacement[covered, axis] = np.interp(target[covered], reference, result.displacement[:, axis])
    dataset.transforms.set_drift(ACQUISITION, displacement)
    corrected_locs = None
    if locs is not None:
        corrected_locs = np.array(locs, copy=True).view(np.recarray)
        if len(locs):
            rows = dataset.localizations.query(space="corrected")
            corrected_locs.x, corrected_locs.y = rows["x"], rows["y"]
    corrected_movie = None
    if images:
        corrected_movie = np.empty_like(movie)
        for index in range(len(movie)):
            _check_cancelled(cancel_event)
            corrected_movie[index] = dataset.transforms.image(
                ACQUISITION, CHANNEL, index
            )
    _check_cancelled(cancel_event)
    return corrected_movie, corrected_locs


def _trajectory(rows, time_step, fit_lags):
    from molcore import SpatialUnit
    from molcore.trajectory import trajectory_statistics

    dataset, ids = _model(rows=rows)
    track_id = dataset.tracks.add(ids)
    kwargs = {
        "space": CoordinateSpace.RAW,
        "unit": SpatialUnit.PIXELS,
        "frame_interval": time_step,
    }
    result = trajectory_statistics(dataset, track_id, **kwargs)
    lags = result.msd["lag"][:fit_lags]
    if len(lags) >= 2:
        result = trajectory_statistics(
            dataset, track_id, fit_lag_range=(lags[0], lags[-1]), **kwargs
        )
    return result


def track_statistics(
    rows, *, pixel_size, time_step, fit_lags=4, rolling_window=4
):
    data = (
        pd.DataFrame(rows)
        .sort_values("frame", kind="stable")
        .reset_index(drop=True)
    )
    if not len(data):
        return {}
    result = _trajectory(data, time_step, fit_lags)
    distance = np.r_[0.0, result.steps["distance"]] * pixel_size
    speed = np.r_[0.0, result.steps["speed"]] * pixel_size
    diffusion = (
        np.nan
        if result.diffusion is None
        else result.diffusion.coefficient * pixel_size**2
    )
    sigma = data.sigma.to_numpy() * pixel_size if "sigma" in data else None
    corrected = (
        diffusion - np.mean(sigma**2) / time_step
        if sigma is not None
        else np.nan
    )
    rolling, rolling_corrected = [], []
    for index in range(len(data)):
        first = max(0, index - rolling_window // 2)
        last = min(len(data), index + rolling_window // 2 + 1)
        subset = data.iloc[first:last]
        current = (
            _trajectory(subset, time_step, fit_lags)
            if len(subset) >= 3
            else None
        )
        value = (
            np.nan
            if current is None or current.diffusion is None
            else current.diffusion.coefficient * pixel_size**2
        )
        rolling.append(value)
        rolling_corrected.append(
            value - np.mean(sigma[first:last] ** 2) / time_step
            if sigma is not None
            else np.nan
        )
    msd = np.full(len(data), np.nan)
    msd[0] = 0.0
    count = min(len(data) - 1, len(result.msd["msd"]))
    msd[1 : count + 1] = result.msd["msd"][:count] * pixel_size**2
    return {
        "step_size": distance,
        "speed": speed,
        "time": (data.frame.to_numpy() - data.frame.iloc[0]) * time_step,
        "msd": msd,
        "D": diffusion,
        "D*": corrected,
        "rD": np.asarray(rolling),
        "rD*": np.asarray(rolling_corrected),
    }


def segment_images(
    images,
    *,
    settings,
    progress_callback=None,
    cancel_event=None,
    backend=None,
):
    from molcore import SegmentationRequest
    from molcore.cellpose import segment_many

    dataset = Dataset()
    requests = []
    for index, image in enumerate(images):
        name = str(index)
        dataset.add_acquisition(
            Acquisition(
                ArraySource(np.asarray(image)[None]),
                id=name,
                channel_specs={CHANNEL: ChannelSpec()},
            )
        )
        requests.append(
            SegmentationRequest(name, CHANNEL, frame=0, segmentation_id=name)
        )
    results = segment_many(
        dataset,
        requests,
        settings=settings,
        apply=False,
        num_cpus=1,
        progress=progress_adapter(progress_callback),
        cancel_event=cancel_event,
        backend=backend,
    )
    return [results[str(index)].labels for index in range(len(images))]


def filter_locs(locs, metric, minimum, maximum, *, subtract_background=False):
    from molcore import FilterSettings
    from molcore.filtering import filter_results

    data = pd.DataFrame(locs).copy()
    if data.empty:
        return np.array(locs, copy=True).view(np.recarray)
    values = data[metric].to_numpy(dtype=float)
    if subtract_background and metric + "_bg" in data:
        values = values - data[metric + "_bg"].to_numpy(dtype=float)
    data["_filter_value"] = values
    model, ids = _model(rows=data)
    # MolCore bounds are inclusive; the existing GUI uses open bounds.
    low, high = (
        np.nextafter(float(minimum), np.inf),
        np.nextafter(float(maximum), -np.inf),
    )
    if low > high:
        return np.array(locs[:0], copy=True).view(np.recarray)
    result = filter_results(
        model,
        ACQUISITION,
        CHANNEL,
        settings=FilterSettings(
            metric="_filter_value", minimum=low, maximum=high
        ),
    )
    return np.array(locs[np.isin(ids, result.selected_ids)], copy=True).view(
        np.recarray
    )


def merge_locs(rows, radius, progress_callback=None):
    from molcore import ClusteringSettings
    from molcore.clustering import cluster_localizations

    output = []
    groups = list(pd.DataFrame(rows).groupby("frame", sort=True))
    for index, (_, group) in enumerate(groups):
        model, ids = _model(rows=group)
        result = cluster_localizations(
            model,
            ACQUISITION,
            CHANNEL,
            settings=ClusteringSettings(radius=radius, min_samples=2),
        )
        labels = (
            pd.Series(result.columns["cluster"], index=result.columns["id"])
            .reindex(ids)
            .to_numpy()
        )
        keep = (labels == -1) | ~pd.Series(labels).duplicated().to_numpy()
        output.append(group.iloc[np.flatnonzero(keep)])
        if progress_callback is not None:
            getattr(progress_callback, "emit", progress_callback)(
                int(100 * (index + 1) / len(groups))
            )
    return (
        pd.concat(output, ignore_index=True)
        if output
        else pd.DataFrame(rows).copy()
    )


def image_source(path):
    from pathlib import Path

    from molcore import FitsSource, TiffSource

    extension = Path(path).suffix.lower()
    if extension in (".tif", ".tiff"):
        return TiffSource(path)
    if extension in (".fits", ".fit", ".fts"):
        return FitsSource(path)
    raise ValueError(f"Unsupported movie format: {extension}")


def read_movie(path, *, limit=None, crop=None, progress_callback=None):
    source = image_source(path)
    count = len(source) if limit is None else min(len(source), int(limit))
    frames = []
    try:
        for index in range(count):
            frame = source[index]
            frames.append(frame if crop is None else crop(frame))
            if progress_callback is not None:
                progress_callback(int(100 * (index + 1) / count))
        return (
            np.stack(frames)
            if frames
            else np.empty((0, *source.shape[1:]), dtype=source.dtype)
        )
    finally:
        source.close()
