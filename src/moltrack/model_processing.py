"""Detached processing requests and atomic GUI-thread publication."""

from concurrent.futures import CancelledError
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from weakref import ref

import numpy as np
import pandas as pd
from molcore import (
    ApertureShape,
    Calibration,
    CameraSettings,
    LocalisationSettings,
    RenderSettings,
    SegmentationRequest,
    StaleResultError,
    cellpose,
    localisation,
    photometry,
    rendering,
    tracking,
    undrifting,
)
from molcore import (
    PhotometrySettings as ApertureSettings,
)
from molcore.staging import state_token, transact, validate_expected

from .backend import (
    MoltrackDetection,
    PicassoGpu,
    _filter_candidates,
    progress_adapter,
)
from .model_views import channel_rows
from .session import ProjectSession, clear_statistics


@dataclass(frozen=True)
class DetectionSettings:
    detect: bool = True
    fit: bool = False
    threshold: float = 100.0
    box_size: int = 7
    frame_index: int | None = None
    roi: tuple = ()
    remove_overlapping: bool = False
    detect_mode: str = "Picasso"
    kernel_size: int = 3
    fit_mode: str = "Picasso"


@dataclass(frozen=True)
class TrackingSettings:
    search_range: float = 5.0
    memory: int = 0
    min_length: int = 1
    remove_unlinked: bool = False
    polygons: tuple = ()
    segmentation_column: str | None = None


@dataclass(frozen=True)
class PhotometrySettings:
    spot_size: int = 3
    spot_shape: str = "square"
    background_buffer: int = 1
    background_width: int = 1
    tracks: bool = False
    calculate_fret: bool = False


@dataclass(frozen=True)
class RenderingSettings:
    blur_method: str | None = None
    min_blur_width: float = 1.0
    oversampling: float = 10.0


@dataclass(frozen=True)
class SegmentationSettings:
    cellpose: object
    frame_index: int | None = None


@dataclass(frozen=True)
class RegistrationSettings:
    matrix: tuple


@dataclass(frozen=True)
class DriftSettings:
    drift: object
    references: tuple


@dataclass(frozen=True)
class ProcessingRequest:
    operation: str
    partitions: tuple[tuple[str, str], ...]
    settings: object


@dataclass(frozen=True)
class PreparedOperation:
    request: ProcessingRequest
    session: ProjectSession
    bindings: tuple
    expected: object
    owner: object


@dataclass(frozen=True)
class PendingOperation:
    prepared: PreparedOperation
    results: tuple


def _cancel(event):
    if event is not None and event.is_set():
        raise CancelledError()


def prepare(session, request):
    settings_types = {
        "detect": DetectionSettings,
        "tracking": TrackingSettings,
        "photometry": PhotometrySettings,
        "render": RenderingSettings,
        "segmentation": SegmentationSettings,
        "registration": RegistrationSettings,
        "drift": DriftSettings,
    }
    if request.operation not in settings_types or not isinstance(
        request.settings, settings_types[request.operation]
    ):
        raise ValueError("Unsupported processing request")
    bindings = tuple(session.partition(*p) for p in request.partitions)
    expected = state_token(session.model)
    staged = ProjectSession(session.model.snapshot())
    for (dataset, channel), (acquisition, name) in zip(
        request.partitions, bindings, strict=True
    ):
        staged.bind(dataset, channel, acquisition, name)
    return PreparedOperation(
        deepcopy(request), staged, bindings, expected, ref(session.model)
    )


def _detect_partition(
    session, dataset, channel, settings, progress, cancel_event
):
    acquisition, name = session.partition(dataset, channel)
    view = session._view(dataset, channel)
    frame_range = None
    if settings.frame_index is not None:
        camera_frame = int(view.source_frames[settings.frame_index])
        frame_range = (camera_frame, camera_frame + 1)
    options = {
        "num_cpus": 1,
        "progress": progress_adapter(progress),
        "cancel_event": cancel_event,
    }
    params = LocalisationSettings(
        min_net_gradient=settings.threshold, box_size=settings.box_size
    )
    if settings.detect:
        detected = localisation.detect(
            session.model,
            acquisition,
            name,
            settings=params,
            frame_range=frame_range,
            backend=MoltrackDetection(settings.kernel_size)
            if settings.detect_mode.lower() != "picasso"
            else None,
            **options,
        )
        original = None
    else:
        original = session.query_localizations(
            dataset, channel, space="raw"
        ).reset_index(drop=True)
        if settings.frame_index is not None:
            original = original[
                original.frame == settings.frame_index
            ].reset_index(drop=True)
        detected = localisation.detect(
            session.model,
            acquisition,
            name,
            settings=params,
            frame_range=(0, 0),
            **options,
        )
        columns = {k: original[k].to_numpy(copy=True) for k in ("x", "y")}
        columns["frame"] = view.source_frames[
            original.frame.to_numpy(dtype=int)
        ]
        columns["net_gradient"] = (
            original.net_gradient.to_numpy(copy=True)
            if "net_gradient" in original
            else np.zeros(len(original))
        )
        columns["n_id"] = np.arange(len(original), dtype=np.uint32)
        detected = replace(detected, columns=columns, frame_range=frame_range)
    roi = dict(settings.roi).get((dataset, channel))
    detected = replace(
        detected,
        columns=_filter_detection_candidates(
            session.model,
            acquisition,
            name,
            detected.columns,
            roi,
            settings.remove_overlapping and settings.detect,
            settings.box_size,
        ),
    )
    if settings.fit:
        fitted = localisation.fit(
            session.model,
            detected,
            camera=CameraSettings(100.0, 1.0, 1.0),
            apply=False,
            backend=PicassoGpu()
            if settings.fit_mode == "Picasso GPU"
            else None,
            **options,
        )
        if fitted.failed_fits and not len(fitted.columns["frame"]):
            raise ValueError(
                "No fits converged; previous localisations were preserved"
            )
        columns = dict(fitted.columns)
    else:
        columns = dict(detected.columns)
    result = channel_rows(columns, view)
    if original is not None:
        identities = result.pop("n_id").to_numpy(dtype=int)
        for key in original:
            if key not in result:
                result[key] = original[key].to_numpy()[identities]
    if "lpx" in result and "lpy" in result:
        result["sigma"] = np.sqrt((result.lpx**2 + result.lpy**2) / 2)
    return result


def _filter_detection_candidates(
    model, acquisition, channel, columns, roi, remove_overlapping, box_size
):
    # Fitting boxes and overlap distances are defined on original channel pixels.
    # Evaluate overlap before ROI filtering, retaining outside-ROI competitors.
    columns = _filter_candidates(columns, None, remove_overlapping, box_size)
    if roi is None or not len(columns["frame"]):
        return columns
    rows = dict(
        columns,
        acquisition_id=np.full(len(columns["frame"]), acquisition),
        channel=np.full(len(columns["frame"]), channel),
    )
    x, y = model.transforms.coordinates(rows, space="corrected")
    (y0, x0), (y1, x1) = roi
    keep = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    return {
        key: np.asarray(values)[keep].copy() for key, values in columns.items()
    }


def compute(prepared, *, progress=None, cancel_event=None):
    settings = prepared.request.settings
    if prepared.request.operation != "detect":
        return _compute_workflow(
            prepared, progress=progress, cancel_event=cancel_event
        )
    results = []
    for dataset, channel in prepared.request.partitions:
        _cancel(cancel_event)
        rows = _detect_partition(
            prepared.session,
            dataset,
            channel,
            settings,
            progress,
            cancel_event,
        )
        results.append((dataset, channel, rows))
    _cancel(cancel_event)
    return PendingOperation(prepared, tuple(results))


def publish(session, pending, *, cancel_event=None):
    _cancel(cancel_event)
    prepared = pending.prepared
    try:
        bindings = tuple(
            session.partition(*p) for p in prepared.request.partitions
        )
    except KeyError as exc:
        raise StaleResultError("Partition bindings changed") from exc
    if session.model is not prepared.owner() or bindings != prepared.bindings:
        raise StaleResultError("Partition bindings changed")
    settings = prepared.request.settings
    if prepared.request.operation != "detect":
        return _publish_workflow(session, pending, cancel_event=cancel_event)

    def commit(model):
        staged = ProjectSession(model)
        for (dataset, channel), (acquisition, name) in zip(
            prepared.request.partitions, prepared.bindings, strict=True
        ):
            staged.bind(dataset, channel, acquisition, name)
        for dataset, channel, rows in pending.results:
            _cancel(cancel_event)
            if not len(rows):
                rows = pd.DataFrame(
                    {
                        "frame": np.array([], dtype=int),
                        "x": np.array([], dtype=float),
                        "y": np.array([], dtype=float),
                    }
                )
            staged.replace_localizations(
                dataset, channel, rows, frame_index=settings.frame_index
            )
        clear_statistics(model)
        _cancel(cancel_event)

    transact(session.model, commit, expected=prepared.expected)
    annotations = deepcopy(session.annotations.get("partitions", {}))
    for dataset, channel, _ in pending.results:
        entry = annotations.setdefault(dataset, {}).setdefault(channel, {})
        entry.update(fitted=settings.fit, box_size=settings.box_size)
    session.set_annotations("partitions", annotations)


def compute_worker(prepared, cancel_event, progress_callback=None):
    """Free worker entry point: arguments contain no widgets or live model."""
    return compute(
        prepared, progress=progress_callback, cancel_event=cancel_event
    )


def selected_partitions(
    session, dataset="All Datasets", channel="All Channels"
):
    return session.selected_partitions(dataset, channel)


def _measure_partition(session, dataset, channel, settings, options):
    rows = (
        session.query_tracks
        if settings.tracks
        else session.query_localizations
    )(dataset, channel)
    ids = rows.id.to_numpy(dtype=np.uint64)
    result = photometry.measure_localizations(
        session.model,
        ids,
        settings=ApertureSettings(
            spot_size=settings.spot_size,
            shape=ApertureShape(settings.spot_shape.lower()),
            buffer=settings.background_buffer,
            background_width=settings.background_width,
            exclude_neighbours=False,
        ),
        apply=False,
        **options,
    )
    values = (
        pd.DataFrame(result.columns).set_index("localization_id").reindex(ids)
    )
    output = rows.reset_index(drop=True).copy()
    for column in output:
        if "_fret" in column:
            output[column] = np.nan
    for metric in ("mean", "median", "sum", "max", "std"):
        output[f"pixel_{metric}"] = values[f"signal_{metric}"].to_numpy()
        background = values[f"local_background_{metric}"]
        if metric == "sum":
            background = values.local_background_mean * values.signal_count
        output[f"pixel_{metric}_bg"] = background.to_numpy()
    return output


def _compute_workflow(prepared, *, progress, cancel_event):
    session, request = prepared.session, prepared.request
    model, settings = session.model, request.settings
    options = {
        "num_cpus": 1,
        "progress": progress_adapter(progress),
        "cancel_event": cancel_event,
    }
    results = []
    if request.operation == "drift":
        references = {}
        for dataset, channel in settings.references:
            _cancel(cancel_event)
            acquisition, name = session.partition(dataset, channel)
            ids = session.query_localizations(
                dataset, channel, space="raw"
            ).id.to_numpy(dtype=np.uint64)
            references[dataset] = undrifting.estimate_drift(
                model,
                acquisition,
                name,
                ids=ids,
                settings=settings.drift,
                **options,
            )
        seen = set()
        for dataset, channel in request.partitions:
            acquisition, _ = session.partition(dataset, channel)
            if acquisition in seen:
                continue
            seen.add(acquisition)
            result = references[dataset]
            frames = np.arange(
                len(model.acquisitions[acquisition].source), dtype=np.int64
            )
            displacement = np.zeros((len(frames), 2), dtype=float)
            covered = (frames >= result.frames[0]) & (
                frames <= result.frames[-1]
            )
            for axis in (0, 1):
                displacement[covered, axis] = np.interp(
                    frames[covered],
                    result.frames,
                    result.displacement[:, axis],
                )
            results.append((dataset, channel, (frames, displacement)))
    else:
        for dataset, channel in request.partitions:
            _cancel(cancel_event)
            acquisition, name = session.partition(dataset, channel)
            if request.operation == "photometry":
                value = _measure_partition(
                    session, dataset, channel, settings, options
                )
            elif request.operation == "render":
                ids = session.query_localizations(
                    dataset, channel
                ).id.to_numpy(dtype=np.uint64)
                height, width = session._view(*request.partitions[0]).shape[
                    -2:
                ]
                value = rendering.render_localizations(
                    model,
                    acquisition,
                    name,
                    ids=ids,
                    settings=RenderSettings(
                        oversampling=settings.oversampling,
                        blur_method=settings.blur_method or "none",
                        min_blur_width=settings.min_blur_width,
                    ),
                    viewport=((0.0, 0.0), (float(height), float(width))),
                    progress=options["progress"],
                    cancel_event=cancel_event,
                )
            elif request.operation == "registration":
                from .backend import _matrix

                value = _matrix(settings.matrix)
            elif request.operation == "segmentation":
                view = session._view(dataset, channel)
                frames = (
                    view.source_frames
                    if settings.frame_index is None
                    else [view.source_frames[settings.frame_index]]
                )
                requests = tuple(
                    SegmentationRequest(
                        acquisition,
                        name,
                        frame=int(frame),
                        segmentation_id=f"{acquisition}:{name}:cellpose:{frame}",
                        space="corrected",
                    )
                    for frame in frames
                )
                value = cellpose.segment_many(
                    model,
                    requests,
                    settings=settings.cellpose,
                    apply=False,
                    **options,
                )
            elif request.operation == "tracking":
                value = _track_partition(
                    session, dataset, channel, settings, options
                )
            else:
                raise ValueError("Unsupported processing request")
            results.append((dataset, channel, value))
    _cancel(cancel_event)
    if request.operation == "photometry" and settings.calculate_fret:
        results = _photometry_fret(results)
    return PendingOperation(prepared, tuple(results))


def _photometry_fret(results):
    """Retain GUI FRET formulas with explicit row correspondence by track/frame."""
    output = [(d, c, rows.copy()) for d, c, rows in results]
    for dataset in dict.fromkeys(d for d, _, _ in output):
        channels = {c.lower(): rows for d, c, rows in output if d == dataset}
        pair = (
            ("dd", "da")
            if {"dd", "da"} <= channels.keys()
            else ("donor", "acceptor")
        )
        if not set(pair) <= channels.keys():
            continue
        donor, acceptor = (channels[c] for c in pair)
        keys = [
            k for k in ("particle", "frame") if k in donor and k in acceptor
        ]
        left, right = donor.copy(), acceptor.copy()
        # Distinct same-frame localisations correspond by their original row order.
        left["ordinal"] = left.groupby(keys, dropna=False).cumcount()
        right["ordinal"] = right.groupby(keys, dropna=False).cumcount()
        keys += ["ordinal"]
        aligned = right.set_index(keys).reindex(
            pd.MultiIndex.from_frame(left[keys])
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            fields = [
                f"pixel_{metric}"
                for metric in ("mean", "median", "sum", "max", "std")
            ]
            if {"photons", "bg"} <= set(left.columns) and {
                "photons",
                "bg",
            } <= set(aligned.columns):
                fields.append("photons")
            for field in fields:
                bg_field = "bg" if field == "photons" else field + "_bg"
                d = left[field].to_numpy() - left[bg_field].to_numpy()
                a = aligned[field].to_numpy() - aligned[bg_field].to_numpy()
                if pair[0] == "dd":
                    value = np.clip(a / (a + d), 0, 1)
                else:
                    value = (
                        aligned[field].to_numpy() - left[field].to_numpy()
                    ) / aligned[field].to_numpy()
                values = pd.Series(
                    value, index=pd.MultiIndex.from_frame(left[keys])
                )
                for channel_rows in channels.values():
                    target = channel_rows.copy()
                    target["ordinal"] = target.groupby(
                        keys[:-1], dropna=False
                    ).cumcount()
                    channel_rows[field + "_fret"] = values.reindex(
                        pd.MultiIndex.from_frame(target[keys])
                    ).to_numpy()
                if pair[0] != "dd":
                    bg_value = (
                        aligned[bg_field].to_numpy()
                        - left[bg_field].to_numpy()
                    ) / aligned[bg_field].to_numpy()
                    bg_values = pd.Series(
                        bg_value, index=pd.MultiIndex.from_frame(left[keys])
                    )
                    for channel_rows in channels.values():
                        target = channel_rows.copy()
                        target["ordinal"] = target.groupby(
                            keys[:-1], dropna=False
                        ).cumcount()
                        channel_rows[field + "_fret_bg"] = bg_values.reindex(
                            pd.MultiIndex.from_frame(target[keys])
                        ).to_numpy()
    return output


def _track_partition(session, dataset, channel, settings, options):
    from molcore import TrackingSettings as LinkSettings

    acquisition, name = session.partition(dataset, channel)
    rows = session.query_localizations(dataset, channel)
    groups = [(None, rows.id.to_numpy(dtype=np.uint64))]
    if settings.segmentation_column:
        from shapely.geometry import Point, Polygon

        groups = []
        for index, coordinates in enumerate(settings.polygons):
            polygon = Polygon(coordinates)
            inside = [
                polygon.contains(Point(x, y))
                for x, y in zip(rows.x, rows.y, strict=True)
            ]
            groups.append((index, rows.id.to_numpy(dtype=np.uint64)[inside]))
    memberships, labels = [], {}
    used = set()
    for label, ids in groups:
        ids = np.asarray(
            [i for i in ids if int(i) not in used], dtype=np.uint64
        )
        if not len(ids):
            continue
        result = tracking.track(
            session.model,
            acquisition,
            name,
            ids=ids,
            settings=LinkSettings(
                search_range=settings.search_range,
                memory=settings.memory,
                min_length=settings.min_length,
                frame_clock="channel",
            ),
            apply=False,
            **options,
        )
        memberships.extend(result.memberships)
        for members in result.memberships:
            used.update(map(int, members))
            if label is not None:
                labels.update((int(i), label) for i in members)
    return tuple(memberships), labels


def _publish_workflow(session, pending, *, cancel_event):
    prepared, request = pending.prepared, pending.prepared.request
    settings = request.settings
    if request.operation == "render":
        validate_expected(session.model, prepared.expected)
        return pending.results

    def commit(model):
        staged = ProjectSession(model)
        for p, binding in zip(
            request.partitions, prepared.bindings, strict=True
        ):
            staged.bind(*p, *binding)
        for dataset, channel, value in pending.results:
            _cancel(cancel_event)
            acquisition, name = staged.partition(dataset, channel)
            if request.operation == "photometry":
                metrics = {
                    k: value[k].to_numpy()
                    for k in value
                    if k.startswith("pixel_") or "_fret" in k
                }
                if metrics and len(value):
                    model.localizations.update_measurements(
                        value.id.to_numpy(dtype=np.uint64), **metrics
                    )
            elif request.operation == "registration":
                calibration_name = (
                    f"moltrack-registration:{acquisition}:{name}"
                )
                model.transforms.add_calibration(
                    Calibration(calibration_name, value)
                )
                model.transforms.select(acquisition, name, calibration_name)
            elif request.operation == "drift":
                model.transforms.set_drift(
                    acquisition, value[1], frames=value[0]
                )
            elif request.operation == "segmentation":
                for segmentation in value.values():
                    model.add_segmentation(segmentation)
            elif request.operation == "tracking":
                memberships, labels = value
                detections = staged.query_localizations(
                    dataset, channel, space="raw"
                ).id.to_numpy(dtype=np.uint64)
                track_ids = model.tracks.replace(
                    acquisition,
                    name,
                    list(memberships),
                    ids=detections,
                    metadata={"origin": "detection"},
                )
                preserved = model.tracks.query(
                    acquisition_id=acquisition, channel=name, space="raw"
                )
                preserved = {
                    k: v[~np.isin(preserved["track_id"], track_ids)]
                    for k, v in preserved.items()
                }
                occupied = set(
                    pd.to_numeric(
                        pd.Series(
                            [
                                model.tracks.metadata[int(track)][
                                    "parameters"
                                ].get("particle", int(track))
                                for track in np.unique(preserved["track_id"])
                            ]
                        ),
                        errors="coerce",
                    ).dropna()
                )
                particle = 1
                for track_id in track_ids:
                    while particle in occupied:
                        particle += 1
                    model.tracks.metadata[int(track_id)]["parameters"].update(
                        particle=particle
                    )
                    occupied.add(particle)
                    particle += 1
                linked = (
                    np.concatenate(memberships)
                    if memberships
                    else np.array([], dtype=np.uint64)
                )
                if labels:
                    ids = np.array(list(labels), dtype=np.uint64)
                    model.localizations.update_measurements(
                        ids,
                        **{
                            settings.segmentation_column: np.array(
                                list(labels.values()), dtype=int
                            )
                        },
                    )
                if settings.remove_unlinked:
                    model.localizations.delete(
                        detections[~np.isin(detections, linked)]
                    )
            # App-captured settings use a per-acquisition namespace so native
            # source cascade deletion removes provenance. Native photometry
            # run records retain their own schema and are never overwritten.
            model.processing.settings.setdefault(
                "moltrack_operations", {}
            ).setdefault(acquisition, {}).setdefault(name, {})[
                request.operation
            ] = {"settings": asdict(settings)}
        clear_statistics(model)
        _cancel(cancel_event)

    transact(session.model, commit, expected=prepared.expected)
    return pending.results
